import os
import pathlib
import copy
import random
import threading

import wandb
import shutil
import hydra
import numpy as np
import torch
import dill
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange, reduce
from functools import partial
from tqdm import tqdm

from interagent.utils.common import get_scheduler
from interagent.utils.data import dict_apply
from interagent.utils.ema_model import EMAModel


# from accelerate import Accelerator, DistributedDataParallelKwargs
# from accelerate.state import AcceleratorState


class DiffusionPolicyWorkspace:
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        if cfg.training.debug:
            cfg.dataloader.num_workers = 0

        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        # TODO: Maybe use accelerate for distributed training
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # try split_batches = True as well... ``
        # state = AcceleratorState()
        # num_processes = state.num_processes
        # self.accelerator.wait_for_everyone()

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        optim_groups = self.model.get_optim_groups(weight_decay=cfg.optimizer.weight_decay)
        self.optimizer = torch.optim.AdamW(optim_groups, **cfg.optimizer)

        # Configure dataset and dataloader
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # assert dataset.horizon >= self.model.T_range

        # NOTE: In PDP we evaluated the validation set offline on various checkpoints
        # val_dataset = dataset.get_validation_dataset()
        # self.val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.set_normalizer(normalizer)
        self.model.to(self.device)  # Send model to GPU after setting normalizer
        self.ema_model.to(self.device)

        self.global_step = 0
        self.epoch = 0

        # TODO: Maybe use accelerate for distributed training
        # self.train_dataloader = self.accelerator.prepare(train_dataloader)
        # self.val_dataloader = self.accelerator.prepare(val_dataloader)
        # self.ema_model = self.accelerator.prepare(ema_model)
        # self.model = self.accelerator.prepare(model)
        # self.optimizer = self.accelerator.prepare(optimizer)

    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir

    @classmethod
    def create_from_checkpoint(cls, path, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(payload=payload, **kwargs)
        return instance

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def save_checkpoint(self, path=None, tag='latest', use_thread=True):
        def _copy_to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to('cpu')
            elif isinstance(x, dict):
                result = dict()
                for k, v in x.items():
                    result[k] = _copy_to_cpu(v)
                return result
            elif isinstance(x, list):
                return [_copy_to_cpu(k) for k in x]
            else:
                return copy.deepcopy(x)

        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if use_thread:
                    # payload['state_dicts'][key] = _copy_to_cpu(
                    #     self.accelerator.unwrap_model(value).state_dict())
                    payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                else:
                    # payload['state_dicts'][key] = self.accelerator.unwrap_model(value).state_dict()
                    payload['state_dicts'][key] = value.state_dict()

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

        return str(path.absolute())

    def load_payload(self, payload, **kwargs):
        for key, value in payload['state_dicts'].items():
            self.__dict__[key].load_state_dict(value, **kwargs)

    def load_checkpoint(self, path=None, tag='latest', **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload)
        return payload

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # load_loop = (
        #     partial(tqdm, position=1, desc=f"Batch", leave=False, mininterval=cfg.training.tqdm_interval_sec)
        #     if self.accelerator.is_main_process
        #     else lambda x: x
        # )
        load_loop = partial(tqdm, position=1, desc=f"Batch", leave=False, mininterval=cfg.training.tqdm_interval_sec)
        if cfg.training.logging:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # NOTE: The lr_scheduler is implemented as a pyorch LambdaLR scheduler. We step the learning rate at every
        # batch, so num_training_steps := len(train_dataloader) * cfg.training.num_epochs
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            # num_warmup_steps=cfg.training.lr_warmup_steps * self.accelerator.num_processes,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            # num_training_steps=len(self.train_dataloader) * cfg.training.num_epochs * self.accelerator.num_processes,
            num_training_steps=len(self.train_dataloader) * cfg.training.num_epochs,
            last_epoch=self.global_step - 1
        )
        # lr_scheduler = self.accelerator.prepare(lr_scheduler)

        if cfg.training.use_ema:
            ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            # ema = self.accelerator.prepare(ema)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.rollout_every = 1
            cfg.training.val_every = 1
            cfg.training.save_checkpoint_every = 1

        # self.accelerator.wait_for_everyone()
        for local_epoch_idx in range(cfg.training.num_epochs):
            train_losses = list()
            for batch_idx, batch in enumerate(load_loop(self.train_dataloader)):
                batch = dict_apply(batch, lambda x: x.to(self.device))
                loss = self.model(batch)
                # self.accelerator.backward(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                if cfg.training.use_ema:
                    ema.step(self.model)

                # if self.accelerator.is_main_process:
                loss_cpu = loss.item()
                train_losses.append(loss_cpu)
                step_log = {
                    'train_loss': loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }
                self.global_step += 1

            self.epoch += 1
            self.post_step(locals())

    def post_step(self, locs):
        # if self.accelerator.is_main_process:
        cfg = locs['cfg']

        if cfg.training.logging:
            wandb_run = locs['wandb_run']
            step_log = locs['step_log']
            step_log['train_loss'] = np.mean(locs['train_losses'])
            wandb_run.log(step_log, step=self.global_step)

        if (
                self.epoch % cfg.training.save_checkpoint_every == 0 or
                self.epoch == cfg.training.num_epochs
        ):
            self.save_checkpoint(tag=f'checkpoint_epoch_{self.epoch}')
