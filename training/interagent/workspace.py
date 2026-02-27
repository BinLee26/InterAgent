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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from einops import rearrange, reduce
from functools import partial
from tqdm import tqdm

from pdp.policy import DiffusionPolicy
from pdp.dataset.dataset import DiffusionPolicyDataset
from pdp.utils.common import get_scheduler
from pdp.utils.data import dict_apply
from pdp.utils.ema_model import EMAModel

from torch.utils.tensorboard import SummaryWriter


class DiffusionPolicyWorkspace:
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        if cfg.training.debug:
            cfg.dataloader.num_workers = 0

        self.cfg = cfg

        # Initialize distributed process group (for multi-GPU)
        # if dist.is_available() and dist.is_initialized():
        self.rank = dist.get_rank()  # 当前进程的 rank
        print("self.rank:",self.rank)
        self.world_size = dist.get_world_size()  # 总进程数
        self.device = torch.device('cuda',self.rank)

        # Set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        optim_groups = self.model.get_optim_groups(weight_decay=cfg.optimizer.weight_decay)
        self.optimizer = torch.optim.AdamW(optim_groups, **cfg.optimizer)

        # Configure dataset and dataloader with DistributedSampler for multi-GPU
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        self.train_dataloader = DataLoader(dataset, sampler=self.train_sampler, **cfg.dataloader)


        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.set_normalizer(normalizer)
        self.model.to(self.device)      # Send model to GPU after setting normalizer
        self.ema_model.to(self.device)

        self.global_step = 0
        self.epoch = 0

        # Wrap model with DistributedDataParallel (DDP)
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank], output_device=self.rank,find_unused_parameters=False #True
            )
            self.ema_model = torch.nn.parallel.DistributedDataParallel(
                self.ema_model, device_ids=[self.rank], output_device=self.rank,find_unused_parameters=False #True
            )

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
                    payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                else:
                    payload['state_dicts'][key] = value.state_dict()

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
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
        load_loop = partial(tqdm, position=1, desc=f"Batch", leave=False, mininterval=cfg.training.tqdm_interval_sec)

        if cfg.training.logging and self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir=str(self.output_dir))


        # NOTE: The lr_scheduler is implemented as a pyorch LambdaLR scheduler. We step the learning rate at every
        # batch, so num_training_steps := len(train_dataloader) * cfg.training.num_epochs
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(self.train_dataloader) * cfg.training.num_epochs,
            last_epoch=self.global_step-1
        )

        if cfg.training.use_ema:
            ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.rollout_every = 1
            cfg.training.val_every = 1
            cfg.training.save_checkpoint_every = 1

        for local_epoch_idx in range(cfg.training.num_epochs):
            total_train_losses = list()
            proprioception_train_losses = list()
            exteroception_train_losses = list()
            action_train_losses = list()
            self.train_sampler.set_epoch(local_epoch_idx)
            for batch_idx, batch in enumerate(load_loop(self.train_dataloader)):
                batch = dict_apply(batch, lambda x: x.to(self.device))

                total_loss, proprioception_loss,exteroception_loss, action_loss = self.model(self.epoch, batch)

                self.optimizer.zero_grad()
                total_loss.backward()
                #grad cut ----------------------
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                lr_scheduler.step()
                if cfg.training.use_ema:
                    ema.step(self.model)

                total_loss_cpu = total_loss.item()
                total_train_losses.append(total_loss_cpu)

                proprioception_loss_cpu = proprioception_loss.item()
                proprioception_train_losses.append(proprioception_loss_cpu)

                exteroception_loss_cpu = exteroception_loss.item()
                exteroception_train_losses.append(exteroception_loss_cpu)

                action_loss_cpu = action_loss.item()
                action_train_losses.append(action_loss_cpu)
                self.global_step += 1
                if cfg.training.logging and self.rank == 0:
                    self.tb_writer.add_scalar("total_loss", total_loss_cpu, self.global_step)
                    self.tb_writer.add_scalar("proprioception_loss", proprioception_loss_cpu, self.global_step)
                    self.tb_writer.add_scalar("exteroception_loss", exteroception_loss_cpu, self.global_step)
                    self.tb_writer.add_scalar("action_loss", action_loss_cpu, self.global_step)
                    self.tb_writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], self.global_step)
                    self.tb_writer.add_scalar("epoch", self.epoch, self.global_step)

            self.epoch += 1
            self.post_step(locals())


    def post_step(self, locs):
        cfg = locs['cfg']
        if cfg.training.logging and self.rank == 0:
            epoch_mean_total_loss = np.mean(locs['total_train_losses'])
            self.tb_writer.add_scalar("epoch_total_train_loss", epoch_mean_total_loss, self.epoch)
            epoch_mean_proprioception_loss = np.mean(locs['proprioception_train_losses'])
            self.tb_writer.add_scalar("epoch_proprioception_train_loss", epoch_mean_proprioception_loss, self.epoch)
            epoch_mean_exteroception_loss = np.mean(locs['exteroception_train_losses'])
            self.tb_writer.add_scalar("epoch_exteroception_train_loss", epoch_mean_exteroception_loss, self.epoch)
            epoch_mean_action_loss = np.mean(locs['action_train_losses'])
            self.tb_writer.add_scalar("epoch_action_train_loss", epoch_mean_action_loss, self.epoch)

        if (
                self.epoch % cfg.training.save_checkpoint_every == 0 or
                self.epoch == cfg.training.num_epochs
        ) and self.rank == 0:
            self.save_checkpoint(tag=f'checkpoint_epoch_{self.epoch}')

