import os
import sys
import pathlib
import torch
import torch.distributed as dist

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import dill


def init_distributed_mode(cfg):
    """Initialize the distributed training environment."""
    if dist.is_available():
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Set device for each process
    torch.cuda.set_device(rank)
    return rank, world_size


@hydra.main(config_path=os.path.join(str(pathlib.Path(__file__).parent), 'training/interagent/cfg'))
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Initialize distributed mode
    rank, world_size = init_distributed_mode(cfg)

    # Create the workspace
    if cfg.training.resume:
        print("-------------------------starting !-------------")
        checkpoint_path = cfg.training.resume_path
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg_load = payload['cfg']
        cfg_load.dataset._target_= 'training.interagent.dataset.dataset_from_lmdb.DiffusionPolicyDataset'
        cfg_load.dataset.motion_path = '/data/dataset_0525version/data.lmdb'
        cfg_load.dataset.obs_path = '/data/dataset_0525version/obs'
        cfg_load.dataset.action_path = '/data/dataset_0525version/action'

        cls = hydra.utils.get_class(cfg_load._target_)
        workspace = cls(cfg_load)
        workspace.load_payload(payload)
        print("-------------------------load checkpoint-------------:",checkpoint_path)

    else:
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
