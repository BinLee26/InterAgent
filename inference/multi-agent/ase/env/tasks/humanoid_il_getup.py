# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils.motion_lib_il import MotionLibIl
from utils.motion_lib_il_ig import MotionLibIlIg

from env.tasks.humanoid_il_dm_v2 import HumanoidIlDMv2
# from env.tasks.humanoid_il_dm_v3 import HumanoidIlDMv3
from env.tasks.humanoid_il_ig_v2 import HumanoidIlIgv2
# from env.tasks.humanoid_il_ig_v3 import HumanoidIlIgv3
from env.tasks.humanoid_amp_task_dual import HumanoidAMPTaskDual
from utils import torch_utils

import numpy as np

class HumanoidIlGetup(HumanoidIlDMv2):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        #self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recovery_steps"]
        self._fall_init_prob = cfg["env"]["fall_init_prob"]

        #self._reset_fall_env_ids = []

        #self.availalbe_fall_states = torch.zeros(cfg["env"]['num_envs']).long().to(device_id)
        #self.fall_id_assignments = torch.zeros(cfg["env"]['num_envs']).long().to(device_id)
        #self.getup_udpate_epoch = cfg['env'].get("getup_udpate_epoch", 10000)

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_recovery_count()

        return


    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        fall_probs = to_torch(np.array([self._fall_init_prob] * num_envs), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = env_ids[fall_mask]
        
        if (len(fall_ids) > 0):
            self._reset_fall_episode(fall_ids)  # here we set them to default t-pose

        nonfall_ids = env_ids[torch.logical_not(fall_mask)]
        
        if (len(nonfall_ids) > 0):
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return

    def _reset_fall_episode(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        motion_times0 = torch.zeros(num_envs, device=self.device)
        
        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
               = self._motion_lib.get_dual_state(motion_ids, motion_times0)
        #print('+++++',all_pos.shape)
        ref_root_pos = all_pos[...,0,:]
        ref_root_rot = all_rot[...,0,:]
        
        init_root_pos, init_root_rot = self.compute_face_to_face_init(ref_root_pos, ref_root_rot)
        
        self._humanoid_root_states[env_ids, :, 0:2] = init_root_pos[...,:,:2]
        self._humanoid_root_states[env_ids, :, 2] = self._initial_humanoid_root_states[env_ids, :, 2]
        self._humanoid_root_states[env_ids, :, 3:7] = init_root_rot
        #self._humanoid_root_states[env_ids, :, 3:7] = self._initial_humanoid_root_states[env_ids, :, 3:7]
        self._humanoid_root_states[env_ids, :, 7:13] = 0
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        self._recovery_counter[env_ids] = self._recovery_steps
        self._motion_start_times[env_ids,0] = motion_times0
        self._motion_start_times[env_ids,1] = motion_times0
        self._motion_ids[env_ids] = motion_ids
        return

    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)

        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if (len(self._reset_fall_env_ids) > 0):
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        recovered_mask = (self.reset_buf[is_recovery] == 0)
        #print(recovered_mask)
        
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        self.progress_buf[is_recovery] -= 1
        return
    
    def compute_face_to_face_init(self, root_pos, root_rot, distance=1.5, alpha=0.5):
        pos1_ref = root_pos[:, 0, :]  # (N, 3)
        pos2_ref = root_pos[:, 1, :]  # (N, 3)
        # rot1_ref = root_rot[:, 0, :]  # (N, 4)
        # rot2_ref = root_rot[:, 1, :]  # (N, 4)

        center = (pos1_ref + pos2_ref) / 2  # (N, 3)
        direction = pos2_ref - pos1_ref
        direction[:, 2] = 0.0  # ignore vertical
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)  # (N, 3)

        offset = direction * (distance / 2)
        pos1 = center - offset
        pos2 = center + offset

        # face-to-face yaw
        yaw = torch.atan2(direction[:, 1], direction[:, 0])        # (N,)
        yaw_inv = torch.atan2(-direction[:, 1], -direction[:, 0])  # (N,)
        rot1_init = torch_utils.quat_from_yaw(yaw)       # (N, 4)
        rot2_init = torch_utils.quat_from_yaw(yaw_inv)   # (N, 4)
        
        init_pos = torch.stack([pos1, pos2], dim=1)  # (N, 2, 3)
        init_rot = torch.stack([rot1_init, rot2_init], dim=1)
        
        return init_pos, init_rot