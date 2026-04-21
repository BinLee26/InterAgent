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
import numpy as np

from env.tasks.humanoid_il_dm_v2 import HumanoidIlDMv2
from utils.motion_lib_il_weighted import MotionLibIlWeighted

class HumanoidIlDMv3(HumanoidIlDMv2):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # Add tracking for current frame indices
        self._current_frame_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Value-based sampling parameters
        self._value_alpha = cfg["env"].get("value_alpha", 0.5)  # Value update rate
        self._value_min = cfg["env"].get("value_min", 0.01)  # Minimum value for numerical stability
        
        # Initialize reward tracking
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 添加监控相关的配置
        self._enable_value_monitoring = cfg["env"].get("enable_value_monitoring", True)
        self._value_stats_interval = cfg["env"].get("value_stats_interval", 1000)  # 每隔多少步记录一次统计信息
        self._value_monitoring_motion_ids = cfg["env"].get("value_monitoring_motion_ids", None)  # 指定要监控的motion ids
        
    def _load_motion(self, motion_file):
        """Override to use MotionLibIlWeighted"""
        self._motion_lib = MotionLibIlWeighted(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
        
    def _reset_ref_state_init(self, env_ids):
        """Override to use value-based frame sampling"""
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        # Sample initial frames based on values
        motion_times = torch.zeros(num_envs, device=self.device)
        frame_indices = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        
        for i, motion_id in enumerate(motion_ids):
            time, frame_idx = self._motion_lib.sample_frame(motion_id)
            motion_times[i] = time
            frame_indices[i] = frame_idx
            
        self._current_frame_indices[env_ids] = frame_indices
        
        # Reset episode tracking for these envs
        self._episode_rewards[env_ids] = 0
        self._episode_lengths[env_ids] = 0

        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
               = self._motion_lib.get_dual_state(motion_ids, motion_times)
               
        root_pos = all_pos[...,0,:]
        root_rot = all_rot[...,0,:]
        root_pos[...,:,2] = self._initial_humanoid_root_states[env_ids,:,2] # + self.height_offset
        
        self._set_env_state(env_ids=env_ids, 
                root_pos=root_pos, 
                root_rot=root_rot, 
                dof_pos=dof_pos, 
                root_vel=root_vel, 
                root_ang_vel=root_ang_vel, 
                dof_vel=dof_vel)
            
        self._motion_start_times[env_ids,0] = motion_times
        self._motion_start_times[env_ids,1] = motion_times
        self._motion_ids[env_ids] = motion_ids
        return
        
    def _compute_reward(self, actions):
        """Override to track rewards"""
        super()._compute_reward(actions)
        
        # Track episode rewards
        self._episode_rewards += self.rew_buf.mean(dim=-1)  # Average reward across agents
        self._episode_lengths += 1
        return
        
    def _compute_reset(self):
        """Override to update frame values before reset"""
        # Get reset and termination info
        super()._compute_reset()
        
        # Update frame values for environments that need reset
        reset_mask = (self.reset_buf[:,0] == 1)
        # print("reset_mask.shape", reset_mask.shape)
        if reset_mask.any():
            reset_envs = torch.where(reset_mask)[0]
            #print("reset_envs.shape", torch.where(reset_mask), reset_envs)   
            # 获取需要重置的环境的motion_ids和frame_indices
            motion_ids = self._motion_ids[reset_envs]
            frame_indices = self._current_frame_indices[reset_envs]
            
            # 计算平均奖励
            valid_mask = self._episode_lengths[reset_envs] > 0
            if valid_mask.any():
                valid_envs = reset_envs[valid_mask]
                avg_rewards = self._episode_rewards[valid_envs] / self._episode_lengths[valid_envs]
                
                # 批量更新frame values
                self._motion_lib.update_frame_values_batch(
                    motion_ids[valid_mask],
                    frame_indices[valid_mask],
                    avg_rewards
                )
            
            # 重置episode统计信息
            self._episode_rewards[reset_envs] = 0
            self._episode_lengths[reset_envs] = 0
        return