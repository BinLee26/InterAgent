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
from env.tasks.humanoid_il_getup import HumanoidIlGetup
from utils.motion_lib_il_weighted import MotionLibIlWeighted

class HumanoidIlGetupWeighted(HumanoidIlGetup):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_value_monitoring = cfg["env"].get("enable_value_monitoring", True)
        self._value_stats_interval = cfg["env"].get("value_stats_interval", 100)
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # 添加用于跟踪当前帧索引的属性
        self._current_frame_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 初始化奖励跟踪
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 记录哪些环境是通过随机帧初始化的（非fall初始化）
        self._random_init_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 设置matplotlib后端
        import matplotlib
        matplotlib.use('TkAgg')  # 或者使用 'Agg' 如果在无界面环境

    def _load_motion(self, motion_file):
        """Override to use MotionLibIlWeighted"""
        self._motion_lib = MotionLibIlWeighted(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        
        # 配置motion_lib的可视化参数
        self._motion_lib.enable_vis = self._enable_value_monitoring
        self._motion_lib.vis_freq = self._value_stats_interval
        return

    def _reset_actors(self, env_ids):
        """Override to track which envs are randomly initialized"""
        num_envs = env_ids.shape[0]
        fall_probs = torch.full((num_envs,), self._fall_init_prob, device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = env_ids[fall_mask]
        nonfall_ids = env_ids[torch.logical_not(fall_mask)]
        
        # 重置跟踪标志
        self._random_init_envs[env_ids] = False
        
        if len(fall_ids) > 0:
            self._reset_fall_episode(fall_ids)
            # fall初始化的环境重置其统计信息
            self._episode_rewards[fall_ids] = 0
            self._episode_lengths[fall_ids] = 0
        
        if len(nonfall_ids) > 0:
            # 标记这些环境是随机帧初始化的
            self._random_init_envs[nonfall_ids] = True
            # 在随机初始化之前重置recovery counter
            self._recovery_counter[nonfall_ids] = 0
            # 调用父类的随机帧初始化
            self._reset_actors_random_init(nonfall_ids)

    def _reset_actors_random_init(self, env_ids):
        """随机帧初始化的具体实现"""
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        # 基于value采样初始帧
        motion_times = torch.zeros(num_envs, device=self.device)
        frame_indices = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        
        for i, motion_id in enumerate(motion_ids):
            time, frame_idx = self._motion_lib.sample_frame(motion_id)
            motion_times[i] = time
            frame_indices[i] = frame_idx
        
        # 记录当前帧索引
        self._current_frame_indices[env_ids] = frame_indices
        
        # 重置episode统计信息
        self._episode_rewards[env_ids] = 0
        self._episode_lengths[env_ids] = 0
        
        # 设置运动状态
        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
            = self._motion_lib.get_dual_state(motion_ids, motion_times)
        
        ref_root_pos = all_pos[...,0,:]
        ref_root_rot = all_rot[...,0,:]
        ref_root_pos[...,:,2] = self._initial_humanoid_root_states[env_ids,:,2]
        
        self._set_env_state(env_ids=env_ids,
                           root_pos=ref_root_pos,
                           root_rot=ref_root_rot,
                           dof_pos=dof_pos,
                           root_vel=root_vel,
                           root_ang_vel=root_ang_vel,
                           dof_vel=dof_vel)
        
        self._motion_start_times[env_ids,0] = motion_times
        self._motion_start_times[env_ids,1] = motion_times
        self._motion_ids[env_ids] = motion_ids

    def _compute_reward(self, actions):
        """Override to track rewards"""
        super()._compute_reward(actions)
        
        # 只对随机帧初始化的环境跟踪奖励
        self._episode_rewards[self._random_init_envs] += self.rew_buf[self._random_init_envs].mean(dim=-1)
        self._episode_lengths[self._random_init_envs] += 1

    def _compute_reset(self):
        """Override to update frame values before reset"""
        super()._compute_reset()
        
        # 只更新随机帧初始化环境的frame values
        reset_mask = (self.reset_buf[:,0] == 1) & self._random_init_envs
        if reset_mask.any():
            reset_envs = torch.where(reset_mask)[0]
            
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