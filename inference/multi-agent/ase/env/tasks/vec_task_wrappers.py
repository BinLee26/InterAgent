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

from gym import spaces
import numpy as np
import torch
from env.tasks.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

class VecTaskCPUWrapper(VecTaskCPU):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations, clip_actions)
        return

class VecTaskGPUWrapper(VecTaskGPU):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        return


class VecTaskPythonWrapper(VecTaskPython):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)

        self._amp_obs_space = spaces.Box(np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf)
        #print('--------', task.get_num_amp_obs())
        return

    def reset(self, env_ids=None):
        #print(env_ids)
        self.task.reset(env_ids)
        num_envs = self.task.obs_buf.shape[0]
        num_agents = self.task.obs_buf.shape[1]
        obs_dim = self.task.obs_buf.shape[-1]
        enable_hist_obs = self.task.get_enable_hist_obs_bool()
        if not enable_hist_obs:
            reset_obs_buf = self.task.obs_buf#.reshape(num_envs*num_agents, -1)
        else:
            obs_step = self.task.obs_buf.shape[-2]
            reset_obs_buf = self.task.obs_buf#.reshape(num_envs*num_agents, obs_step*obs_dim)
        # reset_ig_obs_buf = self.task.intergraph_obs_buf#.reshape(num_envs*num_agents, -1)
        reset_rigid_body_pos_buf = self.task.rigid_body_pos_t
        reset_rigid_body_rot_buf = self.task.rigid_body_rot_t
        
        return torch.clamp(reset_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), torch.clamp(reset_rigid_body_pos_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), torch.clamp(reset_rigid_body_rot_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @property
    def amp_observation_space(self):
        return self._amp_obs_space
    
    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)
    
    def fetch_amp_global_obs_demo(self, num_samples, seq_len):
        return self.task.fetch_amp_global_obs_demo(num_samples, seq_len)

    def fetch_amp_obs_demo_with_id(self, num_samples, motion_id=None):
        return self.task.fetch_amp_obs_demo_with_id(num_samples, motion_id)
    
    def fetch_annots(self):
        return self.task.fetch_annots()
    
    def sample_motion_ids(self, n):
        return self.task.sample_motion_ids(n)
    
    def get_flat_future_ref_obs(self):
        return self.task.get_flat_future_ref_obs()