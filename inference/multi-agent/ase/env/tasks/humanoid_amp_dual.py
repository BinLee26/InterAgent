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

from enum import Enum
import numpy as np
import torch
import random

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid_dual import HumanoidDual, dof_to_obs
from utils import gym_util
from utils.motion_lib_dual import MotionLibDual
from isaacgym.torch_utils import *

from utils import torch_utils

class HumanoidAMPDual(HumanoidDual):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMPDual.StateInit[state_init]
        eval_state_init = cfg["env"]["eval_state_init"]
        self._eval_state_init = HumanoidAMPDual.StateInit[eval_state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        
        # self._amp_obs_buf = torch.zeros((self.num_envs, self.num_agents, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        # self._curr_amp_obs_buf = self._amp_obs_buf[:, :, 0]
        # self._hist_amp_obs_buf = self._amp_obs_buf[:, :, 1:]
        self._amp_obs_demo_buf = None
        
        return
    
    def post_physics_step(self):
        super().post_physics_step()
        
        # self._update_hist_amp_obs()
        # self._compute_amp_observations()

        # nenv, nactor,nstep, nobs_per_step = self._amp_obs_buf.size()
        #print('++++++++++',self._amp_obs_buf[:,0,:,:].shape)
        # amp_obs_flat = self._amp_obs_buf[:,0,:,:]
        # amp_obs_flat = self._amp_obs_buf.reshape(nenv*nactor,nstep*nobs_per_step)
        # self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_motions(self):
        return self._motion_lib.num_motions()
    
    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def get_num_amp_obs_steps(self):
        return self._num_amp_obs_steps
    
    def get_num_amp_obs_per_step(self):
        return self._num_amp_obs_per_step
        
    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        
        nenv, nstep, nobs_per_step = self._amp_obs_demo_buf.size()
        amp_obs_demo_flat = self._amp_obs_demo_buf.reshape(nenv,nstep*nobs_per_step)
        return amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0, global_seq_len = None):
        dt = self.dt
        if not global_seq_len:
            motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        else:
            motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, global_seq_len])
        motion_times = motion_times0.unsqueeze(-1)
        if not global_seq_len:
            time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        else:
            time_steps = -dt * torch.arange(0, global_seq_len, device=self.device)
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
               = self._motion_lib.get_dual_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_dual_observations(all_pos, all_rot, root_vel, root_ang_vel, dof_pos,  dof_vel,key_pos, 
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets, is_demo=True)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _build_amp_global_obs_demo_buf(self, num_samples, seq_len):
        self._amp_global_obs_demo_buf = torch.zeros((num_samples, seq_len, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            #self._num_amp_obs_per_step = 2 * (13 + self._dof_obs_size + 28 + 3 * num_key_bodies) + 1 + 4 + 4*2 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            #self._num_amp_obs_per_step = 2 * (13 + self._dof_obs_size + 28 + 3 * num_key_bodies) # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            #self._num_amp_obs_per_step = 2 * 163
            self._num_amp_obs_per_step = 125
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 2 * (13 + self._dof_obs_size + 31 + 3 * num_key_bodies) # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            print('++++++++', self._num_amp_obs_per_step)
            
        elif (asset_file == "mjcf/smpl_humanoid.xml"):
            self._num_amp_obs_per_step = 232
            
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLibDual(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        # self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMPDual.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMPDual.StateInit.Start
              or self._state_init == HumanoidAMPDual.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        # elif (self._state_init == HumanoidAMPDual.StateInit.Hybrid):
        #     self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._humanoid_root_states1[env_ids] = self._initial_humanoid_root_states1[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_pos1[env_ids] = self._initial_dof_pos1[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._dof_vel1[env_ids] = self._initial_dof_vel1[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMPDual.StateInit.Random
            or self._state_init == HumanoidAMPDual.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMPDual.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
    
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    # def _init_amp_obs_default(self, env_ids):
    #     curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
    #     self._hist_amp_obs_buf[env_ids] = curr_amp_obs
    #     return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
        #        = self._motion_lib.get_motion_state(motion_ids, motion_times)
        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
               = self._motion_lib.get_dual_state(motion_ids, motion_times)
               
        amp_obs_demo = build_amp_dual_observations(all_pos, all_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        # curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        # self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        # print('+++++++++initamp',curr_amp_obs.shape,curr_amp_obs[0,0:10])
        return
    
    #TODO: update dual
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        #print('++++++++++r',root_pos[:,0].shape, dof_pos.shape)
        print('&&&&&&&&&&&resetting env')
        print('debug0',self._humanoid_root_states[0, 0, 0:3], self._humanoid_root_states[0, 1, 0:3])
        self._humanoid_root_states[env_ids, :, 0:3] = root_pos
        print('debug1',self._humanoid_root_states[0, 0, 0:3], self._humanoid_root_states[0, 1, 0:3])
        self._humanoid_root_states[env_ids, :, 3:7] = root_rot
        self._humanoid_root_states[env_ids, :, 7:10] = root_vel
        self._humanoid_root_states[env_ids, :, 10:13] = root_ang_vel
        #print("humanoid root states CHECK:", self._humanoid_root_states[env_ids, 0, 0:3].shape)
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[2] - 1)):
                self._amp_obs_buf[:, :, i + 1] = self._amp_obs_buf[:, :, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, :, i + 1] = self._amp_obs_buf[env_ids, :, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[..., self._key_body_ids, :]
        
        if (env_ids is None):
            all_pos = self._rigid_body_pos
            all_rot = self._rigid_body_rot
            root_vel = self._rigid_body_vel[:,:,0,:]
            root_ang_vel = self._rigid_body_ang_vel[:,:,0,:]
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel

            curr_amp_obs = build_amp_dual_observations(all_pos, all_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                                            self._local_root_obs, self._root_height_obs, 
                                                            self._dof_obs_size, self._dof_offsets)
            
            self._curr_amp_obs_buf[:] = curr_amp_obs
                                                                    
        else:
            all_pos = self._rigid_body_pos[env_ids]
            all_rot = self._rigid_body_rot[env_ids]
            root_vel = self._rigid_body_vel[env_ids,:,0]
            root_ang_vel = self._rigid_body_ang_vel[env_ids,:,0]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = key_body_pos[env_ids]
            
            curr_amp_obs = build_amp_dual_observations(all_pos, all_rot, root_vel, root_ang_vel,dof_pos,  dof_vel, key_body_pos, 
                                                            self._local_root_obs, self._root_height_obs, 
                                                            self._dof_obs_size, self._dof_offsets)
            #print('++debugamp',curr_amp_obs.shape)
            self._curr_amp_obs_buf[env_ids] = curr_amp_obs
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def build_amp_dual_observations(all_poses, all_rots, root_vels, root_ang_vels, dof_poses, dof_vels, key_poses,
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor

    obs1 = build_amp_observations(all_poses[:,0,0], all_rots[:,0,0], root_vels[:,0], root_ang_vels[:,0], dof_poses[:,0], dof_vels[:,0], key_poses[:,0],
                                              local_root_obs, root_height_obs, dof_obs_size, dof_offsets)
    obs2 = build_amp_observations(all_poses[:,1,0], all_rots[:,1,0], root_vels[:,1], root_ang_vels[:,1], dof_poses[:,1], dof_vels[:,1], key_poses[:,1],
                                              local_root_obs, root_height_obs, dof_obs_size, dof_offsets)
    obs = torch.cat((obs1.unsqueeze(1), obs2.unsqueeze(1)),dim=1)
    #print('+++debugampp',obs1.shape,obs.shape)
    return obs