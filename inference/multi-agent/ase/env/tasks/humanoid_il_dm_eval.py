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


from env.tasks.humanoid_amp_dual import HumanoidAMPDual
from env.tasks.humanoid_amp_task_dual import HumanoidAMPTaskDual
from utils import torch_utils

# from utils.misc.interaction import Interaction
# from fairmotion.utils import utils,constants
# from fairmotion.ops import conversions, math
# from fairmotion.ops import quaternion

import numpy as np
from scipy.sparse import coo_matrix
import copy

class HumanoidIlDMEval(HumanoidAMPTaskDual):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt
        # cfg["env"]["controlFrequencyInv"] = 1
        # cfg["env"]["pdControl"] = False
        self._recovery_steps = cfg["env"]["recovery_steps"]
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self._motion_start_times = torch.zeros((self.num_envs,self.num_agents),device=self.device)
        self.reward_specs = cfg["env"].get("reward_specs", {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1})
        self.power_reward = cfg["env"].get("power_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0001)
        # self.adjust_motion_weights = cfg['env']['adjust_motion_weights']
        self.avg_terminate = False
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            cur_ref_times = self.progress_buf * self._motion_dt + self._motion_start_times
            cur_ref_times = cur_ref_times[:,0]
            obs = torch.zeros((self.num_envs, self.num_agents, self.get_task_obs_size()), device=self.device, dtype=torch.float)
            for i in range(self._task_ref_steps):
                time_offset = i*self._motion_dt
                    
                ref_all_pos, ref_rot, ref_vel, ref_ang_vel,_ \
                    = self._motion_lib.get_il_state(self._motion_ids, cur_ref_times + time_offset)
                #print('******',ref_all_pos.shape)
                cur_ref_obs = self.compute_ref_observations(body_pos, body_rot, body_vel, body_ang_vel,ref_all_pos,ref_rot,ref_vel,ref_ang_vel)
                obs[...,i*self._task_obs_size_per_step:(i+1)*self._task_obs_size_per_step] = cur_ref_obs.clone()
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            cur_ref_times = self.progress_buf[env_ids] * self._motion_dt + self._motion_start_times[env_ids]
            cur_ref_times = cur_ref_times[:,0]
            obs = torch.zeros((env_ids.size(0), self.num_agents, self.get_task_obs_size()), device=self.device, dtype=torch.float)
            for i in range(self._task_ref_steps):
                time_offset = i*self._motion_dt
                    
                ref_all_pos, ref_rot, ref_vel, ref_ang_vel,_ \
                    = self._motion_lib.get_il_state(self._motion_ids[env_ids], cur_ref_times + time_offset)
                #print('******',ref_all_pos.shape)
                cur_ref_obs = self.compute_ref_observations(body_pos, body_rot, body_vel, body_ang_vel, ref_all_pos,ref_rot,ref_vel,ref_ang_vel)
                obs[...,i*self._task_obs_size_per_step:(i+1)*self._task_obs_size_per_step] = cur_ref_obs.clone()            
        return obs
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_recovery_count()

        return

    def post_physics_step(self):
        super().post_physics_step()
        # self._motion_sync()
        self.extras["curr_motion_ids"] = self._motion_ids
        return
    
    # def _get_humanoid_collision_filter(self):
    #     return 0 # disable self collisions
    
    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLibIl(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    
    def _reset_actors(self, env_ids):
        # print('+++++++++++resetting')
        if (self._recovery_steps > 0):
            self._reset_fall_episode(env_ids)
        elif (self._state_init == HumanoidAMPDual.StateInit.Start
              or self._state_init == HumanoidAMPDual.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_fall_episode(self,env_ids):
        num_envs = env_ids.shape[0]
        if self.collect_dataset:
            motion_ids = self._motion_lib.sample_motions_by_cur_idx(num_envs)
        else:
            motion_ids = self._motion_lib.sample_motions_by_order(num_envs)
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
        
        
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        if self.collect_dataset:
            motion_ids = self._motion_lib.sample_motions_by_cur_idx(num_envs)
        else:
            motion_ids = self._motion_lib.sample_motions_by_order(num_envs)
        #print(motion_ids)
        #print('start phase:', motion_times[0], self._motion_lib.get_motion_length(self._motion_ids[0]))
        if self._eval_state_init == HumanoidAMPDual.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        elif self._eval_state_init == HumanoidAMPDual.StateInit.Random:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            raise NotImplementedError

        all_pos, all_rot, root_vel, root_ang_vel, dof_vel, dof_pos, key_pos \
               = self._motion_lib.get_dual_state(motion_ids, motion_times)
        #print('+++++',all_pos.shape)
        root_pos = all_pos[...,0,:]
        root_rot = all_rot[...,0,:]
        #print('+++checkheight',root_pos[0,0,2],self._initial_humanoid_root_states[0,0,2])        
        root_pos[...,:,2] = self._initial_humanoid_root_states[env_ids,:,2]

        self._set_env_state(env_ids=env_ids, 
                root_pos=root_pos, 
                root_rot=root_rot, 
                dof_pos=dof_pos, 
                root_vel=root_vel, 
                root_ang_vel=root_ang_vel, 
                dof_vel=dof_vel)
            
        # if env_ids[0]==0:
        #     print('++++start',motion_times[0])
        #print('***********',self.ref_obs_buf.shape,self.ref_obs_buf[0,0,:10],self.ref_obs_buf[0,0,448:458])
        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids,0] = motion_times
        self._motion_start_times[env_ids,1] = motion_times
        self._motion_ids[env_ids] = motion_ids
        return
    
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            #print('++++++++1',self._rigid_body_pos[0,0,0:10],self._dof_pos[0,0,0:10])
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self.gym.simulate(self.sim)
            self._refresh_sim_tensors()
            #self._compute_observations(env_ids)
            self._init_obs(env_ids)
            
        if self.collect_dataset:
            #print('++++++dubug',self.obs_buf[0,0,:10])
            self.obs_buf_t = self.obs_buf.cpu().numpy()
            self.intergraph_obs_buf_t = self.intergraph_obs_buf.cpu().numpy()
            self.rigid_body_pos_t = self._rigid_body_pos.cpu().numpy()
            self.rigid_body_rot_t = self._rigid_body_rot.cpu().numpy()
        self.rigid_body_pos_t = copy.deepcopy(self._rigid_body_pos)
        self.rigid_body_rot_t = copy.deepcopy(self._rigid_body_rot)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, :, 0:3] = root_pos
        self._humanoid_root_states[env_ids, :, 3:7] = root_rot
        self._humanoid_root_states[env_ids, :, 7:10] = root_vel
        self._humanoid_root_states[env_ids, :, 10:13] = root_ang_vel
        #print("humanoid root states CHECK:", self._humanoid_root_states[env_ids, 0, 0:3].shape)
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return
    
    def _reset_env_tensors(self, env_ids):
        #num_motions = self._motion_lib.num_motions()
        #self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        super()._reset_env_tensors(env_ids)
        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0
        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return
    
    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)

        #print('+++curmotion',self._motion_lib.get_motion_name(self._motion_ids[0]))
        #print('+++++num_frames',self._motion_lib.get_motion_num_frames(self._motion_ids[0]))
        motion_times = self.progress_buf * self.dt + self._motion_start_times
        motion_times = motion_times[:,0]
        ref_all_pos, ref_rot, ref_vel, ref_ang_vel,_ \
            = self._motion_lib.get_il_state(self._motion_ids, motion_times)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_start_times, self._motion_dt,
                                                    self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, ref_all_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_distances, self._termination_heights,self.avg_terminate)
        
        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        self.progress_buf[is_recovery] -= 1
        return       

    def compute_ref_observations(self, body_pos, body_rot, body_vel, body_ang_vel, ref_pos, ref_rot, ref_vel, ref_ang_vel):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor) -> Tensor
        obses = []
        for i in range(2):
            if self.obs_v == 2:
                self_obs = compute_imitation_observations_self_v2(body_pos[:, i], body_rot[:, i], body_vel[:, i], body_ang_vel[:, i], \
                                                ref_pos[:, i], ref_rot[:, i], ref_vel[:, i], ref_ang_vel[:, i],)
            elif self.obs_v == 3:
                self_obs = compute_imitation_observations_self_v3(body_pos[:, i], body_rot[:, i], body_vel[:, i], body_ang_vel[:, i], \
                                                ref_pos[:, i], ref_rot[:, i], ref_vel[:, i], ref_ang_vel[:, i],)            
            elif self.obs_v == 4:
                self_obs = compute_imitation_observations_self_v4(body_pos[:, i], body_rot[:, i], body_vel[:, i], body_ang_vel[:, i], \
                                                ref_pos[:, i], ref_rot[:, i], ref_vel[:, i], ref_ang_vel[:, i])
            else:
                raise NotImplementedError
            obs = self_obs
            obses.append(obs)

        obs = torch.cat([obs.unsqueeze(1) for obs in obses], dim=1)
        return obs
    
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

    # def _compute_reward(self, actions):
    #     #print('++++++env0 imitating',self._motion_ids[0])
    #     #print('+++',self.progress_buf[0][0])
    #     #motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
    #     cur_ref_times = self.progress_buf * self._motion_dt + self._motion_start_times
    #     #assert cur_ref_times[:,0] == cur_ref_times[:,1]
    #     cur_ref_time = cur_ref_times[:,0]
    #     ref_all_pos, ref_rot, ref_vel, ref_ang_vel,ref_key_pos \
    #         = self._motion_lib.get_il_state(self._motion_ids, cur_ref_time)
            
    #     #ref_ig_edge_indices = self._motion_lib.get_ig_edge_indices(self._motion_ids, cur_ref_time)
    #     #print('++++ref_ig_edge',len(ref_ig_edge_indices))
    #     ref_root_pos = ref_all_pos[..., 0, :]
    #     sim_root_pos = self._rigid_body_pos[..., 0, :]
    #     sim_key_pos = self._rigid_body_pos[..., self._key_body_ids, :]
    #     sim_rot = self._rigid_body_rot
    #     sim_vel = self._rigid_body_vel
    #     sim_ang_vel = self._rigid_body_ang_vel
    #     sim_all_pos = self._rigid_body_pos

    #     #print('++test_height',ref_root_pos[0,0,2],sim_root_pos[0,0,2])
    #     final_rew, final_rew_raw = compute_imitation_reward(sim_all_pos, sim_rot, sim_vel, sim_ang_vel, \
    #                 ref_all_pos, ref_rot, ref_vel, ref_ang_vel, self.reward_specs)
        
    #     #print('+++rew',final_rew.shape)
    #     self.rew_buf = final_rew
    #     #print('+++rew',self.rew_buf)
    #     if self.power_reward:
    #         force_tensor = self.dof_force_tensor.reshape(self.num_envs, self.num_agents,-1)
    #         power = torch.abs(torch.multiply(force_tensor, self._dof_vel)).sum(dim=-1) 
    #         # power_reward = -0.00005 * (power ** 2)
    #         power_reward = -self.power_coefficient * power
    #         power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

    #         self.rew_buf[:] += power_reward
    #         #print('++++checkpower',power_reward.mean())
    #         #self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)     
    #     return
    
 
@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, motion_start_times, dt, contact_buf, contact_body_ids, rigid_body_pos, ref_pos, 
                           max_episode_length, enable_early_termination, termination_distance, termination_heights, avg_terminate):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + motion_start_times

    if (enable_early_termination):
        # if avg_terminate:
        #     use_mean = True
        #     if use_mean:
        #         has_fallen = torch.any(torch.norm(rigid_body_pos - ref_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0], dim=-1)  # using average, same as UHC"s termination condition
        #     else:
        #         has_fallen = torch.any(torch.norm(rigid_body_pos - ref_pos, dim=-1) > termination_distance, dim=-1)
        if True:
            masked_contact_buf = contact_buf.clone()
            masked_contact_buf[:,:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
            #print('+++++++fall',fall_contact.shape)
            fall_contact = torch.any(fall_contact, dim=-1)
            #print('+++++++fall1',fall_contact.shape)
            body_height = rigid_body_pos[..., 2]
            fall_height = body_height < termination_heights
            fall_height[:,:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)

            has_fallen = torch.logical_and(fall_contact, fall_height)
        # if one fallen, entire env should reset
        fall_mask = torch.any(has_fallen, dim=1, keepdim=True)
        has_fallen = torch.where(fall_mask, torch.tensor(True,device=reset_buf.device), has_fallen)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    motion_lengths = 10.0 * torch.ones_like(motion_lengths, device=motion_lengths.device)
    reset = torch.where(motion_times > motion_lengths.unsqueeze(1), torch.ones_like(reset_buf), torch.zeros_like(reset_buf))

    return reset, terminated

@torch.jit.script
def compute_imitation_reward(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)
    # import ipdb
    # ipdb.set_trace()
    return reward, reward_raw

    
@torch.jit.script
def compute_imitation_observations_self_v2(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    time_steps = 1
    obs = []
    B, J, _ = body_pos.shape

    root_pos = body_pos[..., 0, :]
    root_rot = body_rot[..., 0, :]
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(B, 1, J, 3)
    diff_local_body_pos_flat = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.reshape(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_rot.reshape(-1, 4)), heading_rot_expand.reshape(-1, 4))  # Need to be change of basis
    
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(B, 1, J, 3)
    diff_local_vel = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.reshape(B, time_steps, J, 3) - body_ang_vel.reshape(B, 1, J, 3)
    diff_local_ang_vel = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_ang_vel.reshape(-1, 3))
    

    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(B, 1, 1, 3)
    local_ref_body_pos = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), ref_body_rot.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    obs.append(diff_local_body_pos_flat.reshape(B, time_steps, -1))
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).reshape(B, time_steps, -1))
    obs.append(diff_local_ang_vel.reshape(B, time_steps, -1))
    obs.append(local_ref_body_pos.reshape(B, time_steps, -1))
    obs.append(local_ref_body_rot.reshape(B, time_steps, -1))

    obs = torch.cat(obs, dim=-1).reshape(B, -1)
    #print('++++++++imobs',obs.shape)
    return obs

@torch.jit.script
def compute_imitation_observations_self_v3(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    time_steps = 1
    obs = []
    B, J, _ = body_pos.shape

    root_pos = body_pos[..., 0, :]
    root_rot = body_rot[..., 0, :]
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(B, 1, J, 3)
    diff_local_body_pos_flat = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))
    
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(B, 1, J, 3)
    diff_local_vel = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3))


    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(B, 1, 1, 3)
    local_ref_body_pos = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3))

    obs.append(diff_local_body_pos_flat.reshape(B, time_steps, -1))
    obs.append(diff_local_vel.view(B, time_steps, -1))
    obs.append(local_ref_body_pos.reshape(B, time_steps, -1))

    obs = torch.cat(obs, dim=-1).reshape(B, -1)
    #print('++++++++imobs',obs.shape)
    return obs

@torch.jit.script
def compute_imitation_observations_self_v4(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    time_steps = 1
    obs = []
    B, J, _ = body_pos.shape

    root_pos = body_pos[..., 0, :]
    root_rot = body_rot[..., 0, :]
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(B, 1, J, 3)
    diff_local_body_pos_flat = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.reshape(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_rot.reshape(-1, 4)), heading_rot_expand.reshape(-1, 4))  # Need to be change of basis
    
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(B, 1, J, 3)
    diff_local_vel = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.reshape(B, time_steps, J, 3) - body_ang_vel.reshape(B, 1, J, 3)
    diff_local_ang_vel = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_ang_vel.reshape(-1, 3))
    

    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(B, 1, 1, 3)
    local_ref_body_pos = quat_rotate(heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), ref_body_rot.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    obs.append(diff_local_body_pos_flat.reshape(B, time_steps, -1))
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).reshape(B, time_steps, -1))
    obs.append(diff_local_vel.view(B, time_steps, -1))
    obs.append(diff_local_ang_vel.reshape(B, time_steps, -1))
    obs.append(local_ref_body_pos.reshape(B, time_steps, -1))
    obs.append(local_ref_body_rot.reshape(B, time_steps, -1))

    obs = torch.cat(obs, dim=-1).reshape(B, -1)
    #print('++++++++imobs',obs.shape)
    return obs

@torch.jit.script
def compute_opponent_observations(root_pos, root_rot, opp_body_pos, opp_body_rot, opp_body_vel, opp_body_ang_vel, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    op_root_pos = opp_body_pos[:, 0, :]
    # op_root_rot = body_rot[:, 0, :]

    root_h = op_root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # if (not root_height_obs):
    #     root_h_obs = torch.zeros_like(root_h)
    # else:
    #     root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, opp_body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    opp_local_body_pos = opp_body_pos - root_pos_expand
    opp_flat_local_body_pos = opp_local_body_pos.reshape(opp_local_body_pos.shape[0] * opp_local_body_pos.shape[1], opp_local_body_pos.shape[2])
    opp_flat_local_body_pos = quat_rotate(flat_heading_rot, opp_flat_local_body_pos)
    opp_local_body_pos = opp_flat_local_body_pos.reshape(opp_local_body_pos.shape[0], opp_local_body_pos.shape[1] * opp_local_body_pos.shape[2])
    #opp_local_body_pos = opp_local_body_pos[..., 3:] # remove root pos

    opp_flat_body_rot = opp_body_rot.reshape(opp_body_rot.shape[0] * opp_body_rot.shape[1], opp_body_rot.shape[2])
    opp_flat_local_body_rot = quat_mul(flat_heading_rot, opp_flat_body_rot)
    opp_flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(opp_flat_local_body_rot)
    opp_local_body_rot_obs = opp_flat_local_body_rot_obs.reshape(opp_body_rot.shape[0], opp_body_rot.shape[1] * opp_flat_local_body_rot_obs.shape[1])
    
    opp_flat_body_vel = opp_body_vel.reshape(opp_body_vel.shape[0] * opp_body_vel.shape[1], opp_body_vel.shape[2])
    opp_flat_local_body_vel = quat_rotate(flat_heading_rot, opp_flat_body_vel)
    opp_local_body_vel = opp_flat_local_body_vel.reshape(opp_body_vel.shape[0], opp_body_vel.shape[1] * opp_body_vel.shape[2])
    
    opp_flat_body_ang_vel = opp_body_ang_vel.reshape(opp_body_ang_vel.shape[0] * opp_body_ang_vel.shape[1], opp_body_ang_vel.shape[2])
    opp_flat_local_body_ang_vel = quat_rotate(flat_heading_rot, opp_flat_body_ang_vel)
    opp_local_body_ang_vel = opp_flat_local_body_ang_vel.reshape(opp_body_ang_vel.shape[0], opp_body_ang_vel.shape[1] * opp_body_ang_vel.shape[2])
    
    obs = torch.cat((opp_local_body_pos, opp_local_body_rot_obs, opp_local_body_vel, opp_local_body_ang_vel), dim=-1)
    #print('$$$$$$$$$$', obs.shape)
    return obs

