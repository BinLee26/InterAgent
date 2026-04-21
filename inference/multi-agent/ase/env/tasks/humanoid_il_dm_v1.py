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

from env.tasks.humanoid_amp_dual import HumanoidAMPDual
from env.tasks.humanoid_amp_task_dual import HumanoidAMPTaskDual
from utils import torch_utils

from utils.misc.interaction import Interaction
from fairmotion.utils import utils,constants
from fairmotion.ops import conversions, math
from fairmotion.ops import quaternion

import numpy as np
from scipy.sparse import coo_matrix


class HumanoidIlDMv1(HumanoidAMPTaskDual):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        # cfg["env"]["controlFrequencyInv"] = 1
        # cfg["env"]["pdControl"] = False

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
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.ref_obs_buf = torch.zeros(
                (self.num_envs, self.num_agents, self.get_task_obs_size()), device=self.device, dtype=torch.float)
        
        self.ig_reward = cfg["env"].get("ig_reward", False)
        if self.ig_reward:
            self._interaction_kernel = 10
            self._self_interaction_vert_cnt = 15
            self._oppo_interaction_vert_cnt = 15
            self._base_kin_ig = [None] * self.num_envs
            self._base_sim_ig = [None] * self.num_envs
            self.ig_coefficient = cfg["env"].get("ig_coefficient", 0.5)
        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3*self._num_obs
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            cur_ref_times = self.progress_buf * self._motion_dt + self._motion_start_times
            cur_ref_times = cur_ref_times[:,0]
            obs = torch.zeros((self.num_envs, self.num_agents, self.get_task_obs_size()), device=self.device, dtype=torch.float)
            for i in range(3):
                if i == 0:
                    time_offset = self._motion_dt
                elif i==1:
                    time_offset = 2*self._motion_dt
                else:
                    time_offset = 3*self._motion_dt
                    
                ref_all_pos, ref_rot, ref_vel, ref_ang_vel,_ \
                    = self._motion_lib.get_il_state(self._motion_ids, cur_ref_times + time_offset)
                #print('******',ref_all_pos.shape)
                cur_ref_obs = self.compute_ref_observations(ref_all_pos,ref_rot,ref_vel,ref_ang_vel)
                obs[...,i*self._num_obs:(i+1)*self._num_obs] = cur_ref_obs.clone()
        else:
            cur_ref_times = self.progress_buf[env_ids] * self._motion_dt + self._motion_start_times[env_ids]
            cur_ref_times = cur_ref_times[:,0]
            obs = torch.zeros((env_ids.size(0), self.num_agents, self.get_task_obs_size()), device=self.device, dtype=torch.float)
            for i in range(3):
                if i == 0:
                    time_offset = self._motion_dt
                elif i==1:
                    time_offset = 2*self._motion_dt
                else:
                    time_offset = 3*self._motion_dt
                    
                ref_all_pos, ref_rot, ref_vel, ref_ang_vel,_ \
                    = self._motion_lib.get_il_state(self._motion_ids[env_ids], cur_ref_times + time_offset)
                #print('******',ref_all_pos.shape)
                cur_ref_obs = self.compute_ref_observations(ref_all_pos,ref_rot,ref_vel,ref_ang_vel)
                obs[...,i*self._num_obs:(i+1)*self._num_obs] = cur_ref_obs.clone()            
        return obs
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        # self.actions = actions.to(self.device).clone()
        # #print("action shape check:", self.actions.shape)
        # forces = torch.zeros_like(self.actions)
        # force_tensor = gymtorch.unwrap_tensor(forces)
        # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        #print("pre_physics_step_CHECK: successful")
        return

    def post_physics_step(self):
        super().post_physics_step()
        # self._motion_sync()
        return
    
    def _get_humanoid_collision_filter(self):
        return 0 # disable self collisions
    
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
        if (self._state_init == HumanoidAMPDual.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMPDual.StateInit.Start
              or self._state_init == HumanoidAMPDual.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMPDual.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
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

        if self.ig_reward:
            ref_all_pos, ref_rot, ref_vel, ref_ang_vel,ref_key_pos \
                = self._motion_lib.get_il_state(motion_ids, motion_times)

            sim_all_pos = ref_all_pos.clone()
            sim_all_pos[..., 0,2] = self._initial_humanoid_root_states[env_ids,:,2]
            
            for i, env_id in enumerate(env_ids):
                new_base_kin_ig = compute_sim_ig_full(ref_all_pos,ref_vel)
                new_base_sim_ig = compute_sim_ig_full(sim_all_pos,ref_vel)
                self._base_kin_ig[env_id] = new_base_kin_ig[i]
                self._base_sim_ig[env_id] = new_base_sim_ig[i]
            
        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids,0] = motion_times
        self._motion_start_times[env_ids,1] = motion_times
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
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        super()._reset_env_tensors(env_ids)
        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0
        return
    
    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_start_times, self._motion_dt,
                                                    self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return       
            
    def compute_ref_observations(self, body_pos, body_rot, body_vel, body_ang_vel):
        obses = []
        for i in range(self.num_agents):
            self_obs = compute_humanoid_observations_max(body_pos[:, i], body_rot[:, i], body_vel[:, i], body_ang_vel[:, i], self._local_root_obs,
                                                self._root_height_obs)
            if i == 0:
                oppo_obs = compute_opponent_observations(body_pos[:,0,0], body_rot[:,0,0], body_pos[:,1], body_rot[:,1], body_vel[:,1], body_ang_vel[:, 1], self._root_height_obs)
            else:
                oppo_obs = compute_opponent_observations(body_pos[:,1,0], body_rot[:,1,0], body_pos[:,0], body_rot[:,0], body_vel[:,0], body_ang_vel[:, 0], self._root_height_obs)
            obs = torch.cat((self_obs, oppo_obs),dim=-1)
            obses.append(obs)

        obs = torch.cat([obs.unsqueeze(1) for obs in obses], dim=1)
        return obs
    
    def _compute_reward(self, actions):
        #print('+++',self.progress_buf[0][0])
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        cur_ref_times = self.progress_buf * self._motion_dt + self._motion_start_times
        #assert cur_ref_times[:,0] == cur_ref_times[:,1]
        cur_ref_time = cur_ref_times[:,0]
        ref_all_pos, ref_rot, ref_vel, ref_ang_vel,ref_key_pos \
            = self._motion_lib.get_il_state(self._motion_ids, cur_ref_time)

        ref_root_pos = ref_all_pos[..., 0, :]
        sim_root_pos = self._rigid_body_pos[..., 0, :]
        sim_key_pos = self._rigid_body_pos[..., self._key_body_ids, :]
        sim_rot = self._rigid_body_rot
        sim_vel = self._rigid_body_vel
        sim_ang_vel = self._rigid_body_ang_vel
        sim_all_pos = self._rigid_body_pos

        #print('++test_height',ref_root_pos[0,0,2],sim_root_pos[0,0,2])
        final_rew, final_rew_raw = compute_imitation_reward(sim_all_pos, sim_rot, sim_vel, sim_ang_vel, \
                    ref_all_pos, ref_rot, ref_vel, ref_ang_vel, self.reward_specs)
        
        #print('+++rew',final_rew.shape)
        self.rew_buf = final_rew
        #print('+++rew',self.rew_buf)
        if self.power_reward:
            force_tensor = self.dof_force_tensor.reshape(self.num_envs, self.num_agents,-1)
            power = torch.abs(torch.multiply(force_tensor, self._dof_vel)).sum(dim=-1) 
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            #print('++++checkpower',power_reward.mean())
            #self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
        if self.ig_reward:
            w_igp = 1
            w_igv = 0.2
            k_igp = 10
            k_igv = 0.5    
            edge_indices = compute_ref_interaction_mesh(ref_all_pos)
            sim_ig = compute_sim_ig(sim_all_pos,sim_vel,edge_indices)
            ref_ig = compute_sim_ig(ref_all_pos,ref_vel,edge_indices)
            
            dist_weights = compute_distance_weights(ref_ig)
            num_verts = 30 
            ig_pos_err = self.weighted_ig_pos_base_relative_error(sim_ig,ref_ig,dist_weights,edge_indices,num_verts)
            ig_vel_err = self.weighted_im_vel_error(sim_ig,ref_ig,dist_weights)
            ig_pos_rew = w_igp * torch.exp(-k_igp*ig_pos_err)
            ig_vel_rew = w_igv * torch.exp(-k_igv*ig_vel_err)
            ig_reward = self.ig_coefficient * (ig_pos_rew + ig_vel_rew)
            self.rew_buf[:] += ig_reward
        
        # print('++++raw reward',final_rew.mean(),final_rew_raw[...,0].mean(), final_rew_raw[...,2].mean(), \
        #     power_reward.mean(),ig_pos_rew.mean(),ig_vel_rew.mean())   
        return
    
    def weighted_im_vel_error_single(self, sim_sim_interaction_graph, kin_kin_interaction_graph, kin_dist_weights):
        error = 0
        assert len(sim_sim_interaction_graph.shape)==2
        diff = np.array(sim_sim_interaction_graph)[:,3:]-np.array(kin_kin_interaction_graph)[:,3:]
        per_joint_vel_dist = np.linalg.norm(diff,axis=1)
        per_joint_vel_dist = per_joint_vel_dist * per_joint_vel_dist * kin_dist_weights
        error = np.sum(per_joint_vel_dist)
        #print('+++velerr',error)
        return error  

    def weighted_im_vel_error(self, sim_sim_interaction_graph, kin_kin_interaction_graph, kin_dist_weights):
        errors = []
        for n in range(len(sim_sim_interaction_graph)):
            error1 = self.weighted_im_vel_error_single(sim_sim_interaction_graph[n][0],
                        kin_kin_interaction_graph[n][0],kin_dist_weights[n][0])
            error2 = self.weighted_im_vel_error_single(sim_sim_interaction_graph[n][1],
                        kin_kin_interaction_graph[n][1],kin_dist_weights[n][1])
            errors.append([error1, error2])
        err = torch.tensor(errors,device=self.device)
        return err
    
    def weighted_ig_pos_base_relative_error_single(self, sim_sim_interaction_graph, kin_kin_interaction_graph, kin_dist_weights, pruned_edges, num_verts, env_idx, agent_idx):
        error = 0
        ## Reconstruct full matrix
        sim_pairwise_dist_full_mat = []
        kin_pairwise_dist_full_mat = []
        row = pruned_edges[0]
        col = pruned_edges[1]
        ones = np.ones_like(row)
        kin_dist_weights = np.array(kin_dist_weights)
        edges_full_mat = coo_matrix((ones,(row,col)),shape=(num_verts,num_verts)).toarray()
        weights_full_mat = coo_matrix((kin_dist_weights,(row,col)),shape=(num_verts,num_verts)).toarray()

        for i in range(3):
            sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(num_verts,num_verts)).toarray()
            kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(num_verts,num_verts)).toarray()
            sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
        
        sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
        kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

        base_sim_full_mat = self._base_sim_ig[env_idx][agent_idx][:,:,:3]
        base_kin_full_mat = self._base_kin_ig[env_idx][agent_idx][:,:,:3]

        sim_diff_2_base = (sim_pairwise_dist_full_mat - base_sim_full_mat)*edges_full_mat[:,:,np.newaxis]
        kin_diff_2_base = (kin_pairwise_dist_full_mat - base_kin_full_mat)*edges_full_mat[:,:,np.newaxis]

        sim_diff_2_base_ratio =  np.nan_to_num(sim_diff_2_base / np.linalg.norm(base_sim_full_mat,axis=2)[:,:,np.newaxis])
        kin_diff_2_base_ratio =  np.nan_to_num(kin_diff_2_base / np.linalg.norm(base_kin_full_mat,axis=2)[:,:,np.newaxis])

        self_vert_cnt = self._self_interaction_vert_cnt
        oppo_vert_cnt = self._oppo_interaction_vert_cnt

        sim_diff_2_base_ratio[self_vert_cnt:,:-oppo_vert_cnt] = sim_pairwise_dist_full_mat[self_vert_cnt:,:-oppo_vert_cnt]*edges_full_mat[self_vert_cnt:,:-oppo_vert_cnt,np.newaxis]
        sim_diff_2_base_ratio[:-oppo_vert_cnt,self_vert_cnt:] = sim_pairwise_dist_full_mat[:-oppo_vert_cnt,self_vert_cnt:]*edges_full_mat[:-oppo_vert_cnt,self_vert_cnt:,np.newaxis]

        kin_diff_2_base_ratio[self_vert_cnt:,:-oppo_vert_cnt] = kin_pairwise_dist_full_mat[self_vert_cnt:,:-oppo_vert_cnt]*edges_full_mat[self_vert_cnt:,:-oppo_vert_cnt,np.newaxis]
        kin_diff_2_base_ratio[:-oppo_vert_cnt,self_vert_cnt:] = kin_pairwise_dist_full_mat[:-oppo_vert_cnt,self_vert_cnt:]*edges_full_mat[:-oppo_vert_cnt,self_vert_cnt:,np.newaxis]

        sim_pos_dist = np.linalg.norm(sim_pairwise_dist_full_mat,axis=2)
        kin_pos_dist = np.linalg.norm(kin_pairwise_dist_full_mat,axis=2)

        diff = sim_diff_2_base_ratio - kin_diff_2_base_ratio
        dist_old = np.linalg.norm(diff,axis=2)
        
        dist = np.array(dist_old)
        dist[self_vert_cnt:,:-oppo_vert_cnt] = 0.5 * dist_old[self_vert_cnt:,:-oppo_vert_cnt]/(kin_pos_dist[self_vert_cnt:,:-oppo_vert_cnt]+1e-6) + 0.5* dist_old[self_vert_cnt:,:-oppo_vert_cnt]/(sim_pos_dist[self_vert_cnt:,:-oppo_vert_cnt]+1e-6)
        dist[:-oppo_vert_cnt,self_vert_cnt:] = 0.5 * dist_old[:-oppo_vert_cnt,self_vert_cnt:]/(kin_pos_dist[:-oppo_vert_cnt,self_vert_cnt:]+1e-6) + 0.5* dist_old[:-oppo_vert_cnt,self_vert_cnt:]/(sim_pos_dist[:-oppo_vert_cnt,self_vert_cnt:]+1e-6)
        
        dist = dist * dist * weights_full_mat
        error = np.sum(dist)
        return error

    def weighted_ig_pos_base_relative_error(self, sim_sim_interaction_graph, kin_kin_interaction_graph, kin_dist_weights, pruned_edges, num_verts):
        errors = []
        for n in range(len(sim_sim_interaction_graph)):
            error1 = self.weighted_ig_pos_base_relative_error_single(sim_sim_interaction_graph[n][0], kin_kin_interaction_graph[n][0],
                        kin_dist_weights[n][0],pruned_edges[n][0],num_verts,n,0)
            error2 = self.weighted_ig_pos_base_relative_error_single(sim_sim_interaction_graph[n][1], kin_kin_interaction_graph[n][1],
                        kin_dist_weights[n][1],pruned_edges[n][1],num_verts,n,1)
            errors.append([error1, error2])
        err = torch.tensor(errors,device=self.device)
        return err
 
@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, motion_start_times, dt, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    #print('+++++',progress_buf[0])
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + motion_start_times
    if (enable_early_termination):
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
        has_fallen = torch.where(fall_mask, torch.tensor(True), has_fallen)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(motion_times > motion_lengths.unsqueeze(1), torch.ones_like(reset_buf), terminated)
    #print(motion_lengths)
    #print(motion_times.shape, reset_buf.shape, terminated.shape, reset.shape, motion_lengths.shape)
    #print("reset CHECKK:", reset.shape)
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
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])

    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    #print('++++++obs_vel', flat_body_vel.shape, flat_body_vel[0], flat_local_body_vel[0])
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    #print('++++++input_obs_shape', obs.shape)
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

def compute_ref_interaction_mesh(ref_all_pos):
    edge_indices = []
    n_envs, _, n_joints, _ = ref_all_pos.shape
    for n in range(n_envs):
        cur_env_pos = ref_all_pos[n]
        cur_pos1 = torch.cat([cur_env_pos[0],cur_env_pos[1]],dim=0).reshape(-1,3)
        cur_pos2 = torch.cat([cur_env_pos[1],cur_env_pos[0]],dim=0).reshape(-1,3)
        cur_ig1 = Interaction(cur_pos1)
        cur_ig2 = Interaction(cur_pos2)
        eg_idx1 = cur_ig1.build_interaction_graph()
        eg_idx2 = cur_ig2.build_interaction_graph()
        edge_indices.append([eg_idx1, eg_idx2])
    return edge_indices

def compute_interaction_mesh(points,edge_index=None,T=constants.EYE_T):

    R,p = conversions.T2Rp(T)
    R_inv = R.transpose()
    int_i = np.array(points)
    int_j = np.array(points)
    if edge_index is None:
        int_i_exp = np.expand_dims(int_i,1)
        int_i_repeat = np.repeat(int_i_exp,len(points),1)
        int_j_exp = np.expand_dims(int_j,0)
        int_j_repeat = np.repeat(int_j_exp,len(points),0)

        d_int = (int_j_repeat - int_i_repeat)

        d_int  [:,:,:3]=  (R_inv @ d_int[:,:,:3].transpose(0,2,1)).transpose(0,2,1)
        d_int  [:,:,3:]= (R_inv @ d_int[:,:,3:].transpose(0,2,1)).transpose(0,2,1)
        return d_int
    else:
        int_i = int_i[edge_index[0]]
        int_j = int_j[edge_index[1]]

        d_int = int_j - int_i
        
        d_int[:,:3] = (R_inv @ d_int[:,:3].T).T
        d_int [:,3:]= (R_inv @ d_int[:,3:].T).T
        return d_int

def compute_distance_weights(mesh,return_dist = False,weight_type='kin',interaction_kernel=10):
    if weight_type == 'kin':
        weights = []
        for n in range(len(mesh)):
            cur_mesh1 = mesh[n][0]
            cur_mesh2 = mesh[n][1]
            dist1 = np.linalg.norm(cur_mesh1[:,:3],axis=1)
            
            weights1 = np.exp(-interaction_kernel*dist1)
            weights1 = weights1/np.sum(weights1)

            dist2 = np.linalg.norm(cur_mesh2[:,:3],axis=1)
            
            weights2 = np.exp(-interaction_kernel*dist2)
            weights2 = weights2/np.sum(weights2)
            weights.append([weights1, weights2])
        return weights  
    else:
        raise NotImplementedError 

def compute_sim_ig(sim_all_pos, sim_vel, edge_indices):
    n_envs, _, n_joints, _ = sim_all_pos.shape
    sim_ig = []
    for n in range(n_envs):
        cur_pos = sim_all_pos[n]
        cur_vel = sim_vel[n]
        cur_pv = torch.cat([cur_pos, cur_vel],dim=-1)
        cur_state1 = torch.cat([cur_pv[0],cur_pv[1]],dim=0).reshape(-1,6)
        cur_state2 = torch.cat([cur_pv[1],cur_pv[0]],dim=0).reshape(-1,6)
        cur_sim_ig1 = compute_interaction_mesh(cur_state1, edge_indices[n][0])
        cur_sim_ig2 = compute_interaction_mesh(cur_state2, edge_indices[n][1])
        sim_ig.append([cur_sim_ig1, cur_sim_ig2])
    return sim_ig

def compute_sim_ig_full(sim_all_pos, sim_vel):
    n_envs, _, n_joints, _ = sim_all_pos.shape
    sim_ig = []
    for n in range(n_envs):
        cur_pos = sim_all_pos[n]
        cur_vel = sim_vel[n]
        cur_pv = torch.cat([cur_pos, cur_vel],dim=-1)
        cur_state1 = torch.cat([cur_pv[0],cur_pv[1]],dim=0).reshape(-1,6)
        cur_state2 = torch.cat([cur_pv[1],cur_pv[0]],dim=0).reshape(-1,6)
        cur_sim_ig1 = compute_interaction_mesh(cur_state1)
        cur_sim_ig2 = compute_interaction_mesh(cur_state2)
        sim_ig.append([cur_sim_ig1, cur_sim_ig2])
        
    return sim_ig