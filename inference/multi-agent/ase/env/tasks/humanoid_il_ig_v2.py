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
from env.tasks.humanoid_il_dm_v2 import HumanoidIlDMv2
from utils.motion_lib_il_ig import MotionLibIlIg
from utils import torch_utils

from utils.misc.interaction import Interaction
from fairmotion.utils import utils,constants
from fairmotion.ops import conversions, math
from fairmotion.ops import quaternion

import numpy as np
from scipy.sparse import coo_matrix


class HumanoidIlIgv2(HumanoidIlDMv2):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._interaction_kernel = 10
        self._self_interaction_vert_cnt = 15
        self._oppo_interaction_vert_cnt = 15
        self._base_kin_ig = [None] * self.num_envs
        self._base_sim_ig = [None] * self.num_envs
        self.ig_reward_specs = cfg["env"].get("ig_reward_specs", {"k_igp": 10, "k_igv": 0.5, "k_igr": 2.5, "w_igp": 1, "w_igv": 1, "w_igr": 1})

        
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        return

    def post_physics_step(self):
        super().post_physics_step()
        return
    
    
    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLibIlIg(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _compute_reward(self, actions):
        #print('+++',self.progress_buf[0][0])
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        cur_ref_times = self.progress_buf * self._motion_dt + self._motion_start_times
        #assert cur_ref_times[:,0] == cur_ref_times[:,1]
        cur_ref_time = cur_ref_times[:,0]
        ref_all_pos, ref_rot, ref_vel, ref_ang_vel,ref_key_pos \
            = self._motion_lib.get_il_state(self._motion_ids, cur_ref_time)
        #print('++++',cur_ref_time[0])
        ref_root_pos = ref_all_pos[..., 0, :]
        sim_root_pos = self._rigid_body_pos[..., 0, :]
        sim_key_pos = self._rigid_body_pos[..., self._key_body_ids, :]
        sim_rot = self._rigid_body_rot
        sim_vel = self._rigid_body_vel
        sim_ang_vel = self._rigid_body_ang_vel
        sim_all_pos = self._rigid_body_pos
        
        #edge_indices = self.compute_ref_interaction_mesh(ref_all_pos)
        edge_indices = self._motion_lib.get_ig_edge_indices(self._motion_ids, cur_ref_time)

        sim_ig = compute_ig(sim_all_pos,sim_vel,edge_indices)
        ref_ig = compute_ig(ref_all_pos,ref_vel,edge_indices)
        
        dist_weights = self.compute_distance_weights(ref_ig)
        num_verts = 30

        ig_pos_err = self.compute_ig_pos_error(sim_ig,ref_ig,dist_weights)
        ig_vel_err = self.compute_ig_vel_error(sim_ig,ref_ig,dist_weights)
        root_err = self.compute_root_error(sim_root_pos,ref_root_pos,sim_rot[...,0,:],ref_rot[...,0,:],
                                    sim_vel[...,0,:],ref_vel[...,0,:],sim_ang_vel[...,0,:],ref_ang_vel[...,0,:])        
        
        ig_rew_specs = self.ig_reward_specs
        k_igp, k_igv, k_igr = ig_rew_specs["k_igp"], ig_rew_specs["k_igv"], ig_rew_specs["k_igr"]
        w_igp, w_igv, w_igr = ig_rew_specs["w_igp"], ig_rew_specs["w_igv"], ig_rew_specs["w_igr"]
        
        ig_pos_rew = w_igp*torch.exp(-k_igp*ig_pos_err) 
        ig_vel_rew = w_igv*torch.exp(-k_igv*ig_vel_err)
        root_rew = w_igr*torch.exp(-k_igr*root_err)
        ig_pos_rew_exp = ig_pos_rew.unsqueeze(-1).expand(-1,2)
        ig_vel_rew_exp = ig_vel_rew.unsqueeze(-1).expand(-1,2)
        #print(f"ig reward shape check: pos:{ig_pos_rew_exp.shape}, vel:{ig_vel_rew_exp.shape}, root: {root_rew.shape}")
        final_rew = ig_vel_rew_exp * ig_pos_rew_exp * root_rew
        #print(f'ig err scale check: pos:{ig_pos_err.mean()}, vel:{ig_vel_err.mean()},root:{root_err.mean()}')
        #print(f'ig rew scale check: pos:{ig_pos_rew.mean()}, vel:{ig_vel_rew.mean()},root:{root_rew.mean()}')
        self.rew_buf = final_rew
        #self.rew_buf = torch.zeros_like(self.rew_buf)
        # if self.power_reward:
        #     force_tensor = self.dof_force_tensor.reshape(self.num_envs, self.num_agents,-1)
        #     power = torch.abs(torch.multiply(force_tensor, self._dof_vel)).sum(dim=-1) 
        #     power_reward = -self.power_coefficient * power
        #     power_reward[self.progress_buf <= 3] = 0 

        #     self.rew_buf[:] += power_reward
        return
        
    def compute_root_error(self,sim_root_pos,ref_root_pos,sim_rot,ref_rot,sim_vel,ref_vel,sim_ang_vel,ref_ang_vel):
        w_p = 1
        w_r = 0.1
        w_v = 0.01
        w_a = 0.001
        pos_err = torch.norm(sim_root_pos - ref_root_pos, dim=-1)
        theta = torch_utils.quat_dist(sim_rot, ref_rot).squeeze(-1)
        rot_err = theta * theta
        vel_err = torch.norm(sim_vel - ref_vel, dim=-1)
        ang_vel_err = torch.norm(sim_ang_vel - ref_ang_vel, dim=-1)
        err = w_p * pos_err + w_r * rot_err + w_v * vel_err + w_a * ang_vel_err
        #print('++++++rooterr',pos_err[0,0],rot_err[0,0],vel_err[0,0],ang_vel_err[0,0])
        return err

    def compute_distance_weights(self,mesh,return_dist = False,weight_type='kin'):
        if weight_type == 'kin':
            weights_vec = []
            for n in range(len(mesh)):
                cur_mesh = mesh[n]
                dist = np.linalg.norm(cur_mesh[:,:3],axis=1)
                
                weight = np.exp(-self._interaction_kernel*dist)
                weight = weight/np.sum(weight)
                
                weights_vec.append(weight)
            return weights_vec  
        else:
            raise NotImplementedError 

    def compute_ig_vel_error(self, sim_ig, kin_ig, kin_dist_weights):
        errors = []
        for n in range(len(sim_ig)):
            error = weighted_im_vel_error(sim_ig[n],kin_ig[n],kin_dist_weights[n])
            errors.append(error)
        err = torch.tensor(errors,device=self.device)
        return err

    def compute_ig_pos_error(self, sim_ig, kin_ig, kin_dist_weights):
        errors = []
        for n in range(len(sim_ig)):
            error = weighted_ig_pos_ratio(sim_ig[n], kin_ig[n],
                        kin_dist_weights[n])
            errors.append(error)
        err = torch.tensor(errors,device=self.device)
        return err    

def compute_ig(sim_all_pos, sim_vel, edge_indices):
    n_envs, _, n_joints, _ = sim_all_pos.shape
    sim_ig = []
    for n in range(n_envs):
        cur_pos = sim_all_pos[n]
        cur_vel = sim_vel[n]
        cur_pv = torch.cat([cur_pos, cur_vel],dim=-1).reshape(-1,6)
        cur_ig = compute_interaction_mesh(cur_pv, edge_indices[n])
        sim_ig.append(cur_ig)
    return sim_ig
    
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


def weighted_ig_pos_ratio(sim_ig, kin_ig, kin_dist_weights):
    error = 0
    kin_pos_dist = np.linalg.norm(kin_ig[:,:3],axis=1)
    diff_norm = np.linalg.norm((sim_ig[:,:3]-kin_ig[:,:3]),axis=1)

    ratio = (diff_norm/(kin_pos_dist+1e-6))
    ratio = np.clip(ratio,0,5)
    ratio_sq = ratio * ratio * kin_dist_weights
    
    error = np.sum(ratio_sq)
    return error

def weighted_im_vel_error(sim_ig, kin_ig, kin_dist_weights):
    error = 0
    assert len(sim_ig.shape)==2
    diff = np.array(sim_ig)[:,3:]-np.array(kin_ig)[:,3:]
    per_joint_vel_dist = np.linalg.norm(diff,axis=1)
    per_joint_vel_dist = per_joint_vel_dist * per_joint_vel_dist * kin_dist_weights
    error = np.sum(per_joint_vel_dist)
    #print('+++velerr',error)
    return error  