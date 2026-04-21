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
# from utils.motion_lib_il_ig_single import MotionLibIlIgSingle
from env.tasks.humanoid_amp_dual import HumanoidAMPDual
from env.tasks.humanoid_amp_task_dual import HumanoidAMPTaskDual
from env.tasks.humanoid_il_dm_v2 import HumanoidIlDMv2
from utils.motion_lib_il_ig import MotionLibIlIg
from utils.motion_lib_il_ig_multi import MotionLibIlIgMulti
from utils import torch_utils

from utils.misc.interaction import Interaction
from fairmotion.utils import utils,constants
from fairmotion.ops import conversions, math
from fairmotion.ops import quaternion

import numpy as np
from scipy.sparse import coo_matrix


class HumanoidIlIgv3(HumanoidIlDMv2):
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
        self._motion_lib = MotionLibIlIgMulti(motion_file=motion_file,
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
        
        sim_ig_pos = sim_all_pos.reshape(self.num_envs, -1, 3)
        sim_ig_vel = sim_vel.reshape(self.num_envs, -1, 3)
        ref_ig_pos = ref_all_pos.reshape(self.num_envs, -1, 3)
        ref_ig_vel = ref_vel.reshape(self.num_envs, -1, 3)
        num_verts = 30
        #print("shape check", sim_all_pos.shape, sim_vel.shape, ref_all_pos.shape, ref_vel.shape)
        ig_pos_err, ig_vel_err = self.compute_ig_errors(sim_ig_pos,sim_ig_vel,ref_ig_pos,ref_ig_vel,self._motion_ids,cur_ref_time)
        #ig_vel_err = self.compute_ig_vel_error(sim_ig,ref_ig,dist_weights)
        #print("shape check", ig_pos_err.shape, ig_vel_err.shape)
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

    def compute_ig_errors(self, sim_pos, sim_vel, ref_pos, ref_vel, motion_ids, motion_times):
        """
        sim_pos: [batch_size, num_vertices, 3]
        sim_vel: [batch_size, num_vertices, 3]
        ref_pos: [batch_size, num_vertices, 3]
        ref_vel: [batch_size, num_vertices, 3]
        """
        # 获取当前帧的邻接矩阵和权重
        adj_matrices, weight_matrices = self._motion_lib.get_frame_data(motion_ids, motion_times)
        batch_size = sim_pos.shape[0]
        
        # 计算位置误差
        sim_pos_exp1 = sim_pos.unsqueeze(2)  # [B, N, 1, 3]
        sim_pos_exp2 = sim_pos.unsqueeze(1)  # [B, 1, N, 3]
        ref_pos_exp1 = ref_pos.unsqueeze(2)  # [B, N, 1, 3]
        ref_pos_exp2 = ref_pos.unsqueeze(1)  # [B, 1, N, 3]
        
        # 计算实际和参考的位置差向量
        sim_pos_diff = (sim_pos_exp1 - sim_pos_exp2) * adj_matrices.unsqueeze(-1)  # [B, N, N, 3]
        ref_pos_diff = (ref_pos_exp1 - ref_pos_exp2) * adj_matrices.unsqueeze(-1)  # [B, N, N, 3]
        
        # 计算位置误差
        pos_diff_norm = torch.norm(sim_pos_diff - ref_pos_diff, dim=-1)  # [B, N, N]
        ref_pos_norm = torch.norm(ref_pos_diff, dim=-1)  # [B, N, N]
        pos_ratio = pos_diff_norm / (ref_pos_norm + 1e-6)
        pos_ratio = torch.clamp(pos_ratio, 0, 5)
        pos_error = torch.sum(pos_ratio * pos_ratio * weight_matrices, dim=(1,2))  # [B]
        
        # 计算速度误差
        sim_vel_diff = (sim_vel.unsqueeze(2) - sim_vel.unsqueeze(1)) * adj_matrices.unsqueeze(-1)  # [B, N, N, 3]
        ref_vel_diff = (ref_vel.unsqueeze(2) - ref_vel.unsqueeze(1)) * adj_matrices.unsqueeze(-1)  # [B, N, N, 3]
        vel_diff = sim_vel_diff - ref_vel_diff
        vel_error = torch.sum(torch.norm(vel_diff, dim=-1) * weight_matrices, dim=(1,2))  # [B]
        
        return pos_error, vel_error