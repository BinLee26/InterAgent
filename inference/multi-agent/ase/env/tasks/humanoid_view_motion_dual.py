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
import os
import yaml
import sys

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils.motion_lib_il import MotionLibIl
from utils.motion_lib_il_ig import MotionLibIlIg

from env.tasks.humanoid_amp_dual import HumanoidAMPDual
import env.tasks.humanoid_il_ig_v2 as humanoid_il_ig_v2
from env.tasks.humanoid_il_ig_v2 import HumanoidIlIgv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HumanoidViewMotionDual(HumanoidIlIgv2):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self.cur_view_motion_id = torch.tensor(0,device=self.device)
        
        self.view_ig = False
        self.plot_once = False
        return

    
    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        #print("action shape check:", self.actions.shape)
        forces = torch.zeros_like(self.actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        #print("pre_physics_step_CHECK: successful")
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return
    
    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions
    
    def get_task_obs_size(self):
        return self._num_obs*3

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLibIl(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        
        ########## filter ############
        
        # valid_motion_entries = []
        # invalid_motion_entries = []

        # for motion_id in range(self._motion_lib.num_motions()):

        #     motion_name = self._motion_lib.get_motion_name(motion_id)

        #     if self._motion_lib.is_motion_valid(motion_id, ground_threshold=0.1, airborne_thresh=45,pos_jump_thresh=0.5):
        #         f1, f2 = self._motion_lib._motion_files[motion_id]
        #         fname1 = os.path.basename(f1)
        #         fname2 = os.path.basename(f2)
        #         valid_motion_entries.append({'file': fname1, 'weight': 1.0})
        #         valid_motion_entries.append({'file': fname2, 'weight': 1.0})
        #     else:
        #         f1, f2 = self._motion_lib._motion_files[motion_id]
        #         fname1 = os.path.basename(f1)
        #         fname2 = os.path.basename(f2)
        #         invalid_motion_entries.append({'file': fname1, 'weight': 1.0})
        #         invalid_motion_entries.append({'file': fname2, 'weight': 1.0})

        # # good motion
        # with open("ase/data/motions/intergen_amp/cleaned_motions.yaml", 'w') as f:
        #     yaml.dump({'motions': valid_motion_entries}, f, sort_keys=False)

        # # illegal motion
        # with open("ase/data/motions/intergen_amp/filtered_motions.yaml", 'w') as f:
        #     yaml.dump({'motions': invalid_motion_entries}, f, sort_keys=False)

        # print(f" Saved {len(valid_motion_entries)//2} valid motions to cleaned_motions.yaml")
        # print(f" Saved {len(invalid_motion_entries)//2} filtered motions to filtered_motions.yaml")
        # sys.exit()
        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        # print("Building environment in humanoid_loc_dual: ", env_id)
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89
        rand_pos = 10.0 * (2.0 * torch.rand([2], device=self.device) - 1.0)
        rand_pos1 = - rand_pos

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        start_pose.p.x += rand_pos[0]
        start_pose.p.y += rand_pos[1]

        humanoid_handle1 = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid1", col_group, col_filter, segmentation_id)
        

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle1)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle1, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle1, dof_prop)

        # actor 2
        segmentation_id = 1

        start_pose = gymapi.Transform()
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.p.x += rand_pos1[0]
        start_pose.p.y += rand_pos1[1]

        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        humanoid_handle2 = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid2", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle2)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle2, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.4, 0.8))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle2, dof_prop)

        self.humanoid_handles.append(humanoid_handle1)
        self.humanoid_handles.append(humanoid_handle2)
        
        # if (not self.headless):
        #     self._build_marker(env_id, env_ptr)

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
    
    def _reset_actors(self, env_ids):
        # env_ids = torch.arange(self.num_envs, device=self.device)
        # #print("env_ids CHECK:", env_ids)
        # n = len(env_ids)
        # rand_pos = 10.0 * (2.0 * torch.rand([n, self.num_agents, 2], device=self.device) - 1.0)
        # self._initial_humanoid_root_states[env_ids, :, :2] = rand_pos[env_ids]

        # self._root_states[env_ids, :, :2] = rand_pos[env_ids]
        
        # self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        # self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        num_envs = env_ids.shape[0]
        if env_ids[0] == 0:
            self.cur_view_motion_id = torch.remainder(self.cur_view_motion_id+1,self._motion_lib.num_motions())
            self._motion_lib.set_cur_sample_idx(self.cur_view_motion_id)
            print(f"cur viewing motion {self.cur_view_motion_id}{self._motion_lib.get_motion_name(self.cur_view_motion_id)}")
        motion_ids = self._motion_lib.sample_motions_by_cur_idx(num_envs)
        # motion_ids = self._motion_lib.sample_motions(num_envs)
        motion_times0 = torch.zeros(motion_ids.shape, device=self.device)
        motion_times = self._motion_lib.sample_time(motion_ids)
        root_poses, root_rots, dof_poses, root_vels, root_ang_vels, dof_vels, key_poses \
                = self._motion_lib.get_motion_state(motion_ids, motion_times)
        self._motion_ids[env_ids] = motion_ids
        #print('root_poses',root_rots[:,0])    
        return
    
    def _compute_reward(self, actions):
        self.rew_buf = torch.zeros_like(self.rew_buf)
        return
    
    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        curr_num_frame = self._motion_lib.get_motion_num_frames(motion_ids)
        #curr_file_name = self._motion_lib.get_motion_name(motion_ids)
        #print('++++curr id:',motion_ids,curr_num_frame,curr_file_name)
        motion_times = self.progress_buf * self._motion_dt
        #print('debug',motion_times)
        motion_times = motion_times[:, 0] 
        #print("motion_times", motion_times)
        #motion_times = torch.zeros_like(motion_times)
        # root_poses, root_rots, dof_poses, root_vels, root_ang_vels, dof_vels, key_poses \
        #    = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        all_root_poses, all_root_rots, root_vels, root_ang_vels, dof_vels, dof_poses, key_poses \
            = self._motion_lib.get_dual_state(motion_ids, motion_times)
        root_poses = all_root_poses[...,0,:]
        root_rots = all_root_rots[...,0,:]
        
        # root_vels = torch.zeros_like(root_vels)
        # root_ang_vels = torch.zeros_like(root_ang_vels)
        # dof_vels = torch.zeros_like(dof_vels)
        #print('debug', root_poses[0,0,0:3])
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_poses, 
                            root_rot=root_rots, 
                            dof_pos=dof_poses, 
                            root_vel=root_vels, 
                            root_ang_vel=root_ang_vels, 
                            dof_vel=dof_vels)
        
        if self.view_ig and (not self.plot_once):
            ref_all_pos, ref_rot, ref_vel, ref_ang_vel,ref_key_pos \
                = self._motion_lib.get_il_state(self._motion_ids, motion_times)
            edge_indices = self._motion_lib.get_ig_edge_indices(self._motion_ids, motion_times)
            ref_ig = humanoid_il_ig_v2.compute_ig(ref_all_pos,ref_vel,edge_indices)
            dist_weights = self.compute_distance_weights(ref_ig)
            #print("+++++++d",dist_weights[0].shape)
            ig_connectivity = list(zip(edge_indices[0][0], edge_indices[0][1]))
            ig_points = ref_all_pos[0].reshape(2*15,3)
            plot_ig(ig_points, ig_connectivity,dist_weights[0])
            #self.plot_once = True
        
        
        #env_ids_int32 = self._humanoid_actor_ids[env_ids]
        env_ids_int32 = torch.arange(self.num_envs*self.num_agents, dtype=torch.int32, device=self.device)
        #print(env_ids_int32)
        #print("root states:", self._root_states.shape, self._root_states)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states.view(self.num_envs*self.num_agents, -1)),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)) 
        return
    
    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        #self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        #print('+++curmotion',self._motion_lib.get_motion_name(self._motion_ids[0]))
        #print('+++++num_frames',self._motion_lib.get_motion_num_frames(self._motion_ids[0]))
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return


@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths.unsqueeze(1), torch.ones_like(reset_buf), terminated)
    #print(motion_lengths)
    #print(motion_times.shape, reset_buf.shape, terminated.shape, reset.shape, motion_lengths.shape)
    #print("reset CHECKK:", reset.shape)
    return reset, terminated

def plot_ig(points, connectivity, weights):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=40)

    for (i, (a, b)) in enumerate(connectivity):
        xs = [points[a, 0], points[b, 0]]
        ys = [points[a, 1], points[b, 1]]
        zs = [points[a, 2], points[b, 2]]
        color = (0, 0, weights[i])
        ax.plot(xs, ys, zs, linewidth=2, alpha=weights[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()