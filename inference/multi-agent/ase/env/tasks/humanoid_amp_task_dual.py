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
from isaacgym import gymapi
from isaacgym.torch_utils import *
# import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_dual as humanoid_amp_dual
import copy
class HumanoidAMPTaskDual(humanoid_amp_dual.HumanoidAMPDual):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._task_ref_steps =  cfg["env"]["task_ref_steps"]
        self.obs_v = cfg["env"]["obs_v"]
        if self.obs_v == 2:
            self._task_obs_size_per_step = 315
        elif self.obs_v == 3:
            self._task_obs_size_per_step = 135
        elif self.obs_v == 4:
            self._task_obs_size_per_step = 360
        else:
            raise NotImplementedError

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._intergraph_body_ids = self._build_intergraph_body_ids_tensor(cfg["env"]["intergraphBodies"])
        self.intergraph_obs_buf = torch.zeros((self.num_envs, self.num_agents, 15, 15, 3), device=self.device)
        return

    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_num_motions(self):
        return self._motion_lib.num_motions()
    
    def get_motion_name(self, motion_ids):
        return self._motion_lib.get_motion_name(motion_ids)

    def get_motion_num_frames(self, motion_ids):
        return self._motion_lib.get_motion_num_frames(motion_ids)
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = self._task_ref_steps*self._task_obs_size_per_step
        return obs_size

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        #print("Building environment in humanoid_dual_imitate: ", env_id)
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0
        #motion_ids = self._motion_lib.sample_motions(1)
        #print('++++++build_env')
        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89
        rand_pos = (2.0 * torch.rand([2], device=self.device) - 1.0)
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

        start_pose.r = gymapi.Quat(0.0,0.0, 1.0, 0.0)
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
    
    def obs_max_to_dof(self, obs_max):
        #############reverse part ################
        orig_local_rot = torch_utils.tan_norm_to_quat(local_body_rot_obs.reshape(-1,15,6))
        orig_root_rot = orig_local_rot[:,0,:]
        orig_heading_rot = torch_utils.calc_heading_quat_inv(orig_root_rot)
        orig_heading_rot_expand = orig_heading_rot.unsqueeze(-2)
        orig_heading_rot_expand = orig_heading_rot_expand.repeat((1, orig_local_rot.shape[1], 1))
        flat_orig_heading_rot = orig_heading_rot_expand.reshape(orig_heading_rot_expand.shape[0] * orig_heading_rot_expand.shape[1], 
                                                orig_heading_rot_expand.shape[2])
        
        flat_orig_local_rot = orig_local_rot.reshape(orig_local_rot.shape[0] * orig_local_rot.shape[1], orig_local_rot.shape[2])
        flat_orig_body_rot = quat_mul(quat_conjugate(flat_orig_heading_rot), flat_orig_local_rot)
        orig_body_rot = flat_body_rot.reshape(orig_local_rot.shape[0], orig_local_rot.shape[1], orig_local_rot.shape[2])
        #print("checkrot",flat_local_body_rot.reshape(-1,15,4)[0,:,:],torch_utils.tan_norm_to_quat(local_body_rot_obs.reshape(-1,15,6))[0,:,:],body_rot[0,0,:])
        #print("checkrot",body_rot[0,:,:],orig_body_rot[0,:,:])
        #print('++++++input_obs_shape', obs.shape)
        orig_local_body_pos = local_body_pos.reshape(-1,14,3)
        dummy_root = torch.zeros(orig_local_body_pos.shape[0],1,orig_local_body_pos.shape[2],device=orig_local_body_pos.device,dtype=orig_local_body_pos.dtype)
        orig_local_body_pos = torch.cat([dummy_root, orig_local_body_pos],dim=1)
        flat_orig_local_body_pos = orig_local_body_pos.reshape(orig_local_body_pos.shape[0] * orig_local_body_pos.shape[1], orig_local_body_pos.shape[2])
        flat_orig_global_body_pos = quat_rotate(quat_conjugate(flat_orig_heading_rot), flat_orig_local_body_pos)
        orig_relative_global_pos = flat_orig_global_body_pos.reshape(orig_local_body_pos.shape[0], orig_local_body_pos.shape[1], orig_local_body_pos.shape[2])
        global_root_xy = torch.zeros(root_h_obs.shape[0],2,device=root_h_obs.device,dtype=root_h_obs.dtype)
        global_root_pos = torch.cat([global_root_xy,root_h_obs],dim=-1)
        global_root_pos_expand = global_root_pos.unsqueeze(-2)
        orig_global_pos = orig_relative_global_pos + global_root_pos_expand
        #print("checkpos",orig_global_pos[0,:,:],body_pos[0,:,:])
        
        skeleton_xml_path = "ase/data/assets/mjcf/amp_humanoid.xml"
        skeleton_tree = SkeletonTree.from_mjcf(skeleton_xml_path)
        
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,orig_body_rot[0],orig_global_pos[0,0],is_local=False)
        local_body_rot = sk_state.local_rotation
        
        dof_pos = self._motion_lib._local_rotation_to_dof(local_body_rot.unsuqeeze(0))
        print('checklocalrot',dof_pos.shape)
        return dof_pos
        
    def _build_termination_heights(self):
        super()._build_termination_heights()
        termination_distance = self.cfg["env"].get("terminationDistance", 0.5)
        self._termination_distances = to_torch(np.array([termination_distance] * self.num_bodies), device=self.device)
        self._termination_distances = self._termination_distances.unsqueeze(-1).expand(-1,2)
        print('************tm heights', self._termination_distances.shape)
        return
    
    def _init_obs(self, env_ids):
        super()._init_obs(env_ids)
        self._compute_interaction_graph_obs(env_ids)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._compute_interaction_graph_obs()
        # Add interaction graph to extras
        self.extras["intergraph_obs"] = self.intergraph_obs_buf
        
        if self.collect_dataset:
            self.extras['obs_buf'] = self.obs_buf_t.copy()  # n, 945
            self.extras['intergraph_obs_buf_t'] = self.intergraph_obs_buf_t.copy()  # n, 2, 15, 15, 3
            self.extras['actions'] = self.actions.reshape(self.num_envs, 2, -1).cpu().numpy()  # n, 69
            self.extras['clean_actions'] = self.clean_actions.reshape(self.num_envs, 2, -1).cpu().numpy()
            self.extras['reset_buf'] = self.reset_buf.cpu().numpy()  # n

        
            self.obs_buf_t = self.obs_buf.cpu().numpy() # update to next time step
            self.intergraph_obs_buf_t = self.intergraph_obs_buf.cpu().numpy()
            self.rigid_body_pos_t = self._rigid_body_pos.cpu().numpy()
            self.rigid_body_rot_t = self._rigid_body_rot.cpu().numpy()

            self.extras['rigid_body_pos_t'] = self.rigid_body_pos_t.copy()
            self.extras['rigid_body_rot_t'] = self.rigid_body_rot_t.copy()
        # self.extras['rigid_body_pos_t'] = copy.deepcopy(self.rigid_body_pos_t)
        # self.extras['rigid_body_rot_t'] = copy.deepcopy(self.rigid_body_rot_t)
        return
    
    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        if self.collect_dataset:
            self.intergraph_obs_buf_t = self.intergraph_obs_buf.cpu().numpy()
            self.rigid_body_pos_t = self._rigid_body_pos.cpu().numpy()
            self.rigid_body_ros_t = self._rigid_body_ros.cpu().numpy()
        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return

    def _build_intergraph_body_ids_tensor(self, intergraph_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in intergraph_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _compute_interaction_graph_obs(self, env_ids=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
        else:
            body_pos = self._rigid_body_pos[env_ids]
        
        # [num_envs, 15, 1, 3] - agent 0's bodies as reference
        agent0_pos = body_pos[:, 0].unsqueeze(2)  
        # [num_envs, 1, 15, 3] - agent 1's bodies as targets
        agent1_pos = body_pos[:, 1].unsqueeze(1)
        # Compute differences: agent1_positions - agent0_positions
        if env_ids is None: 
            self.intergraph_obs_buf[:, 0] = agent1_pos - agent0_pos
        else:
            self.intergraph_obs_buf[env_ids, 0] = agent1_pos - agent0_pos

        agent1_pos = body_pos[:, 1].unsqueeze(2)
        agent0_pos = body_pos[:, 0].unsqueeze(1)
        if env_ids is None:
            self.intergraph_obs_buf[:, 1] = agent0_pos - agent1_pos
        else:
            self.intergraph_obs_buf[env_ids, 1] = agent0_pos - agent1_pos
        
        return