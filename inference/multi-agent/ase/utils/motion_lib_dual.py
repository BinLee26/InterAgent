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

import numpy as np
import os
import yaml
# from transformers import AutoTokenizer, CLIPTextModel

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *

from utils import torch_utils

import torch

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLibDual():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._load_motions(motion_file)

        motions = self._motions
        #print("++++++++++++", motions[0][0].skeleton_tree.node_names)
        self.gts_0 = torch.cat([m[0].global_translation for m in motions], dim=0).float()
        self.gts_1 = torch.cat([m[1].global_translation for m in motions], dim=0).float()
        self.gts = torch.stack([self.gts_0, self.gts_1], dim=0).float()
        #print("gts.shape:", self.gts.shape)
        self.grs_0 = torch.cat([m[0].global_rotation for m in motions], dim=0).float()
        self.grs_1 = torch.cat([m[1].global_rotation for m in motions], dim=0).float()
        self.grs = torch.stack([self.grs_0, self.grs_1], dim=0).float()
        #print("grs.shape:", self.grs.shape)
        self.lrs_0 = torch.cat([m[0].local_rotation for m in motions], dim=0).float()
        self.lrs_1 = torch.cat([m[1].local_rotation for m in motions], dim=0).float()
        self.lrs = torch.stack([self.lrs_0, self.lrs_1], dim=0).float()
        #print("lrs.shape:", self.lrs.shape)
        self.grvs_0 = torch.cat([m[0].global_root_velocity for m in motions], dim=0).float()
        self.grvs_1 = torch.cat([m[1].global_root_velocity for m in motions], dim=0).float()
        self.grvs = torch.stack([self.grvs_0, self.grvs_1], dim=0).float()

        self.gravs_0 = torch.cat([m[0].global_root_angular_velocity for m in motions], dim=0).float()
        self.gravs_1 = torch.cat([m[1].global_root_angular_velocity for m in motions], dim=0).float()
        self.gravs = torch.stack([self.gravs_0, self.gravs_1], dim=0).float()

        self.dvs_0 = torch.cat([m[0].dof_vels for m in motions], dim=0).float()
        self.dvs_1 = torch.cat([m[1].dof_vels for m in motions], dim=0).float()
        self.dvs = torch.stack([self.dvs_0, self.dvs_1], dim=0).float()

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)
    
    def get_total_frames(self):
        return sum(self._motion_num_frames)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        #print('sample motion++++++',self._motion_weights)
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        #print(motion_ids)
        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    # def sample_single_motion_id(self):
    #     motion_ids = torch.multinomial(self._motion_weights, num_samples=1, replacement=True)
    #     return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        #print(motion_time)
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids):
        return self._motion_num_frames[motion_ids]
    
    def get_motion_name(self, motion_ids):
        return self._motion_files[motion_ids][0]
    
    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        root_poses = []
        root_rots = []
        dof_poses = []
        root_vels = []
        root_ang_vels = []
        dof_vels = []
        key_poses = []

        for i in range(2):
            motion_len = self._motion_lengths[motion_ids]
            num_frames = self._motion_num_frames[motion_ids]
            dt = self._motion_dt[motion_ids]
            #print('++++motion_times_dual', motion_times, motion_times.shape)
            frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

            f0l = (frame_idx0 + self.length_starts[motion_ids]).to('cpu')
            #print("f0l.shape:", f0l.shape)
            f1l = (frame_idx1 + self.length_starts[motion_ids]).to('cpu')
            root_pos0 = self.gts[i, f0l, 0].to(self._device)
            root_pos1 = self.gts[i, f1l, 0].to(self._device)
            #print("root_pos0.shape:", root_pos0.shape)
            root_rot0 = self.grs[i, f0l, 0].to(self._device)
            root_rot1 = self.grs[i, f1l, 0].to(self._device)

            local_rot0 = self.lrs[i, f0l].to(self._device)
            local_rot1 = self.lrs[i, f1l].to(self._device)
            #print("local_rot0.shape:", local_rot0.shape)
            root_vel = self.grvs[i, f0l].to(self._device)

            root_ang_vel = self.gravs[i, f0l].to(self._device)
            self._key_body_ids = self._key_body_ids.to('cpu')
            key_pos0 = self.gts[i, f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)].to(self._device)
            key_pos1 = self.gts[i, f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)].to(self._device)

            dof_vel = self.dvs[i, f0l].to(self._device)

            vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
            for v in vals:
                assert v.dtype != torch.float64


            blend = blend.unsqueeze(-1)

            root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

            root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

            blend_exp = blend.unsqueeze(-1)
            key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
            
            local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof(local_rot)

           #print("root pose shape CHECK:" , local_rot.shape)
            root_poses.append(root_pos)
            root_rots.append(root_rot)
            dof_poses.append(dof_pos)
            root_vels.append(root_vel)
            root_ang_vels.append(root_ang_vel)
            dof_vels.append(dof_vel)
            key_poses.append(key_pos)
        
        root_poses = torch.stack(root_poses, dim=0)
        root_rots = torch.stack(root_rots, dim=0)
        dof_poses = torch.stack(dof_poses, dim=0)
        root_vels = torch.stack(root_vels, dim=0)
        root_ang_vels = torch.stack(root_ang_vels, dim=0)
        dof_vels = torch.stack(dof_vels, dim=0)
        key_poses = torch.stack(key_poses, dim=0)

        return root_poses, root_rots, dof_poses, root_vels, root_ang_vels, dof_vels, key_poses
    
    def get_dual_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        root_poses = []
        root_rots = []
        all_poses = []
        all_rots = []
        dof_poses = []
        root_vels = []
        root_ang_vels = []
        dof_vels = []
        key_poses = []

        for i in range(2):
            motion_len = self._motion_lengths[motion_ids]
            num_frames = self._motion_num_frames[motion_ids]
            dt = self._motion_dt[motion_ids]
            #print('++++motion_times_dual', motion_times, motion_times.shape)
            frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

            f0l = (frame_idx0 + self.length_starts[motion_ids]).to('cpu')
            #print("f0l.shape:", f0l.shape)
            f1l = (frame_idx1 + self.length_starts[motion_ids]).to('cpu')

            root_vel = self.grvs[i, f0l].to(self._device)
            root_ang_vel = self.gravs[i, f0l].to(self._device)
            
            all_pos0 = self.gts[i, f0l, :].to(self._device)
            all_pos1 = self.gts[i, f1l, :].to(self._device)
            
            all_rot0 = self.grs[i, f0l, :].to(self._device)
            all_rot1 = self.grs[i, f1l, :].to(self._device)

            local_rot0 = self.lrs[i, f0l].to(self._device)
            local_rot1 = self.lrs[i, f1l].to(self._device)
            
            self._key_body_ids = self._key_body_ids.to('cpu')
            key_pos0 = self.gts[i, f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)].to(self._device)
            key_pos1 = self.gts[i, f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)].to(self._device)
            
            dof_vel = self.dvs[i, f0l].to(self._device)

            vals = [all_pos0, all_pos1, all_rot0, all_rot1, root_vel, root_ang_vel]
            for v in vals:
                assert v.dtype != torch.float64

            blend = blend.unsqueeze(-1)
            blend_exp = blend.unsqueeze(-1)
            
            key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
            all_pos = (1.0 - blend_exp) * all_pos0 + blend_exp * all_pos1
            all_rot = torch_utils.slerp(all_rot0, all_rot1, blend_exp)
            # print('++++get_state++++', root_pos.shape, all_pos0.shape, all_pos.shape, key_pos.shape, root_rot0.shape, root_rot.shape, all_rot0.shape, all_rot.shape
            #       , root_rot[0], all_rot[0,0,:], root_pos[0], all_pos[0,0,:])
            local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof(local_rot)
           #print("root pose shape CHECK:" , local_rot.shape)
            dof_poses.append(dof_pos)
            all_poses.append(all_pos)
            all_rots.append(all_rot)
            root_vels.append(root_vel)
            root_ang_vels.append(root_ang_vel)
            dof_vels.append(dof_vel)
            key_poses.append(key_pos)
        
        all_poses = torch.stack(all_poses, dim=1)
        all_rots = torch.stack(all_rots, dim=1)
        root_vels = torch.stack(root_vels, dim=1)
        root_ang_vels = torch.stack(root_ang_vels, dim=1)
        dof_poses = torch.stack(dof_poses, dim=1)
        dof_vels = torch.stack(dof_vels, dim=1)
        key_poses = torch.stack(key_poses, dim=1)

        # print('++++get_state++++', all_poses.shape, all_rots.shape, root_poses.shape, root_rots.shape, root_vels.shape, root_ang_vels.shape, dof_vels.shape,
        #       key_poses.shape)
        return all_poses, all_rots, root_vels, root_ang_vels, dof_vels, dof_poses, key_poses    
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_annots = []

        total_len = 0.0
                    
        # clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        #motion_files, motion_weights, annot_files = self._fetch_motion_files(motion_file)
        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(0, num_motion_files, 2):

            curr_file_person1 = motion_files[f]
            curr_file_person2 = motion_files[f + 1]
            print("Loading {:d}/{:d} motion files: {:s} and {:s}".format(f // 2 + 1, num_motion_files // 2, curr_file_person1, curr_file_person2))


            curr_motion_person1 = SkeletonMotion.from_file(curr_file_person1)
            curr_motion_person2 = SkeletonMotion.from_file(curr_file_person2)


            motion_fps = curr_motion_person1.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion_person1.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            # if num_frames > 120:
            #     print('too short!')
            #     continue
            
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)


            curr_dof_vels_person1 = self._compute_motion_dof_vels(curr_motion_person1)
            curr_dof_vels_person2 = self._compute_motion_dof_vels(curr_motion_person2)

            #print('++++++read_motions',curr_motion_person1.global_translation.shape)
            curr_motion_person1.dof_vels = curr_dof_vels_person1
            curr_motion_person2.dof_vels = curr_dof_vels_person2
            curr_motion = [curr_motion_person1, curr_motion_person2]
            # if USE_CACHE:
            #     curr_motion = DeviceCache(curr_motion, self._device)
            # else:
            #     curr_motion_person1.tensor = curr_motion_person1.tensor.to(self._device)
            #     curr_motion_person2.tensor = curr_motion_person2.tensor.to(self._device)
            #     curr_motion_person1._skeleton_tree._parent_indices = curr_motion_person1._skeleton_tree._parent_indices.to(self._device)
            #     curr_motion_person1._skeleton_tree._local_translation = curr_motion_person1._skeleton_tree._local_translation.to(self._device)
            #     curr_motion_person1._rotation = curr_motion_person1._rotation.to(self._device)
            #     curr_motion_person2._skeleton_tree._parent_indices = curr_motion_person2._skeleton_tree._parent_indices.to(self._device)
            #     curr_motion_person2._skeleton_tree._local_translation = curr_motion_person2._skeleton_tree._local_translation.to(self._device)
            #     curr_motion_person2._rotation = curr_motion_person2._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)


            curr_weight_person1 = motion_weights[f]
            curr_weight_person2 = motion_weights[f + 1]
            self._motion_weights.append([curr_weight_person1, curr_weight_person2])
            self._motion_files.append([curr_file_person1, curr_file_person2])

        self._motion_weights = torch.tensor([w1 + w2 for w1, w2 in self._motion_weights], dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        #self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        #print("weights shape CHECK:", self._motion_weights)
        #self._motion_weights /= self._motion_weights.sum(dim=1, keepdim=True)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        #print("++++++annots", self._motion_annots[0].shape)

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        
        self._adjust_weights()
        print("Loaded {:d} motion pairs with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights
    
    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion[0].num_joints
        return num_bodies

    def _adjust_weights(self):
        total_num_frames = self.get_total_frames()
        self._motion_weights = (self._motion_num_frames / total_num_frames).clone()
        #print(self._motion_weights)

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets
        #print('++++++++local_to_dof', local_rot.shape, self._num_dof)
        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)
        #print("CheCK:", body_ids, dof_offsets, dof_pos.shape)
        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                #print('-------', joint_q.shape)

                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                #print("joint_exp_map shape CHECK:", joint_exp_map.shape)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                #print('++++++++jlocal_to_dof', joint_q.shape, joint_q[0])
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #print('++++++++alocal_to_dof', joint_axis[0])
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis
                #print('++++++++tlocal_to_dof', joint_theta.shape, joint_theta[0])
                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel