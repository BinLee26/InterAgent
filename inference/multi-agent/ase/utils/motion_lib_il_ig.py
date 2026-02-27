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
from utils.misc.interaction import Interaction
import utils.motion_lib_il as motion_lib_il

import torch

class MotionLibIlIg(motion_lib_il.MotionLibIl):
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device):
        super().__init__(
            motion_file=motion_file,
            dof_body_ids=dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device=device
        )
        
        self.ref_interaction_mesh = []
        
        ig_poses = self.gts.permute(1,0,2,3)
        # print('++++++++++igil',ig_poses.shape)
        self.delaunay_edge_indices = []
        for m in range(len(self._motions)):
            cur_motion_edge_indices = []
            for t in range(self._motion_num_frames[m]):
                cur_pose_all = ig_poses[t].reshape(-1,3) # (2,num_joints,3)->(2*num_joints,3)
                cur_ig = Interaction(cur_pose_all)
                cur_frame_edge_indices = cur_ig.build_interaction_graph()
                cur_motion_edge_indices.append(cur_frame_edge_indices)
            self.delaunay_edge_indices.append(cur_motion_edge_indices)
            print(f"loaded delaunay interaction graph for motion {m}, length {len(cur_motion_edge_indices)}")
        print(f"loaded all {len(self.delaunay_edge_indices)} interaction graph")
        
    def get_ig_edge_indices(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        #print('++++motion_times_dual', motion_times, motion_times.shape)
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        
        return [self.delaunay_edge_indices[motion_ids[i]][frame_idx0[i]] for i in range(motion_ids.size(0))]