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

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

smpl_node_names = ['pelvis',
        'left leg root', 'right leg root',
        'lowerback',
        'left knee', 'right knee',
        'upperback',
        'left ankle', 'right ankle',
        'thorax',
        'left toes', 'right toes',
        'lowerneck',
        'left clavicle', 'right clavicle',
        'upperneck',
        'left armroot', 'right armroot',
        'left elbow', 'right elbow',
        'left wrist', 'right wrist',
        'left hand', 'right hand']
smpl_parent_indices = torch.tensor([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
smpl_local_translation = torch.tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0586, -0.0823, -0.0177],
        [-0.0603, -0.0905, -0.0135],
        [ 0.0044,  0.1244, -0.0384],
        [ 0.0435, -0.3865,  0.0080],
        [-0.0433, -0.3837, -0.0048],
        [ 0.0045,  0.1380,  0.0268],
        [-0.0148, -0.4269, -0.0374],
        [ 0.0191, -0.4200, -0.0346],
        [-0.0023,  0.0560,  0.0029],
        [ 0.0411, -0.0603,  0.1220],
        [-0.0348, -0.0621,  0.1303],
        [-0.0134,  0.2116, -0.0335],
        [ 0.0717,  0.1140, -0.0189],
        [-0.0830,  0.1125, -0.0237],
        [ 0.0101,  0.0889,  0.0504],
        [ 0.1229,  0.0452, -0.0190],
        [-0.1132,  0.0469, -0.0085],
        [ 0.2553, -0.0156, -0.0229],
        [-0.2601, -0.0144, -0.0313],
        [ 0.2657,  0.0127, -0.0074],
        [-0.2691,  0.0068, -0.0060],
        [ 0.0867,  0.0156, -0.0106],
        [-0.0888,  0.0101, -0.0087]])
smpl_skeleton_tree = SkeletonTree(smpl_node_names,smpl_parent_indices,smpl_local_translation / 10)
#print(smpl_skeleton_tree)
smpl_zero_pose = SkeletonState.zero_pose(smpl_skeleton_tree)
#print(smpl_zero_pose.local_rotation)

amp_tpose = SkeletonState.from_file('data/amp_humanoid_tpose.npy')
amp_local_rotation = amp_tpose.local_rotation
amp_local_rotation[0] = quat_mul(
quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
amp_local_rotation[0]
)
#plot_skeleton_state(amp_tpose)
#print(smpl_params['person1'].keys())
zero_pose = SkeletonState.zero_pose(smpl_skeleton_tree)
local_rotation = zero_pose.local_rotation
local_rotation[0] = quat_mul(
quat_from_angle_axis(angle=torch.tensor([0.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
local_rotation[0]
)
local_rotation[0] = quat_mul(
quat_from_angle_axis(angle=torch.tensor([0.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
local_rotation[0]
)
local_rotation[0] = quat_mul(
quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
local_rotation[0]
)
# translation = zero_pose.root_translation
# translation += torch.tensor([0, 0, 0.9])
# save and visualize T-pose
zero_pose.to_file("data/smpl_humanoid_tpose.npy")
plot_skeleton_state(zero_pose)