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
import math
import numpy as np

import torch.nn.functional as F
from isaacgym.torch_utils import *

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

@torch.jit.script
def matrix_to_quaternion_ijkr(matrix: torch.Tensor) -> torch.Tensor:
    """
    x, y, z, w
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
            1.0 + m00 + m11 + m22,
        ],
        dim=-1,
    ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_ijkr = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m10 + m01, q_abs[..., 1]**2, m21 + m12, m02 - m20], dim=-1),
            torch.stack([m02 + m20, m12 + m21, q_abs[..., 2]**2, m10 - m01], dim=-1),
            torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 3]**2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_ijkr / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))  # pyre-ignore[16]

@torch.jit.script
def tan_norm_to_mat(tan_norm):
    B = tan_norm.shape[0]
    tan = tan_norm.view(-1, 2, 3)[:, 0]
    norm = tan_norm.view(-1, 2, 3)[:, 1]
    tan_n = F.normalize(tan, dim=-1)

    norm_n = norm - (tan_n * norm).sum(-1, keepdim=True) * tan_n
    norm_n = F.normalize(norm_n, dim=-1)

    cross = torch.cross(norm_n, tan_n)

    rot_mat = torch.stack([tan_n, cross, norm_n], dim=-1).reshape(B, -1, 3, 3)
    return rot_mat


@torch.jit.script
def tan_norm_to_quat(tan_norm):
    B = tan_norm.shape[0]
    rot_mat = tan_norm_to_mat(tan_norm)
    quat_new = matrix_to_quaternion_ijkr(rot_mat).view(B, -1, 4)
    return quat_new

@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map

@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta);
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
    ratioB = torch.sin(t * half_theta) / sin_half_theta; 
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    #print('+++++heading', heading.shape)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

@torch.jit.script
def quat_from_yaw(yaw):
    half_yaw = yaw / 2
    qw = torch.cos(half_yaw)
    qy = torch.sin(half_yaw)
    zeros = torch.zeros_like(qw)
    ones = torch.ones_like(qw)
    return torch.stack([zeros, zeros, qy, qw], dim=-1)  # shape (N, 4)

@torch.jit.script
def normalize_quat(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return  torch.div(q, norm[..., None])

@torch.jit.script
def quat_multiply(q1, q2):
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)

@torch.jit.script
def quat_conj(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)
    return q * conj.expand_as(q)

@torch.jit.script
def quat_diff(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    q1_inv = quat_conj(q1)
    q_diff = quat_multiply(q1_inv, q2)
    # return quat_multiply(q1, q2_conj)
    return q_diff

@torch.jit.script
def wrap_angle(theta):
    """
    Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
    Odd multiples of pi are wrapped to +pi (as opposed to -pi).
    """
    pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    result[result.eq(-pi_tensor)] = math.pi

    return result

@torch.jit.script
def q_angle(q):
    """
    Determine the rotation angle of given quaternion tensors of shape [*, 4].
    Return as tensor of shape [*, 1]
    """
    assert q.shape[-1] == 4

    q = normalize_quat(q)
    q_re, q_im = torch.split(q, [1, 3], dim=-1)
    norm = torch.linalg.norm(q_im, dim=-1).unsqueeze(dim=-1)
    angle = 2.0 * torch.atan2(norm, q_re)

    return wrap_angle(angle)

@torch.jit.script
def quat_dist(q1,q2):
    q_diff = quat_diff(q1, q2)
    theta = q_angle(q_diff)
    return theta
