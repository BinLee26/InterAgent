from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R
from body_model.body_model import BodyModel

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive, plot_skeleton_states

def get_local_translation(smpl_params, p):
    trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]]).float()
    #print(smpl_params['person1'].keys())
    gender = smpl_params[p]['gender']
    if gender == "male":
        bm_path = "body_model/smpl/smpl_m_beta10.pkl"
    else:
        bm_path = "body_model/smpl/smpl_n_beta10.pkl"

    # 原始张量尺寸示例
    original_sizes = [
        torch.Size([207, 69]),
        torch.Size([207, 90]),
        torch.Size([207, 10]),
        torch.Size([207, 3])
    ]

    # 新的全零张量尺寸列表
    new_sizes = [torch.Size([1, size[1]]) for size in original_sizes]
    new_tensors = [torch.zeros(size).float() for size in new_sizes]
    #print(new_tensors[0].shape)

    trans_np = smpl_params[p]["trans"]
    trans = trans_np[0].reshape(1, -1)
    trans = torch.from_numpy(trans).float()
    # print(trans)
    bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, model_type="smpl", gender=gender)
    betas = torch.from_numpy(smpl_params[p]["betas"]).reshape(1,-1)
    #print(betas)
    #print(new_tensors[2])
    with torch.no_grad():
        body = bm(pose_body=new_tensors[0], betas=betas, root_orient=new_tensors[3], trans=trans)
    #print(new_tensors[3])
    # ground = body.Jtr[:,:,2].min()
    # print(ground)
    # trans = trans - ground
    joints_full = body['Jtr']
    # print(joints_full.shape)
    # joints_full = joints_full[:,:62,:]
    joints_full = torch.einsum("mn,tn->tm", trans_matrix, joints_full.reshape(-1,3)).reshape(1, 24,3)

    parent_indices = {
        0: -1, 
        1: 0,  
        2: 0,  
        3: 0,  
        4: 1,  
        5: 2,  
        6: 3,  
        7: 4,  
        8: 5,  
        9: 6,  
        10: 7, 
        11: 8,
        12: 9,
        13: 9,
        14: 9,
        15: 12,
        16: 13,
        17: 14,
        18: 16,
        19: 17,
        20: 18, 
        21: 19,
        22: 20,
        23: 21
    }
    # 用来存储本地平移向量的张量
    local_translations = torch.zeros([24,3])
    # print(local_translations.shape)
    # print(trans)
    local_translations[0] = trans
    # 计算每个节点的本地平移向量
    for child_index in range(1, 24):
        parent_index = parent_indices[child_index]
        # print(parent_index,child_index)
        local_translations[child_index] = joints_full[0][child_index] - joints_full[0][parent_index]
    #print(local_translations)
    return local_translations

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [axis_angle * sin_half_angles_over_angles, torch.cos(half_angles)], dim=-1
    )
    return quaternions

def generate_smpl_skeleton_tree(smpl_local_translation):
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
    # smpl_local_translation = torch.tensor([[ 0.0000,  0.0000,  0.8585],
    #     [ 0.0586, -0.0823, -0.0177],
    #     [-0.0603, -0.0905, -0.0135],
    #     [ 0.0044,  0.1244, -0.0384],
    #     [ 0.0435, -0.3865,  0.0080],
    #     [-0.0433, -0.3837, -0.0048],
    #     [ 0.0045,  0.1380,  0.0268],
    #     [-0.0148, -0.4269, -0.0374],
    #     [ 0.0191, -0.4200, -0.0346],
    #     [-0.0023,  0.0560,  0.0029],
    #     [ 0.0411, -0.0603,  0.1220],
    #     [-0.0348, -0.0621,  0.1303],
    #     [-0.0134,  0.2116, -0.0335],
    #     [ 0.0717,  0.1140, -0.0189],
    #     [-0.0830,  0.1125, -0.0237],
    #     [ 0.0101,  0.0889,  0.0504],
    #     [ 0.1229,  0.0452, -0.0190],
    #     [-0.1132,  0.0469, -0.0085],
    #     [ 0.2553, -0.0156, -0.0229],
    #     [-0.2601, -0.0144, -0.0313],
    #     [ 0.2657,  0.0127, -0.0074],
    #     [-0.2691,  0.0068, -0.0060],
    #     [ 0.0867,  0.0156, -0.0106],
    #     [-0.0888,  0.0101, -0.0087]])
    smpl_skeleton_tree = SkeletonTree(smpl_node_names,smpl_parent_indices,smpl_local_translation)
    return smpl_skeleton_tree

def compute_scale(local_translation, amp_leg_length):
    pdist = torch.nn.PairwiseDistance(p=2)
    smpl_leg_length = pdist(torch.zeros(3), local_translation[4]) + pdist(torch.zeros(3), local_translation[7])
    smpl_leg_length = smpl_leg_length.item()
    #print(smpl_leg_length)
    return amp_leg_length / smpl_leg_length

def project_joints(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def retarget_motion(source_motion_path, target_motion_path):
    # load source motion, which is a pkl file
    with open(source_motion_path, 'rb') as f:
        smpl_params = pickle.load(f)
    
    retarget_data_path = "data/configs/retarget_smpl_to_amp.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)
    #plot_skeleton_state(zero_pose)

    #source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])
    #convert source motion to SkeletonMotion format
    #source_motion = SkeletonMotion.from_file(source_motion_path)
    amp_local_translation = target_tpose.local_translation
    #amp_skeleton_tree = target_tpose.skeleton_tree
    #amp_local_rotation = target_tpose.local_rotation
    # amp_local_rotation[0] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
    #     amp_local_rotation[0]
    # )
    #plot_skeleton_state(target_tpose)
    #print(amp_skeleton_tree)
    pdist = torch.nn.PairwiseDistance(p=2)
    amp_leg_length = pdist(torch.zeros(3), amp_local_translation[13]) + pdist(torch.zeros(3), amp_local_translation[14])
    amp_leg_length = amp_leg_length.item()
    #print(amp_leg_length)
    for p in ['person1','person2']:
        #print(smpl_params[p]['betas'])
        #print(smpl_params['person1']['gender'])
        smpl_local_translation = get_local_translation(smpl_params, p)
        height_offset = smpl_local_translation[0][2].clone()
        smpl_local_translation[0] = torch.tensor([0.0,0.0,0.0])
        #print(smpl_local_translation)
        smpl_skeleton_tree = generate_smpl_skeleton_tree(smpl_local_translation)


        print(height_offset)
        # load t-pose files
        #print(smpl_skeleton_tree.local_translation)

        zero_pose = SkeletonState.zero_pose(smpl_skeleton_tree)
        #zero_pose = SkeletonState.from_file(retarget_data["source_tpose"])

        local_rotation = zero_pose.local_rotation
        #plot_skeleton_state(zero_pose)
        local_rotation[0] = quat_mul(quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), local_rotation[0])
        local_rotation[0] = quat_mul(quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), local_rotation[0])
        #source_tpose = zero_pose
        #print(target_tpose.local_rotation)
        #plot_skeleton_state(zero_pose)
        #print(smpl_params[p].keys())
        trans = torch.tensor(smpl_params[p]['trans'])
        rot_vecs = torch.tensor(smpl_params[p]['poses'].reshape(-1,24,3))
        rots = axis_angle_to_quaternion(rot_vecs)
        smpl_skeleton_state = SkeletonState.from_rotation_and_root_translation(smpl_skeleton_tree, rots, trans)
        source_motion = SkeletonMotion.from_skeleton_state(smpl_skeleton_state, smpl_params[p]['mocap_framerate'])
        motion_scale = compute_scale(smpl_local_translation, amp_leg_length)
        #motion_scale = retarget_data["scale"]
        #print(motion_scale)
        #print(trans)
        #plot_skeleton_motion_interactive(source_motion)
    # parse data from retarget config
        joint_mapping = retarget_data["joint_mapping"]
        rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])
        #rotation_to_target_skeleton = quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True)
        #rotation_to_target_skeleton = quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True)
        #rotation_to_target_skeleton = quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True)
        #rotation_to_target_skeleton = rotation_to_target_skeleton[0]
        #print(rotation_to_target_skeleton)
        #plot_skeleton_state(source_tpose)
        #plot_skeleton_state(target_tpose)
        # [0.707,0.0,0.707,0.0]
        # [0.0,0.0,0.0,1.0]
        # run retargeting
        target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=joint_mapping,
        source_tpose=zero_pose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=motion_scale
        )

        # keep frames between [trim_frame_beg, trim_frame_end - 1]
        frame_beg = retarget_data["trim_frame_beg"]
        frame_end = retarget_data["trim_frame_end"]
        if (frame_beg == -1):
            frame_beg = 0
            
        if (frame_end == -1):
            frame_end = target_motion.local_rotation.shape[0]
            
        local_rotation = target_motion.local_rotation
        root_translation = target_motion.root_translation
        local_rotation = local_rotation[frame_beg:frame_end, ...]
        root_translation = root_translation[frame_beg:frame_end, ...]
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)
        #plot_skeleton_motion_interactive(target_motion)
        # need to convert some joints from 3D to 1D (e.g. elbows and knees)
        target_motion = project_joints(target_motion)

        # move the root so that the feet are on the ground
        local_rotation = target_motion.local_rotation
        root_translation = target_motion.root_translation
        tar_global_pos = target_motion.global_translation
        min_h = torch.min(tar_global_pos[..., 2])
        root_translation[:, 2] += -min_h
        print(min_h)
        # adjust the height of the root to avoid ground penetration
        root_height_offset = retarget_data["root_height_offset"]
        #root_height_offset = 0.9 - height_offset
        root_translation[:, 2] += root_height_offset
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)
        #print(target_motion.root_translation[:5,:])
        #(target_motion.rotation[:5,:])

        # save retargeted motion
        if p == 'person1':
            target_path = os.path.splitext(target_motion_path)[0] + f'_person1.npy'
            d1 = target_motion.to_dict()
        else:
            target_path = os.path.splitext(target_motion_path)[0] + f'_person2.npy'
            d2 = target_motion.to_dict()
        target_motion.to_file(target_path)
    #d = {'person1':d1, 'person2':d2}
    #print(d)
        #plot_skeleton_motion_interactive(target_motion)
    return

def main():
    source_path = '../data/InterGen/motions'
    #target_path = '../data/motions/interhuman_amp_split/'
    target_path = '../data/motions/InterGen_amp'
    #source_motion_path = source_path + r"1014_taekwondo/3/smpl_params.pkl"
    #source_motion_path = source_path + r"1208_dance/1/smpl_params.pkl"
    #source_motion_path = source_path + r"1005_daily_motion/1/smpl_params.pkl"
    #target_motion_path = target_path + r"dance.npy"
    #target_motion_path = target_path + r"tawkwondo3.npy"
    for root, dirs, filelist in os.walk(source_path):
        for filename in filelist:
            #if filename == 'smpl_params.pkl':
                #a += 1
                source_motion_path = os.path.join(root, filename)
                target_motion_path = os.path.join(target_path, filename)
                #print(source_motion_path)
                #print(target_motion_path)
    #print(target_motion_path.rstrip('.npy')+'_'+'person1'+r'.npy')
                retarget_motion(source_motion_path, target_motion_path)
    return

if __name__ == '__main__':
    main()