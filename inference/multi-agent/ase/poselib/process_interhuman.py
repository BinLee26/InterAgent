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

smpl_bone_order_names = ["Pelvis",
"L_Hip",
"R_Hip",
"Torso",
"L_Knee",
"R_Knee",
"Spine",
"L_Ankle",
"R_Ankle",
"Chest",
"L_Toe",
"R_Toe",
"Neck",
"L_Thorax",
"R_Thorax",
"Head",
"L_Shoulder",
"R_Shoulder",
"L_Elbow",
"R_Elbow",
"L_Wrist",
"R_Wrist",
"L_Hand",
"R_Hand",]

smpl_mujoco_names = ['Pelvis', 
 'L_Hip', 'L_Knee', 
 'L_Ankle', 'L_Toe', 
 'R_Hip', 'R_Knee', 
 'R_Ankle', 'R_Toe', 
 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 
 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

def get_local_translation(smpl_params, p):
    trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]]).float()
    #print(smpl_params['person1'].keys())
    gender = smpl_params[p]['gender']
    if gender == "male":
        bm_path = "ase/poselib/body_model/smpl/smpl_m_beta10.pkl"
    else:
        bm_path = "ase/poselib/body_model/smpl/smpl_n_beta10.pkl"
        
    original_sizes = [
        torch.Size([207, 69]),
        torch.Size([207, 90]),
        torch.Size([207, 10]),
        torch.Size([207, 3])
    ]

    new_sizes = [torch.Size([1, size[1]]) for size in original_sizes]
    new_tensors = [torch.zeros(size).float() for size in new_sizes]
    #print(new_tensors[0].shape)

    trans_np = smpl_params[p]["trans"]
    trans = trans_np[0].reshape(1, -1)
    trans = torch.from_numpy(trans).float()
    # print(trans)
    bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, model_type="smpl", gender=gender)
    betas = torch.from_numpy(smpl_params[p]["betas"]).reshape(1,-1)
    print(betas)
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
    smpl_node_names = ["Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",]
    smpl_parent_indices = torch.tensor([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
    smpl_skeleton_tree = SkeletonTree(smpl_node_names,smpl_parent_indices,smpl_local_translation)
    return smpl_skeleton_tree

def process_smpl(source_motion_path, target_motion_path):
    mujoco_mjcf_file = 'ase/data/assets/mjcf/smpl_humanoid.xml'
    mujoco_sk_tree = SkeletonTree.from_mjcf(mujoco_mjcf_file)
    # load source motion, which is a pkl file
    with open(source_motion_path, 'rb') as f:
        smpl_params = pickle.load(f)
    print("retargeting:", source_motion_path)
    if(smpl_params['person1']['trans'].size == 0):
        print("empty! retargeting next")
        return
    for p in ['person1','person2']:
        #print(smpl_params[p]['betas'])
        #print(smpl_params['person1']['gender'])
        smpl_local_translation = get_local_translation(smpl_params, p)
        height_offset = smpl_local_translation[0][2].clone()
        smpl_local_translation[0] = torch.tensor([0.0,0.0,0.0])
        #print(height_offset)
        #smpl_skeleton_tree = generate_smpl_skeleton_tree(smpl_local_translation)
        #print('++++smpl_skeleton_tree', smpl_skeleton_tree.node_names)
        smpl_2_mujoco = [smpl_bone_order_names.index(q) for q in smpl_mujoco_names if q in smpl_bone_order_names]
        trans = torch.tensor(smpl_params[p]['trans'])
        num_frame = len(smpl_params[p]['pose_body'])
        root_orient = torch.tensor(smpl_params[p]['root_orient'])
        pose_body = torch.tensor(smpl_params[p]['pose_body'])
        print('++++',pose_body.shape)
        res = torch.zeros((num_frame, 6))
        rot_vecs = torch.cat((root_orient, pose_body, res), dim=1)
        assert rot_vecs.shape == (num_frame, 72) 
        rot_vecs = rot_vecs.reshape(-1,24,3)[:,smpl_2_mujoco] # convert to mujoco
        rots = axis_angle_to_quaternion(rot_vecs)
        smpl_skeleton_state = SkeletonState.from_rotation_and_root_translation(mujoco_sk_tree, rots, trans)
        #print('++++',smpl_skeleton_state.skeleton_tree)
        #plot_skeleton_state(smpl_skeleton_state)
        target_motion = SkeletonMotion.from_skeleton_state(smpl_skeleton_state, 59.94)
        if p == 'person1':
            target_path = os.path.splitext(target_motion_path)[0] + f'_person1.npy'
            d1 = target_motion.to_dict()
        else:
            target_path = os.path.splitext(target_motion_path)[0] + f'_person2.npy'
            d2 = target_motion.to_dict()
        # plot_skeleton_motion_interactive(source_motion)
        target_motion.to_file(target_path)
    return

def main():
    
    source_path = 'ase/data/InterGenTest/'
    target_path = 'ase/data/motions/intergen_smpl/'
        
    mujoco_mjcf_file = 'ase/data/assets/mjcf/smpl_humanoid.xml'
    mujoco_sk_tree = SkeletonTree.from_mjcf(mujoco_mjcf_file)
    #print('+++mujoco',mujoco_sk_tree.node_names)
    for root, dirs, filelist in os.walk(source_path):
        for filename in filelist:
            #if filename == 'smpl_params.pkl':
                #a += 1
                source_motion_path = os.path.join(root, filename)
                target_motion_path = os.path.join(target_path, filename)
                
                process_smpl(source_motion_path, target_motion_path)
    return

if __name__ == '__main__':
    main()