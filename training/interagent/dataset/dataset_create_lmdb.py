import os
import pickle
import clip
import torch

import numpy as np
from tqdm import tqdm

import lmdb

def compute_interaction_graph_obs_batch_pos_local(local_body_pos1, local_body_pos2):

    B, F, A, J, _ = local_body_pos1.shape  # A = 2 agents, J = 15 joints

    rigid_body_pos1 = local_body_pos1.reshape(B * F, A, J, 3)
    rigid_body_pos2 = local_body_pos2.reshape(B * F, A, J, 3)

    intergraph_obs_buf = torch.zeros((B * F, 2, J, J, 3), device=rigid_body_pos1.device)

    agent0_pos = rigid_body_pos1[:, 0].unsqueeze(2)  # (B*F, 15, 1, 3)
    agent1_pos = rigid_body_pos1[:, 1].unsqueeze(1)  # (B*F, 1, 15, 3)

    intergraph_obs_buf[:, 0, :, :, :] = agent1_pos - agent0_pos

    agent1_pos = rigid_body_pos2[:, 0].unsqueeze(2)
    agent0_pos = rigid_body_pos2[:, 1].unsqueeze(1)

    intergraph_obs_buf[:, 1, :, :, :] = agent0_pos - agent1_pos

    return intergraph_obs_buf.reshape(B, F, 2, J, J, 3)

clip_model, _ = clip.load("ViT-L/14@336px", "cpu", jit=False)
clip_model.eval() # frozen clip model

for param in clip_model.parameters():
    param.requires_grad = False

horizon = 368
obs_length = 364
motion_path = '/data/interhuman_train.pkl'
text_path = '/data/texts'

with open(motion_path, "rb") as f:
    motion_data = pickle.load(f)
motion_name = list(motion_data.keys())
motion_nums = len(motion_data)


two_state_save_dir = '/data/two_state'
state_ig_save_dir = '/data/state_ig'
action_mean_save_dir = '/data/action'


os.makedirs(two_state_save_dir, exist_ok=True)
os.makedirs(state_ig_save_dir, exist_ok=True)
os.makedirs(action_mean_save_dir, exist_ok=True)

env = lmdb.open("/data/data.lmdb", map_size=int(8e12))
file_idx = 0


for i in tqdm(range(motion_nums)):

    motion_name_i = motion_name[i]

    motion_i = motion_data[motion_name_i]

    i_text_path = os.path.join(text_path, motion_name_i+'.txt')
    with open(i_text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with torch.no_grad():  # 关闭梯度计算
        text_tokens = clip.tokenize(lines,truncate=True)
        text_features = clip_model.encode_text(text_tokens)

    min_start_idx = 0

    pad_width = ((0, 0), (obs_length-1, horizon - obs_length -1), (0, 0), (0, 0))
    pad_width_ig = ((0, 0), (obs_length-1, horizon - obs_length -1), (0, 0), (0, 0),(0, 0), (0, 0))
    pad_width_pos_rot = ((0, 0), (obs_length-1, horizon - obs_length -1), (0, 0), (0, 0),(0, 0))

    motion_obs = np.pad(motion_i['obs'][:,:-1,:,:],pad_width,mode='edge')
    motion_action = np.pad(motion_i['clean_action'][:,:-1,:,:],pad_width,mode='edge')

    frame_num = motion_obs.shape[1]
    max_start_idx = frame_num - horizon

    rollout_num = motion_obs.shape[0]
    person_num = motion_obs.shape[2]

    #local coordinate
    padding_xy = np.zeros((rollout_num,frame_num, 3))
    local_body_pos_11 = np.concatenate([padding_xy,motion_obs[:,:,0,1:43]],axis=-1).reshape(rollout_num,frame_num, 1, 15, 3)
    local_body_pos_12 = motion_obs[:,:,0,223:268].reshape(rollout_num,frame_num, 1, 15, 3)
    local_body_pos_1 = np.concatenate([local_body_pos_11,local_body_pos_12], axis=2)
    local_body_pos_21 = np.concatenate([padding_xy,motion_obs[:,:,1,1:43]], axis=-1).reshape(rollout_num,frame_num, 1, 15, 3)
    local_body_pos_22 = motion_obs[:,:,1,223:268].reshape(rollout_num,frame_num, 1, 15, 3)
    local_body_pos_2 = np.concatenate([local_body_pos_21,local_body_pos_22], axis=2)

    local_pos_1 = torch.from_numpy(local_body_pos_1)
    local_pos_2 = torch.from_numpy(local_body_pos_2)

    intergraph_obs_buf = compute_interaction_graph_obs_batch_pos_local(local_pos_1, local_pos_2)
    motion_ig = intergraph_obs_buf.numpy()

    motion_ig_reshape = motion_ig.reshape(motion_obs.shape[0], motion_obs.shape[1], motion_obs.shape[2], -1)

    state_ig_list = []
    two_state_list = []
    action_list = []
    ig_list = []
    txn =  env.begin(write=True)


    for j in range(rollout_num):
        min_start_idx = j
        for idx in range(min_start_idx, max_start_idx + 1, 8):
            start_idx = max(idx, 0)
            end_idx = min(idx + horizon, frame_num)

            save_index =  np.round(np.linspace(start_idx, end_idx-8, 12)).astype(int).tolist() + list(range(end_idx-8, end_idx))
            idx_obs = motion_obs[j, save_index, :, :]
            idx_action = motion_action[j, save_index, :, :]
            idx_ig = motion_ig_reshape[j, save_index, :, :]

            two_state_obs = np.concatenate([idx_obs[:,0,:], idx_obs[:,1,:]], axis=-1)
            two_action = np.concatenate([idx_action[:,0,:], idx_action[:,1,:]], axis=-1)
            obs_0 = np.concatenate([idx_obs[:,0,:223], idx_ig[:,0,:]], axis=-1)
            obs_1 = np.concatenate([idx_obs[:,1,:223], idx_ig[:,1,:]], axis=-1)
            state_ig_obs = np.concatenate([obs_0, obs_1], axis=-1)

            two_obs_own = np.concatenate([idx_obs[:,0,:], idx_obs[:,1,:]], axis=-1)

            idx_data = {
                'obs': two_obs_own.astype(np.float16),
                'action': two_action.astype(np.float16),
                'text_features_list':text_features.numpy().astype(np.float16)
            }

            txn.put(str(file_idx).encode(), pickle.dumps(idx_data))
            file_idx += 1

            state_ig_list.append(state_ig_obs.astype(np.float16))
            two_state_list.append(two_state_obs.astype(np.float16))
            action_list.append(two_action.astype(np.float16))

    txn.commit()
    state_ig_max = np.max(state_ig_list, axis=0)
    state_ig_min = np.min(state_ig_list, axis=0)
    two_state_max = np.max(two_state_list, axis=0)
    two_state_min = np.min(two_state_list, axis=0)

    action_max = np.max(action_list, axis=0)
    action_min = np.min(action_list, axis=0)

    state_ig_file_min = os.path.join(state_ig_save_dir, f"state_ig_min_{motion_name_i}.npy")
    state_ig_file_max = os.path.join(state_ig_save_dir, f"state_ig_max_{motion_name_i}.npy")

    two_state_file_min = os.path.join(two_state_save_dir, f"two_state_min_{motion_name_i}.npy")
    two_state_file_max = os.path.join(two_state_save_dir, f"two_state_max_{motion_name_i}.npy")
    action_file_min = os.path.join(action_mean_save_dir, f"action_min_{motion_name_i}.npy")
    action_file_max = os.path.join(action_mean_save_dir, f"action_max_{motion_name_i}.npy")

    np.save(state_ig_file_min, state_ig_min)
    np.save(state_ig_file_max, state_ig_max)

    np.save(two_state_file_min, two_state_min)
    np.save(two_state_file_max, two_state_max)
    np.save(action_file_min, action_min)
    np.save(action_file_max, action_max)

txn =  env.begin(write=True)
txn.put(b"__len__", pickle.dumps(int(file_idx)))
txn.commit()
