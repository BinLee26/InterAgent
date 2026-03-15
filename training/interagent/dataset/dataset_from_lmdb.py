import os
import random
from torch.utils.data import Dataset
import torch
from training.utils.data import dict_apply
from training.utils.normalizer import LinearNormalizer
import numpy as np
import lmdb

import pickle


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


class DiffusionPolicyDataset(Dataset):
    def __init__(self, motion_path, obs_scale_path, action_scale_path):

        self.motion_path_dir = motion_path
        self.obs_path = obs_scale_path
        self.action_path = action_scale_path

        self.all_obs = []
        self.all_action = []
        self.env = lmdb.open(motion_path, readonly=True, lock=False)

        self.txn = self.env.begin()
        self.length = pickle.loads(self.txn.get(b"__len__"))

    def get_normalizer(self):

        obs_path_list = os.listdir(self.obs_path)
        action_path_list = os.listdir(self.action_path)
        for i in range(len(obs_path_list)):
            obs_i = np.load(os.path.join(self.obs_path, obs_path_list[i]))
            self.all_obs.append(obs_i)
        for i in range(len(action_path_list)):
            action_i = np.load(os.path.join(self.action_path, action_path_list[i]))
            self.all_action.append(action_i)

        data = {
            'obs': np.concatenate(self.all_obs, axis=0),
            'action': np.concatenate(self.all_action, axis=0),
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode='limits')
        return normalizer

    def __len__(self) -> int:
        return int(self.length)

    def __getitem__(self, idx):


        data_byte = self.txn.get(str(idx).encode())
        data = pickle.loads(data_byte)

        data['text_feature'] = random.choice(data['text_features_list'])

        motion_obs = data['obs']
        frame_num = motion_obs.shape[0]

        ###ig  pos  rot

        #local coordinate
        padding_xy = np.zeros((motion_obs.shape[0], 3))
        local_body_pos_11 = np.concatenate([padding_xy,motion_obs[:,1:43]],axis=1).reshape(-1, 1, 15, 3)
        local_body_pos_12 = motion_obs[:,223:268].reshape(-1, 1, 15, 3)
        local_body_pos_1 = np.concatenate([local_body_pos_11,local_body_pos_12], axis=1)
        local_body_pos_21 = np.concatenate([padding_xy,motion_obs[:,449:491]], axis=1).reshape(-1, 1, 15, 3)
        local_body_pos_22 = motion_obs[:,671:716].reshape(-1, 1, 15, 3)
        local_body_pos_2 = np.concatenate([local_body_pos_21,local_body_pos_22], axis=1)


        local_pos_1 = torch.from_numpy(local_body_pos_1).unsqueeze(0)
        local_pos_2 = torch.from_numpy(local_body_pos_2).unsqueeze(0)

        intergraph_obs_buf = compute_interaction_graph_obs_batch_pos_local(local_pos_1, local_pos_2)
        motion_ig = intergraph_obs_buf[0].numpy()

        motion_ig_reshape = motion_ig.reshape(motion_obs.shape[0], 2, -1)

        obs_0 = np.concatenate([motion_obs[:,:223], motion_ig_reshape[:,0,:]], axis=-1)
        obs_1 = np.concatenate([motion_obs[:,448:671], motion_ig_reshape[:,1,:]], axis=-1)
        two_obs = np.concatenate([obs_0, obs_1], axis=-1)


        data['obs'] = two_obs

        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))

        return data
