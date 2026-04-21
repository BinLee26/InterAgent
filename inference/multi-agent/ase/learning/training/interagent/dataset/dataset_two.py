import os
import pickle
import random
from torch.utils.data import Dataset
import clip
import torch
from training.utils.data import dict_apply
from training.utils.normalizer import LinearNormalizer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-L/14@336px", device)
clip_model.eval() # frozen clip model

class DiffusionPolicyDataset(Dataset):
    def __init__(self, motion_path, obs_scale_path, action_scale_path,horizon=10, obs_length=2):
        self.keys = ['obs','clean_action','motion_fname']
        self.horizon = horizon

        with open(motion_path, "rb") as f:
            self.motion_data = pickle.load(f)

        self.motion_name = list(self.motion_data.keys())
        self.motion_nums = len(self.motion_data)

        self.text_path = "/data/nas_24/libin/Interhuman/annots/"

        self.data_clips = []
        self.all_obs = []
        self.all_action = []

        self.create_motion_clips()

    def get_normalizer(self):
        data = {
            'obs': 0*np.concatenate(self.all_obs, axis=0),
            'action': 0*np.concatenate(self.all_action, axis=0),
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode='limits')
        return normalizer

    def __len__(self) -> int:
        return len(self.data_clips)

    def __getitem__(self, idx):

        data = self.data_clips[idx]

        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))

        return data

    def create_motion_clips(self):


        for i in range(self.motion_nums):
            motion_name_i = self.motion_name[i]
            motion_i = self.motion_data[motion_name_i]

            i_text_path = os.path.join(self.text_path, motion_name_i+'.txt')
            with open(i_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                random_line = lines[0].strip()#random.choice(lines).strip()

            with torch.no_grad():  # 关闭梯度计算
                text_tokens = clip.tokenize(random_line).to(device)
                text_feature = clip_model.encode_text(text_tokens)

            min_start_idx = 0
            frame_num = motion_i['obs'].shape[1]
            max_start_idx = frame_num - self.horizon

            rollout_num = motion_i['obs'].shape[0]
            person_num = motion_i['obs'].shape[2]

            for j in range(rollout_num):
                # for k in range(person_num):
                for idx in range(min_start_idx, max_start_idx + 1):
                    start_idx = max(idx, 0)
                    end_idx = min(idx + self.horizon, frame_num)

                    idx_obs = motion_i['obs'][j, start_idx:end_idx, :, :]
                    idx_action = motion_i['clean_action'][j, start_idx:end_idx, :, :]

                    two_obs = np.concatenate([idx_obs[:, 0, :], idx_obs[:, 1, :]], axis=-1)
                    two_action = np.concatenate([idx_action[:, 0, :], idx_action[:, 1, :]], axis=-1)

                    pad_width = 2*450
                    two_obs = np.pad(two_obs, ((0,0),(0, pad_width)), mode='constant')

                    idx_data = {
                        'obs': two_obs,
                        'action': two_action,
                        'text': random_line,
                        'text_feature': text_feature.squeeze(0).cpu().numpy(),
                    }

                    self.data_clips.append(idx_data)
                    self.all_obs.append(two_obs)
                    self.all_action.append(two_action)
