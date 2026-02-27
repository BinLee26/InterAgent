import pickle
import os
import torch
import json
import glob
import sys

from body_model.body_model import BodyModel
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from os.path import join as pjoin
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import scipy.ndimage.filters as filters
from tqdm import tqdm
import json
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]]).float()

# src_path = "body_model\smpl\smpl_m.pkl"
# with open(src_path, 'rb') as f:
#     src_data = pickle.load(f, encoding="latin1")
# # print(src_data.keys())
# # np.savez("body_model/smpl/smpl_m.npz", src_data)
# data1 = np.load(src_path, encoding='latin1', allow_pickle=True)
# data2 = np.load(bm_male, encoding='latin1', allow_pickle=True)
# print(data1['J_regressor'].shape)
# print(data2['J_regressor'].shape)
    
# file_path = r'D:\bjc\TOOLS\smpl_params.pkl'
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)
    
# print(data['person1'].keys())
# data = data['person1']
# print(data['poses'][:,:3])
# data["root_orient"] = data['poses'][:,:3]

# frame_rate = data["mocap_framerate"]
# frame_number = data['trans'].shape[0]
# print("frame_rate.shape", frame_rate)
# print("frame_number:", frame_number)

# down_sample_frames = [int(i*frame_rate/30.) for i in range(0,100000) if int(i*frame_rate/30.)<frame_number]
# length = len(down_sample_frames)
# print("length:", length)

# trans_np = data["trans"][down_sample_frames]
# root_orient_np = data["root_orient"][down_sample_frames]
# pose_body_np = data["poses"][:,3:22*3][down_sample_frames]
# pose_body_np = data["poses"][:,3:22*3][down_sample_frames]
# betas_np = data["betas"][:300]
# gender = data["gender"]
# print(gender)
with open('../data/interhuman_mix/1013_daily_motion/1/smpl_params.pkl', 'rb') as f:
    smpl_params = pickle.load(f)
print(smpl_params['person1'].keys())
gender = smpl_params['person1']['gender']
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

trans_np = smpl_params['person1']["trans"]
trans = trans_np[0].reshape(1, -1)
trans = torch.from_numpy(trans).float()
# print(trans)
bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, model_type="smpl", gender="male")
betas = torch.from_numpy(smpl_params['person1']["betas"]).reshape(1,-1)
print(betas)
print(new_tensors[2])
with torch.no_grad():
    body = bm(pose_body=new_tensors[0], betas=betas, root_orient=new_tensors[3], trans=trans)
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
print(local_translations)