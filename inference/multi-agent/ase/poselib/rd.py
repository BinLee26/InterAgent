import numpy as np
#import pandas as pd
import pickle
# 加载npy文件
#data = np.load('amp_humanoid_jog.npy',allow_pickle=True)
#data = np.load('amp_humanoid_walk.npy',allow_pickle=True)
#data = np.load('../data/interhuman_amp/tawkwondo_person1.npy',allow_pickle=True)



with open('../data/InterGen/motions/4106.pkl', 'rb') as f:
    smpl_params1 = pickle.load(f)
with open('../data/interhuman_mix/smpl_params.pkl', 'rb') as f:
    smpl_params2 = pickle.load(f)



# 查看数据形状
#print(data.keys())
#print(type(obj))
#print(obj['person1'])
#print(type(data))
# 查看数据类型
#print(data.dtype)

# 显示前几个数据
#print(obj['person1']['poses'].shape)
#print(data)
#print(smpl_params2['person1'].keys())
print(smpl_params1["person1"].keys())
print(smpl_params1["person1"]['trans'])