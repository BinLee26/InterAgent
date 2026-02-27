import yaml
import random
import os

def load_yaml(file_path):
    """加载 YAML 文件"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(data, file_path):
    """保存数据到 YAML 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def sample_pairs(motions, n):
    """从 motions 中随机采样 n 个 pair"""
    # 按文件名的前缀分组（如 1000, 1001, 1002）
    grouped = {}
    for motion in motions:
        prefix = motion['file'].split('_')[0]
        if prefix not in grouped:
            grouped[prefix] = []
        grouped[prefix].append(motion)
    
    # 只保留有两个 person 的分组
    valid_pairs = [group for group in grouped.values() if len(group) == 2]
    
    # 随机采样 n 个 pair
    sampled_pairs = random.sample(valid_pairs, min(n, len(valid_pairs)))
    
    return sampled_pairs

# 加载原始 YAML 文件
yaml_path = 'ase/data/motions/intergen_amp/dataset_intergen_amp_annot.yaml'  # 替换为你的 YAML 文件路径
data = load_yaml(yaml_path)

# 从 motions 中采样 3 个 pair
n = 1000
sampled_pairs = sample_pairs(data['motions'], n)

# 将采样结果转换为新 YAML 格式
sampled_data = {'motions': [item for pair in sampled_pairs for item in pair]}

# 保存到新的 YAML 文件
output_path = 'ase/data/motions/intergen_amp_datasets/dataset_intergen_amp_annot_sample1.yaml'
save_yaml(sampled_data, output_path)

print(f"采样结果已保存到 {output_path}")