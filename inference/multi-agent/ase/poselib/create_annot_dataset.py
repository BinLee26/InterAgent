import os
import yaml
import re
from collections import defaultdict

motion_dir = "ase/data/motions/intergen_amp"
annot_dir = "ase/data/InterGen/annots/"


motions = defaultdict(list)


# 遍历motion文件夹中的所有文件
for filename in os.listdir(motion_dir):
    # 检查文件是否为numpy文件
    if filename.endswith(".npy"):
        # 获取文件名中的编号和person编号
        number = filename.split("_")[0]
        person = int(filename.split("_")[1].split(".")[0].replace("person", ""))
        
        # 构建对应的annotation文件名
        annot_filename = number + ".txt"
        annot_path = os.path.join(annot_dir, annot_filename)
        
        # 检查annotation文件是否存在,并且motion文件名不包含(数字)
        if os.path.exists(annot_path) and not re.search(r'\(\d+\)', filename):
            # 添加motion和annotation信息到字典中
            motions[number].append({
                "file": filename,
                "annot": annot_filename,
                "weight": 1.0,
                "person": person
            })

# 对每个编号的motion信息按person编号排序
sorted_motions = []
for motions in sorted(motions.values(), key=lambda x: int(x[0]["file"].split("_")[0])):
    sorted_motions.extend(sorted(motions, key=lambda x: x["person"]))

# 将排序后的结果写入yaml文件
with open("motions.yaml", "w") as outfile:
    yaml.dump({"motions": sorted_motions}, outfile, default_flow_style=False)