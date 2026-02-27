import os
import shutil
import yaml

new_path = "ase/data/motions/interhuman_amp"
# for root, dirs, filelist in os.walk('../data/interhuman_amp/'):
#     for filename in filelist:
        #str = '_'
        #print(root.split(os.path.sep)[3:]+[filename])
        #new_filename = '_'.join(root.split(os.path.sep)[3:]+[filename])
        #print(root)
        #print(new_filename)
        #os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
        #shutil.move(os.path.join(root, filename), new_path)
        #print(filename)
file_names = sorted(os.listdir(new_path))
#file_names = os.listdir(new_path)
print(file_names)
weight = 1.0

motions = [{"file": file_name, "weight": weight} for file_name in file_names]


data = {"motions": motions}
with open("ase/data/motions/intergen_amp_datasets/dataset_intergen_amp_dual.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False)