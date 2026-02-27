import pickle

with open("ase/tracking_dataset_recov/tracking_data_1.pkl", "rb") as f:
    motion_data = pickle.load(f)

print(motion_data.keys())
print(motion_data["obs"].shape)
print(motion_data["clean_action"].shape)

print(motion_data["obs"][0:8,1,0,:5])

#print(motion_data["clean_action"][0:3,0,0,:10])
# print(motion_data["env_action"][0:3,0,0,:10])