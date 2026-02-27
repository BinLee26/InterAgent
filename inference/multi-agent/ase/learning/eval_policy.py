import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

def evaluate():
    split_num = "2000_4500"
    success_threshold = 0.95
    model_v = "clean_r5v2norecov_"
    data = joblib.load(f"ase/results/{model_v}split{split_num}.pkl")
    succ_dict = data["motion_success_rates"]
    succ_rate_raw = []
    visualize = False
    save_good = True
    sorted_motions = sorted(succ_dict.items(), key=lambda x: x[1]["success_rate"] if x[1]["success_rate"] is not None else -1)

    good_motions = []
    bad_motions = []
    avg_succ_rate = 0
    num_motion = 0
    for motion_id, stats in sorted_motions:
        succ_rate_raw.append(stats['success_rate'])
        if stats['success_rate'] < success_threshold:
            bad_motions.append(motion_id)
        elif stats['success_rate'] >=success_threshold:
            good_motions.append(motion_id)
        avg_succ_rate += stats['success_rate']
        num_motion += 1

    avg_succ_rate = avg_succ_rate / num_motion
    print('++++bad motions', len(bad_motions), len(bad_motions)/num_motion)
    print('++++avg succ rate:', avg_succ_rate)
    print('++++good motions', len(good_motions), good_motions[:20])

    json_file = f"motion_success_rates_{model_v}_split_{split_num}.json"
    with open(json_file, mode="w") as f:
        json.dump(sorted_motions, f, indent=4)
        
    print(f"Motion success rates saved to {json_file}")

    if save_good:
        good_yaml_file = f"ase/data/motions/intergen_amp/{model_v}_split_{split_num}_good.yaml"
        bad_yaml_file = f"ase/data/motions/intergen_amp/{model_v}_split_{split_num}_bad.yaml"        
        os.makedirs(os.path.dirname(good_yaml_file), exist_ok=True)
        os.makedirs(os.path.dirname(bad_yaml_file), exist_ok=True)
        # save good motions        
        motions_yaml = {"motions": []}
        for motion in good_motions:
            motions_yaml["motions"].append({"file": f"{motion}_person1.npy", "weight": 1.0})
            motions_yaml["motions"].append({"file": f"{motion}_person2.npy", "weight": 1.0})
        
        with open(good_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(motions_yaml, f, default_flow_style=False, sort_keys=False)
        
        good_motion_txt = f"{model_v}_split_{split_num}_good.txt"

        with open(good_motion_txt, 'w', encoding='utf-8') as f:
            for motion in good_motions:
                f.write(f"{motion}.yaml\n")
        #save bad motions
        motions_yaml = {"motions": []}
        for motion in bad_motions:
            motions_yaml["motions"].append({"file": f"{motion}_person1.npy", "weight": 1.0})
            motions_yaml["motions"].append({"file": f"{motion}_person2.npy", "weight": 1.0})
        
        with open(bad_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(motions_yaml, f, default_flow_style=False, sort_keys=False)        
        
        print("Successfully saved good motions!")
        
    if visualize:
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 6))

        sns.histplot(succ_rate_raw, kde=True, bins=20, color='blue', stat='density')
        
        plt.title("distribution", fontsize=16)
        plt.xlabel("success rate", fontsize=12)
        plt.ylabel("density", fontsize=12)

        plt.show()

if __name__ == "__main__":
    evaluate()