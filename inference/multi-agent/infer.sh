#!/bin/bash

motion_list_file="./data/infer.txt"

while IFS= read -r motion_file; do
    echo "=============================="
    echo "Running motion: $motion_file"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=4 python ase/run.py \
        --test \
        --headless \
        --task HumanoidIlDMEval \
        --cfg_env ase/data/cfg/humanoid_il_dm_low.yaml \
        --cfg_train ase/data/cfg/train/rlg/il_humanoid_dm_low_v2.yaml \
        --motion_file "$motion_file" \
        --num_envs 10

    echo "Finished motion: $motion_file"
    echo
done < "$motion_list_file"
