import numpy as np
import torch
import random
import os

from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from utils.utils import *
from utils.plot_script import *
from utils.preprocess import *


def read_text_file_safely(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'latin1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            continue
    print(f"Warning: Could not read file {file_path} with any encoding")
    return []


class InterDPDataset(data.Dataset):
    def __init__(self, opt,replication):
        self.opt = opt
        print(f"Initializing InterDPDataset with mode: {opt.MODE}")
        print(f"Data root: {opt.DATA_ROOT}")
        
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 500
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE

        # Check if data directory exists
        if not os.path.exists(opt.DATA_ROOT):
            raise ValueError(f"Data directory {opt.DATA_ROOT} does not exist!")

        ignore_list = []
        ignore_list_path = os.path.join(opt.DATA_ROOT, "ignore_list.txt")
        if os.path.exists(ignore_list_path):
            try:
                ignore_list = open(ignore_list_path, "r").readlines()
                print(f"Loaded {len(ignore_list)} entries from ignore list")
            except Exception as e:
                print(f"Error loading ignore list: {e}")
        else:
            print(f"Warning: ignore_list.txt not found at {ignore_list_path}")

        data_list = []
        data_list_path = os.path.join(opt.DATA_ROOT, f"{opt.MODE}.txt")
        if os.path.exists(data_list_path):
            try:
                data_list = open(data_list_path, "r").readlines()
                print(f"Loaded {len(data_list)} entries from {opt.MODE}.txt")
            except Exception as e:
                print(f"Error loading {opt.MODE}.txt: {e}")
        else:
            raise ValueError(f"Required file {data_list_path} not found!")

        random.shuffle(data_list)
        # data_list = data_list[:70]

        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT)):
            print(f"Scanning directory: {root}")
            print(f"Found {len(files)} files")
            for file in tqdm(files):
                if file.endswith(".pkl"):
                    motion_name = file.split(".")[0]
                    if file.split(".")[0]+"\n" in ignore_list:
                        print(f"Ignoring file: {file}")
                        continue
                    if file.split(".")[0]+"\n" not in data_list:
                        print(f"Skipping file not in data_list: {file}")
                        continue
                    file_path_person = pjoin(root, file)
                    #text_path = file_path_person.replace("motions", "annots").replace("pkl", "txt")
                    text_path = pjoin(opt.ANNOT_ROOT, file.replace("pkl", "txt"))
                    if not os.path.exists(text_path):
                        print(f"Warning: Text file not found: {text_path}")
                        continue
                    with open(file_path_person, "rb") as f:
                        motion_data = pickle.load(f)
                    if 'text' in motion_data.keys():
                        texts = [motion_data['text'].strip()]
                        print(f"Found text: {texts}")
                    else:
                        texts = read_text_file_safely(text_path)
                    if not texts:
                        print(f"Warning: No texts found in {text_path}")
                        continue
                        
                    texts_swap = [item.replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]
                    if opt.MODE == 'mm_generated':
                        motion1, motion2, motion1_swap, motion2_swap = load_motion_mm_dp(file_path_person, self.min_length, swap=True)
                    elif opt.MODE == 'generated':
                        motion1, motion2, motion1_swap, motion2_swap = load_motion_generated_dp(file_path_person, self.min_length, replication,swap=True)
                    else:
                        motion1, motion2, motion1_swap, motion2_swap = load_motion_dp(file_path_person, self.min_length,swap=True)
                    if motion1 is None:
                        print(f"Warning: Failed to load motion from {file_path_person}")
                        continue

                    if self.cache:
                        self.motion_dict[index] = [motion1, motion2]
                        self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                    else:
                        self.motion_dict[index] = [file_path_person1, file_path_person2]
                        self.motion_dict[index + 1] = [file_path_person1, file_path_person2]

                    self.data_list.append({
                        "name": motion_name,
                        "motion_id": index,
                        "swap":False,
                        "texts":texts
                    })

                    index += 2
        
        print(f"Successfully loaded {len(self.data_list)} samples")
        print(f"Motion dictionary contains {len(self.motion_dict)} entries")
        if len(self.data_list) == 0:
            raise ValueError("No data was loaded! Please check your data directory and file paths.")

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
        if self.opt.MODE == 'mm_generated':
            data = self.data_list[item]
            name = data["name"]
            motion_id = data["motion_id"]

            #TODO: need to specify the text that is chosen
            text = data["texts"][0].strip() #random.choice(data["texts"]).strip()
            print('texts: ', data["texts"], 'chosen text: ', text)
            if self.cache:
                mm_motion1, mm_motion2 = self.motion_dict[motion_id]
                print('mm_motion1.shape: ', mm_motion1.shape)
            else:
                raise NotImplementedError("MM generated dataset does not support caching")

            length = mm_motion1.shape[1]
            if length > self.max_length:
                idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
                gt_length = self.max_gt_length
                motion1 = mm_motion1[:,idx:idx + gt_length]
                motion2 = mm_motion2[:,idx:idx + gt_length]

            else:
                idx = 0
                gt_length = min(length - idx, self.max_gt_length )
                motion1 = mm_motion1[:,idx:idx + gt_length]
                motion2 = mm_motion2[:,idx:idx + gt_length]

            gt_motion1 = motion1
            gt_motion2 = motion2

            gt_length = len(gt_motion1[0])
            if gt_length < self.max_gt_length:
                padding_len = self.max_gt_length - gt_length
                B, D = gt_motion1.shape[0], gt_motion1.shape[2]
                padding_zeros = np.zeros((B, padding_len, D))
                gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=1)
                gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=1)

            print('mm_gt_motion1.shape: ', gt_motion1.shape)
            assert len(gt_motion1[0]) == self.max_gt_length
            assert len(gt_motion2[1]) == self.max_gt_length

            motion_lens = np.array([gt_motion1.shape[1]]*gt_motion1.shape[0])

            return "mm_generated", text, gt_motion1, gt_motion2, motion_lens
        else:
            idx = item % self.real_len()
            data = self.data_list[idx]

            name = data["name"]
            motion_id = data["motion_id"]
            swap = data["swap"]
            text = data["texts"][0].strip() #random.choice(data["texts"]).strip()

            if self.cache:
                full_motion1, full_motion2 = self.motion_dict[motion_id]
            else:
                file_path1, file_path2 = self.motion_dict[motion_id]
                motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)
                motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)
                if swap:
                    full_motion1 = motion1_swap
                    full_motion2 = motion2_swap
                else:
                    full_motion1 = motion1
                    full_motion2 = motion2

            #print('full_motion1.shape: ', full_motion1.shape)
            length = full_motion1.shape[0]
            if length > self.max_length:
                idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
                gt_length = self.max_gt_length
                motion1 = full_motion1[idx:idx + gt_length]
                motion2 = full_motion2[idx:idx + gt_length]

            else:
                idx = 0
                gt_length = min(length - idx, self.max_gt_length )
                motion1 = full_motion1[idx:idx + gt_length]
                motion2 = full_motion2[idx:idx + gt_length]


            gt_motion1 = motion1
            gt_motion2 = motion2

            gt_length = len(gt_motion1)
            if gt_length < self.max_gt_length:
                padding_len = self.max_gt_length - gt_length
                D = gt_motion1.shape[1]
                padding_zeros = np.zeros((padding_len, D))
                gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
                gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


            assert len(gt_motion1) == self.max_gt_length
            assert len(gt_motion2) == self.max_gt_length

            if np.random.rand() > 0.5:
                gt_motion1, gt_motion2 = gt_motion2, gt_motion1

        #print('gt_motion1.shape: ', gt_motion1.shape)
        return name, text, gt_motion1, gt_motion2, gt_length
