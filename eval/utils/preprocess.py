import numpy as np
from utils.utils import *
import pickle

FPS = 30

from utils import torch_utils



def load_motion(file_path, min_length, swap=False):


    try:
        motion = np.load(file_path).astype(np.float32)
    except:
        print("error: ", file_path)
        return None, None
    motion1 = motion[:, :22 * 3]
    motion2 = motion[:, 62 * 3:62 * 3 + 21 * 6]
    motion = np.concatenate([motion1, motion2], axis=1)

    if motion.shape[0] < min_length:
        return None, None
    if swap:
        motion_swap = swap_left_right(motion, 22)
    else:
        motion_swap = None
    return motion, motion_swap

def load_motion_generated_dp(file_path, min_length,replication, swap=False):
    with open(file_path, "rb") as f:
        motion_data = pickle.load(f)

    motion_state = motion_data['obs'][:,:-1,:,:].astype(np.float16)

    motions = motion_state 

    
    motion1 = motions[replication%20,:,0]
    motion2 = motions[replication%20,:,1]
    print("--!!！！！！！-！@@@@@---+++++++++@@@+++++++@@+++++ load motion generated dp replication:",replication)
    
    return motion1, motion2, None, None


def load_motion_dp(file_path, min_length, swap=False):
    with open(file_path, "rb") as f:
        motion_data = pickle.load(f)

    motion_state = motion_data['obs'][:,:-1,:,:].astype(np.float16)

    motions = motion_state


    motion1 = motions[0,:,0]
    motion2 = motions[0,:,1]

    return motion1, motion2, None, None

def load_motion_mm_dp(file_path, min_length, swap=False):
    with open(file_path, "rb") as f:
        motion_data = pickle.load(f)

    motion_state = motion_data['obs'][:,:-1,:,:].astype(np.float16)

    motions = motion_state 

    
    motion1 = motions[:,:,0]
    motion2 = motions[:,:,1]
    
    return motion1, motion2, None, None
