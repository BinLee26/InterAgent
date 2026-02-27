from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

def main():
    source_tpose = SkeletonState.from_file('data/amp_humanoid_tpose.npy')
    #source_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
    #source_tpose = SkeletonMotion.from_file('../data/interhuman_amp/test_person2.npy')
    #print(source_tpose.local_translation)
    plot_skeleton_state(source_tpose)
    plot_skeleton_motion_interactive(source_tpose)

if __name__ == '__main__':
    main()