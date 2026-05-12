# written by Sama Ahmed

import ants
import numpy as np
import time
from src import io
import argparse

# generate mean brain based on first 300 volumes
def generate_fixed(brain_arr, volumes:int):
    mean = np.mean(brain_arr[...,0:volumes], axis=-1)
    fixed = ants.from_numpy(mean)
    return mean, fixed

def apply(fixed, moving):
    moco_moving = ants.registration(fixed, moving, type_of_transform='SyN')
    return moco_moving["warpedmovout"]

def motion_correction(moving_brain, fixed_brain):
    # fixed_brain = generate_fixed(fixed_brain)
    n_vols = moving_brain.shape[-1]
    moco_brain = np.zeros_like(moving_brain)

    for vol in range(n_vols):
        start = time.time()
        moving = ants.from_numpy(moving_brain[:, :, :, vol])
        moco_brain[:, :, :, vol] = apply(fixed_brain, moving).numpy()
        end = time.time()
        # print(f'10 volumes took {end-start} seconds')
        if vol % 10 == 0:
            print(f'remaining volumes: {n_vols - vol}')
    return moco_brain

parser = argparse.ArgumentParser()
parser.add_argument(
    '--rawDirectory', type=str,required=True, help='Path to the moving brain image'
)
parser.add_argument(
    '--saveDirectory', type=str,required=True, help='Path to the motion-corrected brain image'
)

if __name__ == '__main__':
    motion_correction(ar)
