import ants
import numpy as np
from src import io

# generate mean brain based on first 300 volumes
def generate_fixed(brain, volumes:int):
    mean = np.mean(brain[...,0:volumes], axis=-1)
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
        moving = ants.from_numpy(moving_brain[:, :, :, vol])
        moco_brain[:, :, :, vol] = apply(fixed_brain, moving).numpy()
    return moco_brain

