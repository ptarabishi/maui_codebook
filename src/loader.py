import os
import glob
import logging
import numpy as np
import h5py
import pandas as pd
import nibabel as nib

LOG = logging.getLogger(__name__)

## new functions for loading in h5 files
def get_base_path(experiment_id, data_stage, base_dir = '/Volumes/AhmedLab/princess/data/'):
    # only one can be true
    experiment_id = str(experiment_id)
    if data_stage == 'raw':
        experiment_path = glob.glob(os.path.join(base_dir, 'raw', f'*{experiment_id}'))[0]
    if data_stage == 'processed':
        experiment_path = glob.glob(os.path.join(base_dir, 'processed', f'*{experiment_id}'))[0]

    return experiment_path

def load_clusters(experiment_path):
    file = glob.glob(f'{experiment_path}/*signals.h5')[0]
    with h5py.File(file, 'r') as f:
        cluster_labels = f['labels'][...]
        df = f['df/f'][...]
        timestamps = f['timestamps'][...]
    return cluster_labels, df, timestamps

def load_acquisition_params(experiment_path):
    file = glob.glob(f'{experiment_path}/*acquisition_parameters.h5')[0]
    with h5py.File(file, 'r') as f:
        scope_fr = f['scope_fr'][...]
        camera_fr = f['camera_fr'][...]
        brain_dim = f['brain_dimensions'][...]
        brain_dim= brain_dim.reshape(-1)
    return scope_fr, camera_fr, brain_dim

def load_fictrac_speed(dat_filepath, camera_framerate, smoothing_size_s = 0.5):
    fictrac_data = pd.DataFrame(pd.read_csv(dat_filepath, header=None))
    inst_speed = np.rad2deg(fictrac_data[18])
    return inst_speed

    # file = glob.glob(f'{experiment_path}/*fictrac.h5')[0]
    # with h5py.File(file, 'r') as f:
    #     smoothed_speed = f['smoothed_speed'][...]
    #     xy_pos = f['2d_pos'][...]
    #     delta_rot = f['delta_rot'][...]
    # return smoothed_speed, xy_pos, delta_rot