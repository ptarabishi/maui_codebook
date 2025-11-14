import os
import glob
import ants
import logging

import h5py

LOG = logging.getLogger(__name__)

def load_nii(fpath: str):
    """loads volume from filepath and returns N-Dim numpy array"""
    LOG.info(f"loading: {fpath}")
    imgarray = ants.image_read(fpath)
    volume = imgarray.numpy()
    LOG.debug(f"volume shape: {volume.shape}")
    return volume

def save_nii(fpath: str, volume):
    "save volume to fpath"
    ants.image_write(ants.from_numpy(volume), fpath)
    LOG.info(f'saved volume to: {fpath}')

def load_pickle(fpath):
    pass

def save_pickle(fpath):
    pass

## new functions for loading in h5 files
def load_clusters(experiment_path):
    file = glob.glob(f'{experiment_path}/*signals.h5')[0]
    with h5py.File(file, 'r') as f:
        cluster_labels = f['labels'][...]
        df = f['df/f'][...]
    return cluster_labels, df

def load_acquisition_params(experiment_path):
    file = glob.glob(f'{experiment_path}/*acquisition_parameters.h5')[0]
    with h5py.File(file, 'r') as f:
        scope_fr = f['scope_fr'][...]
        camera_fr = f['camera_fr'][...]
        brain_dim = f['brain_dimensions'][...]
        brain_dim= brain_dim.reshape(-1)
    return scope_fr, camera_fr, brain_dim

def load_fictrac_data(experiment_path):
    file = glob.glob(f'{experiment_path}/*fictrac.h5')[0]
    with h5py.File(file, 'r') as f:
        smoothed_speed = f['smoothed_speed'][...]
        xy_pos = f['2d_pos'][...]
        delta_rot = f['delta_rot'][...]
    return smoothed_speed, xy_pos, delta_rot