import os
import glob
import ants
import logging

import h5py

LOG = logging.getLogger(__name__)
# data_path = '/Volumes/AhmedLab/princess/data/' # for mac
data_path = fr'/mnt/z/princess/data/' # for windows


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

def get_experiment(date, id):
    path = fr'{data_path}{date}_{id}'
    return path

def get_file(experiment_path, data_stage, extension):
    file = glob.glob(f'{experiment_path}/{data_stage}/*{extension}')[0]
    return file

def get_dataset(path):
    hf = h5py.File(path, 'r')
    return hf['labels'][...]
    