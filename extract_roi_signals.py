import argparse
import glob
import h5py
from maui_codebook import zdF
import numpy as np
import nibabel as nib
import time
from tqdm import trange
import matplotlib.pyplot as plt

def calculate_zscoredF_voxels(raw_array):
    # loop over every slice
    x_by_y = raw_array.shape[0] * raw_array.shape[1]
    window = raw_array.shape[-1]
    final_array = np.empty((raw_array.shape[2], x_by_y, window))

    for iSlice in range(raw_array.shape[2]):
        # collapse x and y
        array = raw_array[...,iSlice, :]
        voxel_array = array.reshape(x_by_y,-1)
        # print(voxel_array.shape)

        final_array[iSlice,:,:] = zdF._zdff(voxel_array, win = window, smooth = True)
    return final_array

def get_supervoxel_mean_2d(brain_slice, cluster_labels, n_clusters):
    # neural_data = brain_slice.shape[]  # make into vector

    signals = []
    cluster_idx = []

    for nn in range(n_clusters):
        idx = np.where(cluster_labels == nn)[0]
        mean_signal = np.nanmean(brain_slice[idx])

        signals.append(mean_signal)
        cluster_idx.append(idx)

    return np.asarray(signals), cluster_idx

def main(experiment_directory):
    nii_dir = f'{experiment_directory}/motion_corrected_*.nii'

    all_nii = glob.glob(nii_dir)
    brain_array = np.empty([512, 512, 11, len(all_nii)], dtype=np.float32)

    start_time = time.time()
    for i in trange(brain_array.shape[-1]):
        single_nii = f'{experiment_directory}/motion_corrected_volume{i}.nii'
        file = nib.load(single_nii)  # shape (x, y, z)
        data = file.get_fdata()
        brain_array[..., i] = data
        del data, file

    print(f"Brain Array Assignment Completed in {time.time() - start_time}s")

    df = calculate_zscoredF_voxels(brain_array)
    del brain_array

    T = len(all_nii)
    n_clusters = 20

    file = glob.glob(f"{experiment_directory}/*signals_260713.h5")[0]
    with h5py.File(file, "r") as f:
        cluster_labels = f["labels"][...]s

    signal = np.empty(shape=(df.shape[0], n_clusters, T))
    for slice in range(df.shape[0]):
        mean_signal = np.empty(shape=(df.shape[2], n_clusters))  # t, z
        for vol in range(df.shape[2]):
            mean_supervox, _ = get_supervoxel_mean_2d(df[slice, :, vol], cluster_labels[slice], n_clusters)
            mean_signal[vol] = mean_supervox
        signal[slice] = mean_signal.T

    with h5py.File(file, "a") as f:
        f.create_dataset("signals", data=signal)


if __name__ == "__main__":
    experiment_list = glob.glob('/Volumes/AhmedLab/princess/data/pIP10/processed/*')[0:5]
    for iExp in experiment_list:
        if glob.glob(f'{iExp}/*signals_260713.h5'):
            main(iExp)
        else:
            print(f"Skipping {iExp}")

