import glob
import h5py
import numpy as np
import nibabel as nib
import time
from tqdm import trange

from maui_analysis.supervoxels import calculate_zscoredF_voxels, get_supervoxel_mean_2d

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
        cluster_labels = f["labels"][...]

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

