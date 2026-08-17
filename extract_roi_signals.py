import glob
import h5py
import numpy as np
import nibabel as nib
import time
from tqdm import trange

from maui_analysis.supervoxels import calculate_zscoredF_voxels, get_supervoxel_mean_2d

def check_for_dataset(file, dataset:str):
    with h5py.File(file, "r") as f:
        dataset_name = dataset
        if dataset_name in f:
            return True
        else:
            return False


def calculate_supervoxel_signal(supervoxel_labels, df, T, n_clusters):
    n_slices = df.shape[0]
    signal = np.empty(shape=(df.shape[0], n_clusters, T))
    for slice in range(df.shape[0]):
        mean_signal = np.empty(shape=(df.shape[2], n_clusters))  # t, z
        for vol in range(df.shape[2]):
            mean_supervox, _ = get_supervoxel_mean_2d(df[slice, :, vol], supervoxel_labels[slice], n_clusters)
            mean_signal[vol] = mean_supervox
        signal[slice] = mean_signal.T
        return signal


def main(experiment_directory):
    hf_file = glob.glob(f"{experiment_directory}/*supervoxels.h5")[0]
    signal_status = check_for_dataset(hf_file, dataset='signal') or check_for_dataset(hf_file, dataset='signals')
    sh_signal_status = check_for_dataset(hf_file, dataset='shifted_signals')

    nii_dir = f'{experiment_directory}/motion_corrected_*.nii'
    print(f'loading{hf_file}')
    all_nii = glob.glob(nii_dir)
    T = len(all_nii)
    # T = 700
    brain_array = np.empty([512, 512, 11, T], dtype=np.float32)

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

    n_clusters = 20

    with h5py.File(hf_file, "r") as f:
        cluster_labels = f["labels"][...]

    # shift cluster labels
    rng = np.random.default_rng()
    xs, ys = rng.integers(low=-102, high=102, size=[2])
    shifted_labels = np.roll(cluster_labels, shift=[xs, ys], axis=1)

    # check for signal
    if signal_status == False:
        print('calculating supervoxels signals')
        signal = calculate_supervoxel_signal(cluster_labels, df, T, n_clusters)
        with h5py.File(file, "r+") as f:
            f.create_dataset("signals", data=signal)

    # check for shifted signal

    if sh_signal_status == False:
        print('calculating shifted signals')
        sh_signal = calculate_supervoxel_signal(shifted_labels, df, T, n_clusters)
        with h5py.File(file, "r+") as f:
            f.create_dataset("shifted_signals", data=sh_signal)

    print(f'all datasets in h5 file: {f.keys()}')



if __name__ == "__main__":
    experiment_list = glob.glob('/Volumes/AhmedLab/princess/data/pIP10/processed/*')[:-6]
    # experiment_list = all_exps =glob.glob('/Volumes/AhmedLab/princess/data/pIP10/processed/pIP10_TSeries_20260421_GC8m_fly03_win01_trial-001*')
    for iExp in experiment_list:
        if glob.glob(f'{iExp}/*supervoxels.h5'):
            print(f"calculating supervoxels in exp: {iExp}")
            file = glob.glob(f"{iExp}/*supervoxels.h5")[0]
            signal_status = check_for_dataset(file, dataset='signal') or check_for_dataset(file, dataset='signals')
            sh_signal_status = check_for_dataset(file, dataset='shifted_signals')
            status = signal_status + sh_signal_status
            if signal_status or sh_signal_status == 2:
                print(f"Skipping {iExp}")
                continue
                # main(iExp)
            else:
                print('missing dataset')
                main(iExp)
        else:
            print(f"Skipping {iExp}")

