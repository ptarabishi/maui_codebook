import gc
import h5py
from maui_codebook import loader, zdF, roi, timesync
import glob
import time
import numpy as np
from tqdm import tqdm, trange
import nibabel as nib
from maui_codebook import visualizations
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from loguru import logger

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

def create_2d_clusters(brain_slice, dimensions, n_clusters: int, mempath: str):
    xdim = dimensions[0]
    ydim = dimensions[1]
    tdim = brain_slice.shape[-1]

    connectivity = grid_to_graph(xdim, ydim)

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, memory=mempath, linkage='ward', connectivity=connectivity)

    super_to_cluster = brain_slice.reshape(-1, tdim)
    cluster_model.fit(super_to_cluster)

    return cluster_model

def extract_ROIs_from_zscored(voxel_array, n_clusters, dimensions):
    labels = []
    voxel_array = np.nan_to_num(voxel_array)
    for iSlice in trange(voxel_array.shape[0]): # for each slice in Z
    # generate n_clusters within a single slice
    #     print(iSlice)

        cluster_model = create_2d_clusters(voxel_array[iSlice,:, 0:-1:5], dimensions=dimensions, n_clusters=n_clusters, mempath='tmp/cluster_mem')
        labels.append(cluster_model.labels_)
    return labels

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

def main(experiment_path, n_clusters):

    all_nii = glob.glob(experiment_path + '/motion_corrected_*.nii')
    print(f'for experiment {experiment_path}, there are {len(all_nii)} nii files')

    # to z-score first then cluster
    #T = len(all_nii)
    T = 500
    if len(all_nii) < T:
        T = len(all_nii)
    brain_array = np.empty([512, 512, 11, T])
    slice_dimensions = [brain_array.shape[0], brain_array.shape[1]]

    logger.info("Starting Brain Array Assignment")

    start_time = time.time()
    for i in trange(brain_array.shape[-1]):
        t0 = time.time()
        single_nii = f'{experiment_path}/motion_corrected_volume{i}.nii'
        file = nib.load(single_nii)
        data = file.get_fdata()
        brain_array[..., i] = data
        # logger.debug(f"Assignment for slice: {i} took: {time.time() - t0}s")
        del data, file
    logger.info(f"Brain Array Assignment Completed in {time.time() - start_time}s")
    brain_array = brain_array.astype(np.float32)

    logger.info("Starting zscoredF calculation")
    t0 = time.time()
    df = calculate_zscoredF_voxels(brain_array)
    # TODO: can delete the brain_array after - not used anymore?
    del brain_array
    gc.collect()
    logger.info(f'{time.time() - t0}s to calculate zscoredF voxels')

    t0 = time.time()
    logger.info(f"Extracting ROIs from zscoredF voxels for {n_clusters} clusters and across: {slice_dimensions}")
    cluster_labels = extract_ROIs_from_zscored(df, n_clusters=n_clusters, dimensions=slice_dimensions)
    logger.info(f"Finished extracting ROIs")
    cluster_array = np.asarray(cluster_labels)
    logger.info(f"converted cluster labels to array")
    # print(f'{time.time() - t0}s to extract ROIs')
    logger.info(f'{time.time() - t0}s to extract ROIs')

    hf_name = f'{experiment_path}/{n_clusters}_signals_260713.h5'
    hf = h5py.File(hf_name, 'w')
    logger.info(f"Saving to h5py file")
    hf.create_dataset('labels', data=cluster_array)
    hf.create_dataset('df/f', data=df)
    logger.info(f"Saved to h5py file")
    hf.close()
    print(f'saved as {hf_name}')


if __name__ == '__main__':
    all_exps = glob.glob('/Volumes/AhmedLab/princess/data/pIP10/processed/*win01*') + glob.glob(
        '/Volumes/AhmedLab/princess/data/pIP10/processed/*win02*')
    for exp in all_exps:
        if glob.glob(f'{exp}/*signals_260713.h5'):
            continue
        main(exp, n_clusters = 20)