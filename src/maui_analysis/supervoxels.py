import numpy as np
from maui_codebook import zscore
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from tqdm import trange
import nibabel as nib
import time

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

        final_array[iSlice,:,:] = zscore._zdff(voxel_array, win = window, smooth = True)
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

def extract_ROIs_from_zscored(voxel_array, n_clusters, dimensions):
    labels = []
    voxel_array = np.nan_to_num(voxel_array)
    for iSlice in trange(voxel_array.shape[0]): # for each slice in Z
    # generate n_clusters within a single slice
    #     print(iSlice)

        cluster_model = create_2d_clusters(voxel_array[iSlice,:, 0:-1:5], dimensions=dimensions, n_clusters=n_clusters, mempath='tmp/cluster_mem')
        labels.append(cluster_model.labels_)
    return labels

def create_2d_clusters(brain_slice, dimensions, n_clusters: int, mempath: str):
    xdim = dimensions[0]
    ydim = dimensions[1]
    tdim = brain_slice.shape[-1]

    connectivity = grid_to_graph(xdim, ydim)

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, memory=mempath, linkage='ward', connectivity=connectivity)

    super_to_cluster = brain_slice.reshape(-1, tdim)
    cluster_model.fit(super_to_cluster)

    return cluster_model