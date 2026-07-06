import numpy as np
from scipy.ndimage import gaussian_filter1d
from src import roi

def _zscore(x):
    """x: 2D array (n, t)"""

    x_mean = np.mean(x, axis=-1)
    x_std = np.std(x, axis=-1)
    return (x - x_mean[:, None]) / x_std[:, None]


def _zdff(F, win=200, smooth=False):
    """calculate zscored(df/f) based on F baseline activity"""

    # F [pixel]
    # find average signal in first `win` volumes
    Fbase = np.mean(F[:, :win], axis=-1) 
    dff = (F - Fbase[:, None]) / Fbase[:, None]

    if smooth:
        dff = gaussian_filter1d(dff, sigma=1)

    return _zscore(dff)

def calculate_zscoredF(brain, labels_arr, n_clusters=200):
    ROIs = np.empty((brain.shape[2], n_clusters, brain.shape[-1])) # (z, clusters, t)
    # set a baseline F window
    F_WINDOW = ROIs.shape[2] # over the entire experiment (t)

    # loop over every slice
    for iSlice in range(ROIs.shape[0]):
        # initialize array to hold signal over time, by cluster
        mean_signal = np.empty(shape=(ROIs.shape[2], n_clusters))

        # at each time point
        for vol in range(ROIs.shape[2]):
            # get mean 2d value given 2d brain array, slice labels, and number of clusters
            mean_supervox, _ = roi.get_supervoxel_mean_2D(brain[:, :, iSlice, vol], labels_arr[iSlice], n_clusters)
            # assign signal at a given slice as the mean calculation
            mean_signal[vol] = mean_supervox

        # find zscored(df/f) and smooth over time
        ROIs[iSlice, :, :] = _zdff(mean_signal.T, win=F_WINDOW, smooth=True)
    return ROIs

def calculate_zscoredF_voxels(raw_array):
    # loop over every slice
    x_by_y = raw_array.shape[0] * raw_array.shape[1]
    window = raw_array.shape[2]

    for iSlice in range(raw_array.shape[2]):
        # collapse x and y
        array = raw_array[...,iSlice, :]
        voxel_array = raw_array.reshape(x_by_y,-1)

    df = _zdff(voxel_array, win = window, smooth = True)
    return df