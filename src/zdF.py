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

    # find average signal in first `win` volumes
    Fbase = np.mean(F[:, :win], axis=-1)
    dff = (F - Fbase[:, None]) / Fbase[:, None]

    if smooth:
        dff = gaussian_filter1d(dff, sigma=1)

    return _zscore(dff)

def calculate_zscoredF(brain, labels_arr, n_clusters=200):
    ROIs = np.empty((brain.shape[2], n_clusters, brain.shape[-1]))
    # set a baseline F window
    F_WINDOW = ROIs.shape[2]

    for iSlice in range(ROIs.shape[0]):
        mean_signal = np.empty(shape=(ROIs.shape[2], n_clusters))

        for vol in range(ROIs.shape[2]):
            mean_supervox, _ = roi.get_supervoxel_mean_2D(brain[:, :, iSlice, vol], labels_arr[iSlice], n_clusters)
            mean_signal[vol] = mean_supervox

        # find zscored(df/f) and smooth over time
        ROIs[iSlice, :, :] = _zdff(mean_signal.T, win=F_WINDOW, smooth=True)
    return ROIs