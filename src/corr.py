import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def downsize_dataset(fast_ds, slow_ds, slice_num=0):
    slow_signal = pd.DataFrame(slow_ds[slice_num, :, :].T)
    slow = slow_signal.to_numpy()
    fast = fast_ds

    max_divider = fast.shape[0] // slow.shape[0]
    fast_cut = fast[:slow.shape[0] * max_divider]  # cut excess behavior data
    fast_reshaped = np.reshape(fast_cut, (slow.shape[0], -1))
    fast_downsample = np.mean(fast_reshaped, axis=1)
    print(fast_downsample.shape[0], slow.shape[0])

    return fast_downsample


def pearson_analysis(cluster_labels, resampled_behavior):
    pearson_arr = []
    for slice_idx, clusters, in enumerate(cluster_labels):
        for cluster_idx, signal in enumerate(clusters):
            corr = pearsonr(signal, resampled_behavior).statistic
            pearson_arr.append(corr)
    # pearson_arr_reshape = np.reshape(pearson_arr, (cluster_labels.shape[0], -1))
    return np.array(pearson_arr)

def sort_descending(array):
    sort_ascending = np.argsort(array)
    flipped = np.flip(sort_ascending)
    return flipped