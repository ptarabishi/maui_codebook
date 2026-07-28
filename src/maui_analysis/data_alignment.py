import h5py
import numpy as np
from maui_codebook import timesync

def open_processed_behavior_data(synced_data_hf):
    with h5py.File(synced_data_hf, 'r') as f:
        print(f.keys())
        walk_times = f['behavior timestamps'][...]
        walk_speeds = f['fly speed'][...]
    return walk_times, walk_speeds

def convert_scope_frametimes_to_volumetimes(timestamp_csv, slices_per_vol):
    frame_timestamps = timesync.extract_scope_timestamps(timestamp_csv).squeeze()
    reshape_to_volumes = np.reshape(frame_timestamps, [-1, slices_per_vol])
    return np.mean(reshape_to_volumes, axis = 1)

def align_scopetimes_to_behaviortimes(timestamp_csv, synced_data_hf, slices_per_vol):
    volume_times = convert_scope_frametimes_to_volumetimes(timestamp_csv, slices_per_vol)
    # open h5 with walk times
    walk_times, _ = open_processed_behavior_data(synced_data_hf)
    # find last index where behavior time and scope times align
    last_index = np.argmax(volume_times > max(walk_times))
    aligned_scopetimes = volume_times[:last_index]
    return aligned_scopetimes

def downsample_behavior_to_scope(behavior_times, scope_times):
        max_divider = behavior_times.shape[0] // scope_times.shape[0]
        behavior_cut = behavior_times[: scope_times.shape[0] * max_divider]  # cut excess behavior data
        behavior_reshaped = np.reshape(behavior_cut, (scope_times.shape[0], -1))
        behavior_downsampled = np.mean(behavior_reshaped, axis=1)
        print(behavior_downsampled.shape[0], scope_times.shape[0])

        return behavior_downsampled

def save_to_aligned_hf(save_path, aligned_scopetimes, synced_data_hf):
    # downsample behavior times and speed
    walk_times, walk_speeds = open_processed_behavior_data(synced_data_hf)
    downsampled_behavior_ts = downsample_behavior_to_scope(aligned_scopetimes, walk_times)
    downsampled_behavior = downsample_behavior_to_scope(aligned_scopetimes, walk_speeds)

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('volume times', data = aligned_scopetimes)
        f.create_dataset('behavior times', data = downsampled_behavior_ts)
        f.create_dataset('behavior speeds', data = downsampled_behavior)