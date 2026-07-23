import os

import pandas as pd
from scipy.signal import savgol_filter
from maui_codebook.timesync import detect_on_times
import h5py

def pull_fictrac_speed(dat_file, camera_framerate = 130, smoothing_window_s = 0.5):
    speed_df = pd.DataFrame(pd.read_csv(dat_file, header=None)) # radians/frame

    # calculate speed in mm/s, where mm = arc length
    # arc length = radius * radians
    ball_radius = 4.5 # mm
    inst_speed = speed_df[18] * ball_radius * camera_framerate # mm/s

    window_size = int(smoothing_window_s * camera_framerate)
    speed_smoothed = savgol_filter(inst_speed, window_size, polyorder=3)
    return speed_smoothed


def pull_fictrac_timestamps(voltage_file, fly_speed):
    data = pd.read_csv(voltage_file)
    camera_timestamps = detect_on_times(data["Time(ms)"], data[" Input 2"], threshold=3)
    if len(fly_speed) < len(camera_timestamps):
        camera_timestamps = camera_timestamps[:len(fly_speed)]

    return camera_timestamps.squeeze()

def save_speed_data(dat_file, voltage_file, save_path):
    speed = pull_fictrac_speed(dat_file, camera_framerate=130)
    timestamps = pull_fictrac_timestamps(voltage_file, fly_speed=speed)

    hf_path = f'{save_path}/synced_data.h5'
    if not os.path.exists(hf_path):
        hf = h5py.File(hf_path, 'w')
    else:
        hf = h5py.File(hf_path, 'r+')
    hf.create_dataset(name='fly speed', data=speed)
    hf.create_dataset(name='behavior timestamps', data=timestamps)
    hf.close()