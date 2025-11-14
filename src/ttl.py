# functions for ttl synch
# written by Carter Archuleta
# 


import pandas as pd
import numpy as np

def read_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    frames = [x.find("Frame") for x in root]
    sequences = [x.find("Sequence") for x in root]
    # my_text = [item.text for item in tree.iter()]
    print(f'Frames: {len(frames)}')
    print(f'Sequences: {len(sequences)}')

def read_csv(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data.columns = data.columns.str.strip()
    # Determine which column is the scope by comparing the average voltage
    column_means = np.floor(data.mean())
    for idx, column_mean in enumerate(column_means):
        if column_mean == 0:
            scope_idx = idx
        elif 0 < column_mean < 5:
            side_camera_idx = idx
    if len(data.columns) == 3:
        data = data.rename(columns={f'{data.columns[scope_idx]}': 'Scope', f'{data.columns[side_camera_idx]}': 'Side Camera'})
    return data

def get_TTL_frame_period(data):
    '''Get the frame period of the TTL data in ms'''

    return get_frame_period(data['Time(ms)'])

def get_frame_period(timestamps):
    '''Get the frame period ms'''

    frame_period = np.mean(timestamps.diff().dropna())
    return float(frame_period)

def extract_2p_relative_timestamps(data, threshold=5):
    scope_on_signals = data[data['Scope'] > threshold]
    first_signal = scope_on_signals['Time(ms)'].iloc[0]

    timestamps = scope_on_signals[scope_on_signals['Time(ms)'].diff() > 2 * get_TTL_frame_period(data)]['Time(ms)']
    timestamps = np.concatenate([[first_signal], timestamps.to_numpy()])

    return timestamps

def extract_camera_relative_timestamps(data, threshold=3):
    camera_on_signals = data[data['Side Camera'] > threshold]
    first_signal = camera_on_signals['Time(ms)'].iloc[0]

    # Get the first in clusters (voltage is ~3.65 V for whole camera exposure)
    timestamps = camera_on_signals[camera_on_signals['Time(ms)'].diff() > 2 * get_TTL_frame_period(data)]['Time(ms)']
    timestamps = np.concatenate([[first_signal], timestamps.to_numpy()])

    return timestamps

def get_frame_rate(timestamps):
    '''Get the frame rate Hz'''

    frame_rate = 1 / np.mean(timestamps.diff().dropna())
    return float(frame_rate * 1000) # Convert to Hz from mHz

def convert_maui_times(framerate, signal_array):
    time_frames = [x for x in range(1, signal_array.shape[-1])]
    # change framerate to volume rate in Hz
    volume_rate = framerate / signal_array.shape[0]
    time_secs = [0] + [x/volume_rate for x in time_frames]

    print(f'volume rate: {volume_rate:.2f} Hz, total experiment length {time_secs[-1]:.2f} seconds')
    return volume_rate, time_secs