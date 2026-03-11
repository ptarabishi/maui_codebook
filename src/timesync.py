import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def extract_scope_timestamps(experiment_path, plot = False):
    raw_scope_ttl = load_scope_voltage_data(experiment_path)
    raw_scope_xml = load_scope_xml_data(experiment_path)

    true_scope_ttl = pull_nearest_timestamps(raw_scope_xml, raw_scope_ttl)

    if plot == True:
        plot_timestamps(raw_scope_xml, true_scope_ttl)

    return pd.DataFrame(true_scope_ttl)

# load in files containing timestamp data
def load_scope_voltage_data(experiment_path):
    """
       Loads voltage data and pulls out timestamps into arrays
   """
    file = glob.glob(os.path.join(experiment_path, '*.csv'))[0]
    data = pd.read_csv(file)

    # determine signal source by comparing average voltage from Carter's script
    # data.columns = data.columns.str.strip()
    # column_means = np.floor(data.mean())

    maui_timestamps = detect_on_times(data['Time(ms)'], data[' Input 1'], threshold=4)

    return maui_timestamps

def load_scope_xml_data(experiment_path):
    file = glob.glob(os.path.join(experiment_path,'*.xml'))[0]

    tree = ET.parse(file)
    root = tree.getroot()
    xml_frames = []
    for element in root.iter("Frame"):
        time = float(element.attrib['relativeTime'])
        xml_frames.append(time)

    xml_frames_ms = [x*1000 for x in xml_frames]
    return pd.DataFrame(xml_frames_ms)

# calculate frame onset times in voltage data

def detect_on_times(time_array, signal_array, threshold):
    binarized_signal = []
    for signal in signal_array:
        if signal > threshold:
            binarized_signal.append(1)
        else:
            binarized_signal.append(0)

    # get frame starts by finding +1 changes
    # add [0] because diff loses the first value
    signal_changes = [0] + np.diff(binarized_signal)

    timestamps = []
    for i, val in enumerate(signal_changes):
        if val == 1:
            timestamps.append(time_array[i])

    return pd.DataFrame(timestamps)

def pull_nearest_timestamps(xml_frames, ttl_frames):
    true_timestamps = []
    for xml_frame in xml_frames[0]:
        differences = np.abs(ttl_frames[0]-xml_frame)

        #  find ttl timestamp closest to the xml timestamp
        nearest_index = differences.idxmin()
        # assign true timestamp to the ttl time
        nearest_value = ttl_frames[0].iloc[nearest_index]
        true_timestamps.append(nearest_value)
    return pd.DataFrame(true_timestamps)

def plot_timestamps(source1, source2 = None, xlims = [0,1000]):
    fig, ax = plt.subplots()
    ax.vlines(source1, 0, 1, color='red', alpha=0.8)
    if source2 is not None:
        ax.vlines(source2, 0, 0.5, color='green', alpha=0.8)

    plt.xlabel('Time (ms)')
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])