import glob

import ants
import h5py
import pandas as pd
import numpy as np
from src import moco, roi, zdF
import os
from tqdm import tqdm, trange
from nre.io import  save, load
from time import sleep
import nibabel as nib
import gc

# set main data path
base_data_path = "/Volumes/AhmedLab/princess/data/pIP10"

# loop over all directories within princess/data
processed_exps = []
unprocessed_exps = []
for folder_name in os.listdir(os.path.join(base_data_path,'raw')):

    raw_path = os.path.join(base_data_path, 'raw', folder_name)
    processed_path = os.path.join(base_data_path, 'processed', folder_name)

    # do not run on .DS_Store folder
    if folder_name != ".DS_Store":
        # for folders that have a processed + raw folder
        # add to processed experiments list
        #if os.path.isdir(raw_path) and os.path.isdir(processed_path):
        #    processed_exps.append(folder_name)
        #else:
        # on directories that do not have a processed directory
        #    unprocessed_exps.append(folder_name)

        # to run on all experiments
        unprocessed_exps.append(folder_name)
print(f'processed experiments: {processed_exps}')
print(f'unprocessed experiments: {unprocessed_exps}')

for exp in tqdm(unprocessed_exps, desc='unprocessed experiments'):
    if 'win03' in exp:
        continue
    print(f'working on: {exp}')
    path = os.path.join(base_data_path, 'raw', exp)

    # make processed directory
    processed_path = os.path.join(base_data_path, 'processed', exp)
    if not glob.glob(processed_path):
        os.mkdir(processed_path)
    print('    made processed directory')
    # load in functional and structural data
    if glob.glob(os.path.join(path, '*channel_2.nii')):
        func_channel = glob.glob(os.path.join(path, '*channel_2.nii'))[0]
        # struc_channel = glob.glob(os.path.join(path, '*channel_1.nii'))[0]
        struc_channel = func_channel
    else:
        # if there is only data from a single channel
        print(path)
        func_channel = glob.glob(os.path.join(path, '*channel_1.nii'))[0]
        # use the functional data to generate fixed brain
        #struc_channel = func_channel

    # loads in ENTIRE niis
    print('    loading structural brain')
    # struc_data = load_nii(struc_channel)
    volumes = 300
    if not glob.glob(f'{processed_path}/fixed_{volumes}vols.nii'):
        struc_data = load(struc_channel)

    # generate fixed brain
        mean_brain, fixed_brain = moco.generate_fixed(struc_data, volumes)
        del struc_data
        gc.collect()
        print('    generating fixed brain')
        save(f'{processed_path}/fixed_{volumes}vols.nii', mean_brain)
        del mean_brain
        gc.collect()
    else:
        print('    fixed brain already generated')
        fixed_brain = load(f'{processed_path}/fixed_{volumes}vols.nii')
        fixed_brain = ants.from_numpy(fixed_brain)

    # run motion correction on functional data and save out motion corrected brain
    print('    loading functional brain')
    # func_data = load(func_channel)
    func_data = nib.load(func_channel, mmap=True)
    dimensions = pd.DataFrame(func_data.shape)
    print(f'    running motion correction with shape {dimensions}')
    # moco_func_brain = []
    # try up to 3 times

    for idx, image in enumerate(range(func_data.shape[-1])):
        # for i in trange(func_data.shape[-1]):
        if not glob.glob(f'{processed_path}/motion_corrected_volume{idx}.nii'):
            single_volume_data = func_data.dataobj[..., idx]
            single_volume_data = np.asarray(single_volume_data)
            moco_frame = moco.motion_correction(single_volume_data, fixed_brain)
        #moco_func_brain.append(moco_frame)

            save(f'{processed_path}/motion_corrected_volume{idx}.nii', moco_frame)
            del moco_frame
            del single_volume_data
            # print('sleeping for 2 seconds')
            sleep(2)

    # print(getsizeof(moco_func_brain))
    del func_data
    del fixed_brain
    gc.collect()
    # print(getsizeof(moco_func_brain))



    #print('    clustering pixels')
    #n_clusters = 20
    #cluster_labels = roi.extract_ROIs(moco_func_brain, n_clusters)
    #print('    calculating df/F signal')
    #df = zdF.calculate_zscoredF(moco_func_brain, cluster_labels, n_clusters)
    #del moco_func_brain

    # make cluster + zdF h5
    #hf = h5py.File(f'{processed_path}/{n_clusters}_signals.h5', 'w')
    #hf.create_dataset('labels', data=cluster_labels)
    #hf.create_dataset('df/f', data=df)
    #hf.close()
    #del df
    #gc.collect()

    # load in csv
    # csv_file = glob.glob(os.path.join(path, '*csv'))[0]
    #
    # print('    saving acquisition parameters')
    # ttls = ttl.read_csv(csv_file)
    # scope_timestamps = ttl.extract_2p_relative_timestamps(ttls)
    # camera_timestamps = ttl.extract_camera_relative_timestamps(ttls)
    # scope_framerate = ttl.get_frame_rate(pd.Series(scope_timestamps))
    # camera_framerate = ttl.get_frame_rate(pd.Series(camera_timestamps))
    #
    # hf_path = f'{processed_path}/acquisition_parameters.h5'
    # with h5py.File(hf_path,'w') as hf:
    #     hf.create_dataset('scope_fr', data=scope_framerate)
    #     hf.create_dataset('camera_fr', data=camera_framerate)
    #     hf.create_dataset('brain_dimensions', data = dimensions)
    # hf.close()
    # camera_framerate = 170
    # # load fictrac data
    # dat_file = glob.glob(os.path.join(path, '*.dat'))[0]
    # fictrac_data = pd.DataFrame(pd.read_csv(dat_file, header=None))
    #
    # print('    smoothing and saving fictrac speed')
    # win_size = int(.5 * camera_framerate)
    # inst_speed = np.rad2deg(fictrac_data[18])
    # smoothed_speed = savgol_filter(inst_speed, win_size, 3)
    #
    # xy_pos = pd.DataFrame({'x': fictrac_data[14], 'y': fictrac_data[15]})
    # delta_rot = pd.DataFrame({'x': np.rad2deg(fictrac_data[5]), 'y': np.rad2deg(fictrac_data[7]), 'z' : np.rad2deg(fictrac_data[7])})
    #
    # hf_path = f'{processed_path}/fictrac.h5'
    # with h5py.File(hf_path,'w') as hf:
    #     hf.create_dataset('smoothed_speed', data=smoothed_speed)
    #     hf.create_dataset('2d_pos', xy_pos)
    #     hf.create_dataset('delta_rot', delta_rot)
    # hf.close()