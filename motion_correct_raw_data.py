import glob
import pandas as pd

from src import loader, moco
import os
from tqdm import tqdm
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
        if os.path.isdir(raw_path) and os.path.isdir(processed_path):
            processed_exps.append(folder_name)
        else:
        # on directories that do not have a processed directory
            unprocessed_exps.append(folder_name)

print(f'processed experiments: {processed_exps}')
print(f'unprocessed experiments: {unprocessed_exps}')

for exp in tqdm(unprocessed_exps, desc='unprocessed experiments'):
    print(f'working on: {exp}')
    path = os.path.join(base_data_path, 'raw', exp)

    # make processed directory
    processed_path = os.path.join(base_data_path, 'processed', exp)
    os.mkdir(processed_path)
    print('    made processed directory')
    # load in functional and structural data
    if glob.glob(os.path.join(path, '*channel_2.nii')):
        func_channel = glob.glob(os.path.join(path, '*channel_2.nii'))[0]
        # struc_channel = glob.glob(os.path.join(path, '*channel_1.nii'))[0]
        struc_channel = func_channel
    else:
        # if there is only data from a single channel
        func_channel = glob.glob(os.path.join(path, '*channel_1.nii'))[0]
        # use the functional data to generate fixed brain
        struc_channel = func_channel

    # loads in ENTIRE niis
    print('    loading functional brain')
    struc_data = io.load_nii(struc_channel)

    # generate fixed brain
    mean_brain, fixed_brain = moco.generate_fixed(struc_data, 300)
    del struc_data
    gc.collect()
    print('    generating fixed brain')
    io.save_nii(f'{processed_path}/fixed.nii', mean_brain)


    # run motion correction on functional data and save out motion corrected brain
    print('    loading functional brain')
    func_data = io.load_nii(func_channel)
    dimensions = pd.DataFrame(func_data.shape)
    print('    running motion correction')
    moco_func_brain = moco.motion_correction(func_data, fixed_brain)
    del func_data
    gc.collect()
    io.save_nii(f'{processed_path}/motion_corrected.nii', moco_func_brain)