import glob
import numpy as np
import os
from nre.io import load, save
import nibabel as nib
from tqdm import trange


def make_averaged_nii(path):
    # find all nii files
    nii_files = glob.glob(path + '/*.nii')
    print(path, 'number of nii found: ', len(nii_files))
    for file in nii_files:
        raw_volume = load(file)
        print(os.path.basename(file))

        # make average image across z
        averaged_volume = np.mean(raw_volume, axis=2)
        print(f'original shape: {raw_volume.shape}, averaged shape: {averaged_volume.shape}')

        # save new averaged nii to raw directory
        output_path = os.path.join(path, f'{os.path.basename(file[:-4])}_AVG.nii')

        save(output_path, averaged_volume)
        print('saving averaged volume as ', output_path)

def stitch_moco_volumes(directory, n_volumes=200, starting_vol=0):
    all_nii_files = glob.glob(directory + '/motion_corrected*.nii')

    # determine final array size
    starting_ind = starting_vol
    ending_ind = starting_ind + n_volumes
    image = nib.load(all_nii_files[0])
    image_shape = image.shape

    brain_array = np.empty([image_shape[0], image_shape[1], image_shape[2], n_volumes])
    for vol in trange(starting_ind, ending_ind):
        single_file = f'{directory}/motion_corrected_volume{vol}.nii'
        file = nib.load(single_file)
        data = file.get_fdata()
        brain_array[..., vol] = data

    save(f'{directory}/moco_check_{n_volumes}volumes.nii', brain_array)