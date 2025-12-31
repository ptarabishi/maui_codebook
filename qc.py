import glob
import numpy as np
import os
from src.io import load_nii, save_nii

def make_averaged_nii(path):
    # find all nii files
    nii_files = glob.glob(path + '/*.nii')
    print(path, 'number of nii found: ', len(nii_files))
    for file in nii_files:
        raw_volume = load_nii(file)
        print(os.path.basename(file))

        # make average image across z
        averaged_volume = np.mean(raw_volume, axis=2)
        print(f'original shape: {raw_volume.shape}, averaged shape: {averaged_volume.shape}')

        # save new averaged nii to raw directory
        output_path = os.path.join(path, f'{os.path.basename(file[:-4])}_AVG.nii')

        save_nii(output_path, averaged_volume)
        print('saving averaged volume as ', output_path)

if __name__ == '__main__':
    user_input = input('Paste experiment directory: ')
    make_averaged_nii(user_input)