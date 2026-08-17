import glob
import numpy as np
import nibabel as nib
from nre.io import save

def load_brain_data(experiment_path, starting_vol = 0, ending_vol = 300):
    all_nii = glob.glob(experiment_path + '/motion_corrected_*.nii')
    n_volumes = ending_vol - starting_vol

    brain_array = np.empty([512, 512, 11, n_volumes])
    for i in range(n_volumes):
        single_nii = f'{experiment_path}/motion_corrected_volume{starting_vol + i}.nii'
        file = nib.load(single_nii)
        data = file.get_fdata()
        brain_array[..., i] = data
        del data, file

    return brain_array

def save_brain_array(brain_array, save_path):
    brain_array = brain_array.astype(np.float32)
    save(f'{save_path}.nii', brain_array)

if __name__ == "__main__":
    experiment_path = '/Volumes/AhmedLab/princess/data/pIP10/processed/pIP10_TSeries_20260427_GC8m_fly09_win01_trial-001'
    starting_vol = 300
    ending_vol = 900
    brain_array = load_brain_data(experiment_path, starting_vol=starting_vol, ending_vol=ending_vol)
    print(brain_array.shape)

    save_brain_array(brain_array, f'{experiment_path}/volumes{starting_vol}-{ending_vol}.nii')
