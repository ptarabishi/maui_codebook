import gc
import h5py
from maui_codebook import loader, zdF, roi, timesync
import glob
import time
import numpy as np
from tqdm import trange
import nibabel as nib
from maui_codebook import visualizations
from loguru import logger
from maui_analysis.supervoxels import calculate_zscoredF_voxels, extract_ROIs_from_zscored

def main(experiment_path, n_clusters):

    all_nii = glob.glob(experiment_path + '/motion_corrected_*.nii')
    print(f'for experiment {experiment_path}, there are {len(all_nii)} nii files')

    # to z-score first then cluster
    #T = len(all_nii)
    T = 500
    if len(all_nii) < T:
        T = len(all_nii)
    brain_array = np.empty([512, 512, 11, T])
    slice_dimensions = [brain_array.shape[0], brain_array.shape[1]]

    logger.info("Starting Brain Array Assignment")

    start_time = time.time()
    for i in trange(brain_array.shape[-1]):
        # t0 = time.time()
        single_nii = f'{experiment_path}/motion_corrected_volume{i}.nii'
        file = nib.load(single_nii)
        data = file.get_fdata()
        brain_array[..., i] = data
        # logger.debug(f"Assignment for slice: {i} took: {time.time() - t0}s")
        del data, file
    logger.info(f"Brain Array Assignment Completed in {time.time() - start_time}s")
    brain_array = brain_array.astype(np.float32)

    logger.info("Starting zscoredF calculation")
    t0 = time.time()
    df = calculate_zscoredF_voxels(brain_array)
    del brain_array
    gc.collect()
    logger.info(f'{time.time() - t0}s to calculate zscoredF voxels')

    t0 = time.time()
    logger.info(f"Extracting ROIs from zscoredF voxels for {n_clusters} clusters and across: {slice_dimensions}")
    cluster_labels = extract_ROIs_from_zscored(df, n_clusters=n_clusters, dimensions=slice_dimensions)
    logger.info(f"Finished extracting ROIs")
    cluster_array = np.asarray(cluster_labels)
    logger.info(f"converted cluster labels to array")
    # print(f'{time.time() - t0}s to extract ROIs')
    logger.info(f'{time.time() - t0}s to extract ROIs')

    hf_name = f'{experiment_path}/{n_clusters}_signals_260713.h5'
    hf = h5py.File(hf_name, 'w')
    logger.info(f"Saving to h5py file")
    hf.create_dataset('labels', data=cluster_array)
    logger.info(f"Saved to h5py file")
    hf.close()
    print(f'saved as {hf_name}')


if __name__ == '__main__':
    all_exps = glob.glob('/Volumes/AhmedLab/princess/data/pIP10/processed/*win01*') + glob.glob(
        '/Volumes/AhmedLab/princess/data/pIP10/processed/*win02*')
    for exp in all_exps:
        if glob.glob(f'{exp}/*signals_260713.h5'):
            continue
        main(exp, n_clusters = 20)