import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, ListedColormap
import numpy as np


# vis features
# div_cmap = sns.color_palette('coolwarm', as_cmap=True)
# sing_cmap = sns.color_palette('light:b', as_cmap=True)
# color = 'b'


def plot_zscored_activity(clusters_toplot,signal_xarray):
    fig, ax = plt.subplots(figsize=(20,len(clusters_toplot)))
    yshift = 5
    # n_lines = len(clusters_toplot)
    palette_colors = sns.color_palette('Blues', len(clusters_toplot))

    for idx, ind_cluster in enumerate(clusters_toplot):
        to_plot = signal_xarray.isel(zposs=ind_cluster[0], roi=ind_cluster[1])
        ax.plot(signal_xarray.coords['time'],to_plot + (yshift*idx), color='b') #color = palette_colors[idx]
        # activity = signal_array[ind_cluster[0],ind_cluster[1]]
        # ax.plot(timestamp_array, activity + (yshift*idx), color=palette_colors[idx])

def plot_spatial_location(clusters_toplot,cluster_2d_arr, brain_dimensions):
    fig, axes = plt.subplots(nrows=6, ncols=6,figsize=(40,40), constrained_layout=True)
    palette_colors = sns.color_palette('Blues_r', 25)
    cmap = ListedColormap(palette_colors)

    axes = axes.flatten()
    blank_brain = create_blank_brain(cluster_2d_arr)
    for cluster in clusters_toplot:
        identify_single_cluster(cluster_2d_arr, blank_brain, cluster)

    blank_brain_clusters = blank_brain.reshape(brain_dimensions[2], brain_dimensions[0], brain_dimensions[1])
    raw_brain_clusters = cluster_2d_arr.reshape(brain_dimensions[2], brain_dimensions[0], brain_dimensions[1])


    for idx in range(brain_dimensions[2]):
        ax = axes[idx]
        ax.imshow(raw_brain_clusters[idx].T, cmap='gray_r', alpha=0.7)
        ax.imshow(blank_brain_clusters[idx].T, cmap=cmap, alpha=1)
        ax.set_title(f'Slice {idx}')

    for j in range(brain_dimensions[2], len(axes)):
        fig.delaxes(axes[j])

# to make spatial array of clusters of interest
def create_blank_brain(cluster_2d_arr):
    blank_brain = np.full(cluster_2d_arr.shape, np.nan)
    return blank_brain


def identify_single_cluster(cluster_2d_arr, blank_brain, cluster):
    if blank_brain.shape == cluster_2d_arr.shape:
        pass
    for slice, pixels in enumerate(cluster_2d_arr):
        if slice == cluster[0]:
            for idx, cluster_id in enumerate(pixels):
                if cluster_id == cluster[1]:
                    blank_brain[slice, idx] = cluster_id
    return blank_brain