import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import numpy as np

# vis features
# div_cmap = sns.color_palette('coolwarm', as_cmap=True)
# sing_cmap = sns.color_palette('light:b', as_cmap=True)
# color = 'b'

def plot_zscored_activity(clusters_toplot,timestamp_array, signal_array):
    fig, ax = plt.subplots(figsize=(20,len(clusters_toplot)))
    yshift = 5
    n_lines = len(clusters_toplot)
    palette_colors = sns.color_palette('gist_ncar_r', len(clusters_toplot)+1)

    for idx, ind_cluster in enumerate(clusters_toplot):
        activity = signal_array[ind_cluster[0],ind_cluster[1]]
        ax.plot(timestamp_array, activity + (yshift*idx), color=palette_colors[idx])

def plot_spatial_location(clusters_toplot,cluster_2d_arr, brain_dimensions):
    fig, axes = plt.subplots(nrows=6, ncols=6,figsize=(40,40), constrained_layout=True)
    palette_colors = sns.color_palette('gist_ncar_r', len(clusters_toplot), as_cmap=True)

    axes = axes.flatten()
    blank_brain = create_blank_brain(cluster_2d_arr)
    for cluster in clusters_toplot:
        identify_single_cluster(cluster_2d_arr, blank_brain, cluster)

    blank_brain_clusters = blank_brain.reshape(brain_dimensions[2], brain_dimensions[0], brain_dimensions[1])
    raw_brain_clusters = cluster_2d_arr.reshape(brain_dimensions[2], brain_dimensions[0], brain_dimensions[1])


    for idx in range(brain_dimensions[2]):
        ax = axes[idx]
        ax.imshow(raw_brain_clusters[idx].T, cmap='gray_r', alpha=0.8)
        ax.imshow(blank_brain_clusters[idx].T, cmap=palette_colors, alpha=0.8)
        ax.set_title(f'Slice {idx}')

    for j in range(brain_dimensions[2], len(axes)):
        fig.delaxes(axes[j])

# def plot_spatial_clusters(spatial_array, normalize_colors=False):
#     figure = plt.figure(figsize=(40, 20), constrained_layout=True)
#     gspec = figure.add_gridspec(4, 8)
#     nrows, ncols = gspec.get_geometry()
#     axs = np.array([[figure.add_subplot(gspec[i, j]) for j in range(ncols)] for i in range(nrows)])
#     norm = None
#
#     if normalize_colors == True:
#         color_range = Normalize(vmin=np.min(spatial_array), vmax=np.max(spatial_array))
#         # fig.subplots_adjust(right=0.8)
#         # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     else:
#         color_range = None
#     norm = color_range
#
#     counter = 0
#     images = []
#     for i in range(nrows):
#         for j in range(ncols):
#             images.append(axs[i, j].imshow(spatial_array[counter].T, norm=norm))
#             axs[i, j].set_title(f'Z-Slice {counter}')
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])
#             counter += 1
#             if counter == spatial_array.shape[0]:
#                 break
#     figure.colorbar(images[0], ax=axs)

# to make spatial array of clusters of interest
def create_blank_brain(cluster_2d_arr):
    blank_brain = np.zeros(cluster_2d_arr.shape)
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