import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import numpy as np

# vis features
div_cmap = sns.color_palette('coolwarm', as_cmap=True)
sing_cmap = sns.color_palette('light:b', as_cmap=True)
color = 'b'

def plot_raw_calcium_traces(timestamps, signal_2d, sv_toplot=30, y_shift=5):
    """
    Inputs:
        timestamps: timeseries of experiment
        signal_2d: array of (supervoxel, df/F timeseries)
        sv_toplot: number of supervoxel traces to plot
        y_shift: adjust for more/less space in between traces
    """

    figure = plt.figure(figsize=(20, 20))
    for idx, svoxel in enumerate(signal_2d):
        trace = svoxel + (y_shift*idx)
        plt.plot(timestamps, trace, color=color)
        if idx == sv_toplot:
            break

def plot_fictrac_and_calcium(timestamps, fictrac_param, signal_2d,  sv_toplot=30, y_shift=5):
    """
    Plots df/F calcium traces of 2d signal array aligned with behavior

    Inputs:
    timestamps: timeseries of experiment
    fictrac_param: array of fictrac parameter over time
    signal_2d: array of (supervoxel, df/F timeseries)
    sv_toplot: number of supervoxel traces to plot
    y_shift: adjust for more/less space in between traces
    """

    figure = plt.figure(figsize=(30,20))
    gs = GridSpec(nrows=3, ncols=1, figure=figure)

    subplot1 = figure.add_subplot(gs[0:2, :])
    for idx, svoxel in enumerate(signal_2d):
        trace = svoxel + (y_shift*idx)
        subplot1.plot(timestamps, trace, color=color)
        if idx == sv_toplot:
            break
    plt.xlim(0, max(signal_2d)+10)
    
    subplot2 = figure.add_subplot(gs[2, :])
    subplot2.plot(fictrac_param, color=color)
    plt.ylim(max(fictrac_param)*1.5)
    plt.xlim(0, max(signal_2d)+10)
    plt.xlabel('time(s)')


def plot_spatial_clusters(spatial_array, normalize_colors=False):
    figure = plt.figure(figsize=(40, 20), constrained_layout=True)
    gspec = figure.add_gridspec(4, 8)
    nrows, ncols = gspec.get_geometry()
    axs = np.array([[figure.add_subplot(gspec[i, j]) for j in range(ncols)] for i in range(nrows)])
    norm = None

    if normalize_colors == True:
        color_range = Normalize(vmin=np.min(spatial_array), vmax=np.max(spatial_array))
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    else:
        color_range = None
    norm = color_range

    counter = 0
    images = []
    for i in range(nrows):
        for j in range(ncols):
            images.append(axs[i, j].imshow(spatial_array[counter].T, norm=norm))
            axs[i, j].set_title(f'Z-Slice {counter}')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            counter += 1
            if counter == spatial_array.shape[0]:
                break
    figure.colorbar(images[0], ax=axs)