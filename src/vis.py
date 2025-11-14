import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

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
    return figure

def plot_fictrac_and_calcium(timestamps, fictrac_param, signal_2d,  sv_toplot=30, y_shift=5):
    """
    Plots df/F calcium traces of 2d signal array aligned with behavior
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

    return figure