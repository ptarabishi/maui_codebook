#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI: functions for processing voxels into ROIs
11 dec 2023
@author: sama ahmed

todo
[ ]
"""

import numpy as np

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from scipy.ndimage import gaussian_filter1d

import logging

LOG = logging.getLogger(__name__)


def create_2d_clusters(brain_slice, n_clusters: int, mempath: str):
    """
    inputs:
        brain_slice: 3D array (x, y, t)
        n_clusters: # of clusters to generate (e.g. 2000)
        mempath: path to cache the output of the computation of the tree

    usage:
        from brainviz import roi
        n_clusters = 2000
        cluster_model = roi.create_2d_clusters(slice, n_clusters, 'dat/cluster_mem')
        labels = []
        labels.append(cluster_model.labels_)

    this is based on Luke Brezovec's "create_clusters" function
    """

    xdim = brain_slice.shape[0]
    ydim = brain_slice.shape[1]
    tdim = brain_slice.shape[2]

    # enforce that clustered voxels must be neighbors
    connectivity = grid_to_graph(xdim, ydim)

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, memory=mempath, linkage='ward', connectivity=connectivity)

    super_to_cluster = brain_slice.reshape(-1, tdim)
    cluster_model.fit(super_to_cluster)

    return cluster_model


def get_supervoxel_mean_2D(brain_slice, cluster_labels, n_clusters: int):
    """
    inputs:
        brain: 2D array (x, y)
        cluster_labels: (x*y) from |create_2d_clusters| (see example code above)
        n_clusters: e.g. 2000
    outputs:
        signals: mean df/f of supervoxel (n_clusters)
        cluster_idx: list (len = n_clusters) w/brain_slice idx mapped to a cluster
    """
    x_by_y = brain_slice.shape[0] * brain_slice.shape[1]
    neural_data = brain_slice.reshape(x_by_y)  # make into vector

    signals = []
    cluster_idx = []

    for nn in range(n_clusters):
        idx = np.where(cluster_labels == nn)[0]
        mean_signal = np.nanmean(neural_data[idx])

        signals.append(mean_signal)
        cluster_idx.append(idx)

    return np.asarray(signals), cluster_idx


def dFdt(rois):
    """
    Returns a matrix for which each row is a dF/dt signal, computed using a Gaussian derivative kernel of width (2 * sigma=3 / volume_rate) seconds

    If the volume_rate = 3.19 --> (2 * sigma=3 / 3.19) --> 1.88 seconds filter window

    inputs:
        rois: 2D array (n, t)
    outputs:
        d(rois)/dt: 2D array (n, t)
    """
    nan_zero = np.copy(rois)
    nan_zero[np.isnan(rois)] = 0
    nan_zero_filtered = gaussian_filter1d(nan_zero, sigma=3, order=1)

    flat = 0 * rois.copy() + 1
    flat[np.isnan(rois)] = 0
    flat_filtered = gaussian_filter1d(flat, sigma=3, order=0)

    with np.errstate(invalid = 'ignore'):
        return nan_zero_filtered / flat_filtered

def extract_ROIs(nii, n_clusters):
    labels = []
    for iSlice in range(nii.shape[2]): # for each slice in Z
    # generate n_clusters within a single slice

        cluster_model = create_2d_clusters(nii[:, :, iSlice, 0:-1:5], n_clusters, 'tmp/cluster_mem')
        labels.append(cluster_model.labels_)
    return labels