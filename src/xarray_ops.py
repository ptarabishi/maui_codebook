from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

# from flymetrics.epochs import shift_epochs_times
#
# def shift_xary_times(xary:xr.DataArray,
#                      time_0:str= "2025-01-01T00:00:00",
#                      inplace:bool=True):
#     # todo remove dependency on flymetric.epochs
#     if inplace:
#         pass
#     else:
#         xary = xary.copy()
#
#     timestamps = xary['time'].to_dataframe().rename(
#         columns={'time': 'timestamp'}).reset_index(drop=True)
#
#     _ = shift_epochs_times(timestamps, time_0=time_0, inplace=True)
#
#     xary['time'] = timestamps['timestamp']
#
#     if inplace:
#         return None
#     else:
#         return xary


def concat_flex_time(xary_coll: list[xr.DataArray], **kwargs) -> xr.DataArray:
    """
    Special case of xarray concat where the 'time' dimension is aligned by
    index positions and not index value. This allows flexible and robust time
    dimension alignment.
    It asumes equal duration time samples and equal start time.

    :param xary_coll: collection of xarray objects
    :param kwargs: keyword arguments to pass to xarray.concat
    :return: xarray.DataArray
    """
    durations = list()
    to_concat = list()
    for xary in xary_coll:
        assert np.issubdtype(xary['time'].dtype, np.datetime64)

        time_coords = xary['time'].values
        durations.append(dict(start=time_coords[0], stop=time_coords[-1]))

        xary = xary.drop_vars('time')
        xary['time'] = np.arange(xary['time'].size)
        to_concat.append(xary)

    flex_cat = xr.concat(to_concat, **kwargs)

    # how to handle different timestamps?
    # easy way first, take start min and stop max
    durations = pd.DataFrame(durations)

    time_coords = pd.date_range(
        durations.start.min(), durations.stop.max(), periods=flex_cat.time.size
    )

    flex_cat['time'] = time_coords
    return flex_cat


def stack_epochs(
        xary: xr.DataArray,
        epochs: Union[str, list[str]],
        epoch_df: pd.DataFrame,
        start_shift: np.timedelta64 = np.timedelta64(0, 's'),
        end_shift: np.timedelta64 = np.timedelta64(0, 's'),
        mode: str = 'interval',
        verbose: str = True,
):
    """
    Given an xarray with epochs on its time dimension, slices the specified
    epochs generating an xarray with extra dimension for trials and epochs.
    :param xary: xarray with time as one of its dimensions
    :param epochs: name or list of names of epochs to slice
    :param epoch_df: pandas dataframe of epochs with start and stop columns
    :param start_shift: time delta to offset the slice start
    :param end_shift: time delta to offset the slice end
    :param mode: 'interval' uses epoch specified start and stop, 'start' uses
     only the epoch start, require start and end shift. 'stop' same but relative
     to epoch stop
    :return: xarray with original dimensions and additional dimensions of
    stacked trials for each epoch. the array can be ragged on the end of time and
    trial dimensions due to different number of trials present for each epoch.
    """

    if 'timestamp' in xary.coords:
        timestamps = xary['timestamp'].values
    elif 'time' in xary.coords:
        timestamps = xary['time'].values
    else:
        raise ValueError("xarry must have 'time' or 'timestamp' coordinates")

    rec_start = timestamps[0]
    rec_stop = timestamps[-1]

    if isinstance(epochs, str):
        epochs = [epochs]

    unique_epochs = epoch_df.epoch.unique()

    epoch_stack = list()
    durations = list()
    for epoch in epochs:
        if epoch not in unique_epochs:
            if verbose:
                warn("specified epoch not in epochs dataframe")
            continue

        df_subset = epoch_df.query(f"epoch == '{epoch}'"
                                   ).dropna().reset_index(drop=True)

        trial_stack = list()
        for trial_index, row in df_subset.iterrows():
            if mode == 'interval':
                slice_start = row.start + start_shift
                slice_stop = row.stop + end_shift
            elif mode == 'start':
                slice_start = row.start + start_shift
                slice_stop = row.start + end_shift
            elif mode == 'stop':
                slice_start = row.stop + start_shift
                slice_stop = row.stop + end_shift
            else:
                raise ValueError("unrecognized mode, "
                                 "select 'interval' or 'start' or 'stop'")

            # Drop slices falling outside the recording time.
            # Likely due to the padding before and after.
            if slice_start < rec_start or slice_stop > rec_stop:
                if verbose:
                    print("dropping slice outside recording bounds")
                continue

            # keep trial number in indexed dimension.
            time_slice = xary.sel(
                time=slice(slice_start, slice_stop)
            ).expand_dims(dim=dict(trial=[trial_index]), axis=[-1])

            # replaces timestamp index with simple integer index
            # to enable alignment of ragged times and/or handle slight
            # differences in time stamps
            time_slice = time_slice.drop_vars('time')
            time_slice['time'] = np.arange(time_slice['time'].size)

            trial_stack.append(time_slice)
            durations.append(slice_stop - slice_start)

        # concatenates trials, and add dimension with epoch information for
        # later concatenation
        trial_stack = xr.concat(
            trial_stack, dim='trial'
        ).expand_dims(dim=dict(epoch=[epoch]))

        epoch_stack.append(trial_stack)

    epoch_stack = xr.concat(epoch_stack, dim='epoch')
    durations = pd.Series(durations)  # series handle times better than array

    if mode == 'interval':
        dur_std = np.std(durations)
        if dur_std > np.timedelta64(1, 's'):
            warn("high epoch duration variability. "
                 f"are the epochs ragged? std = {dur_std}")

    # defines common time coordinates based on the longest duration
    t0 = np.datetime64("2025-01-01T00:00:00")
    epoch_stack['time'] = pd.date_range(
        t0 + start_shift,
        t0 + start_shift + np.max(durations),
        periods=epoch_stack.time.size,
    )

    return epoch_stack