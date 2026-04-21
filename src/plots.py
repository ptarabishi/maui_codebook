import xarray as xr
from plotly import express as px
from plotly import graph_objects as go
import numpy as np

from flymetrics.motion import egocenter_xary


# tableau colors
Blue = '#4E79A7'
Orange = '#F28E2B'
Red = '#E15759'
Teal = '#76B7B2'
Green = '#59A147'
Yellow = '#EDC948'
Purple = '#B07AA1'
Pink = '#FF9DA7'
Brown = '#9C755F'
Grey = '#BAB0AC'
Black = '#000000'
White = '#FFFFFF'
HotPink = "#ff97ff"

COLORDICT = {
    'Blue': Blue, 'Green': Green, 'Orange': Orange, 'Yellow': Yellow,
    'Purple': Purple, 'Red': Red, 'Pink': Pink, 'Grey': Grey, 'Teal': Teal,
    'Brown': Brown, 'Black': Black, 'White': White, 'HotPink': HotPink,
}

FOURCOLOR = [Grey, Yellow, Red, Teal, Brown]
TENCOLOR = list(COLORDICT.values())
TENCOLOR = [val for key, val in COLORDICT.items() if key not in
            ['Black', 'White', 'Grey']]


def hex2rbga(hex, opacity):
    hex = hex.strip('#')
    rgba = (f'rgba({int(hex[0:2], 16)},'
            f'{int(hex[2:4], 16)},'
            f'{int(hex[4:6], 16)},'
            f'{opacity})')
    return rgba

def hex2bgr(hex):
    return ([int(hex.strip('#')[i:i + 2], 16) for i in (4, 2, 0)])


def plot_proximity(ary: xr.DataArray, ref_animal: str, obs_animal: str):
    """
    Takes an xarray of SLEAP data and a track index, plots the vicinity heatmap
    for the observed track relative to the selected track.
    :param ary: xarray of shape timesamples x nodes x XY x tracks
    :param ref_track_idx: index of reference track
    :param obs_track_idx: index of observed (ploted) track
    :return:
    """

    ego_xary = egocenter_xary(
        xary=ary,
        animal=ref_animal,
    )

    df = ego_xary.sel(animal=obs_animal).unstack(level=2)
    df.columns = df.columns.droplevel(0)  #
    df.reset_index(level=1, inplace=True)

    fig = px.density_heatmap(
        data_frame=df, x='x', y='y', histfunc='count',
        histnorm='percent'
    )
    fig.update_layout(
        xaxis_constrain='domain',
        yaxis_constrain='domain',
        yaxis_scaleanchor='x'
    )
    return fig


def plot_PSTH(
        xarry,
        CI='sem',
        CI_opacity=0.2,
        name='default',
        showlegend=True,
        trial_dim='trial',
        time_dim='time',
        **line_kwargs
):
    line_defaults = dict(color="#000000", width=3, shape='linear')
    line_defaults.update(line_kwargs)

    # sanitizes ragged xarrays removing nan trials
    xarry = xarry.dropna(dim=trial_dim, how='all')
    xarry = xarry.dropna(dim=time_dim, how='any')

    fig = go.Figure()

    psth = xarry.mean(dim=trial_dim)
    time = xarry.coords[time_dim]

    if CI == 'std':
        err = xarry.std(dim=trial_dim)
        up_ci = psth + err
        lo_ci = psth - err
    elif CI == 'sem':
        err = xarry.std(dim=trial_dim) / np.sqrt(xarry[trial_dim].size)
        up_ci = psth + err
        lo_ci = psth - err
    elif isinstance(CI, float):
        # plots selected quantile base confidence interval
        alpha = 1-CI
        quant = xarry.quantile([alpha/2, 1-alpha/2], dim=trial_dim)
        up_ci = quant.sel(quantile=1-alpha/2)
        lo_ci = quant.sel(quantile=alpha/2)
    elif CI == 'all' or CI == None:
        # no need to calculate CI if we are ploting all the lines
        # or just the mean. Bypass calculation.
        pass
    else:
        raise ValueError(
            f"Unknown CI value {CI}. Use None, 'all', 'std', 'sem' or a value "
            f"betwee [0, 1)")

    if CI in ['std', 'sem'] or isinstance(CI, float):
        # confidence interval has been defined previously, plots as
        # a shaded area between two transparent, width-less invisible lines.
        ci_line_defaults = line_defaults.copy()
        ci_line_defaults.update(dict(color='rgba(0,0,0,0)', width=0))
        fill_color = hex2rbga(line_defaults['color'], CI_opacity)

        for y, fill, in zip([lo_ci, up_ci], [None, 'tonexty']):
            _ = fig.add_trace(go.Scatter(
                x=time, y=y, mode='lines', line=ci_line_defaults,
                fill=fill, fillcolor=fill_color, legendgroup=name,
                hoverinfo='none',showlegend=False
            ))
    elif CI == 'all':
        # plots all traces in thin faint lines.
        trial_line_defaults = line_defaults.copy()
        trial_line_defaults.update(dict(width=1))
        for trial in xarry.transpose(trial_dim, time_dim):
            _ = fig.add_trace(go.Scatter(
                x=time, y=trial, mode='lines', line=trial_line_defaults,
                legendgroup=name, opacity=CI_opacity,
                showlegend=False
            ))

    _ = fig.add_trace(go.Scatter(
        x=time, y=psth, mode='lines', line=line_defaults,
        name=name,legendgroup=name, showlegend=showlegend
    ))

    return fig

def plot_tracks(xarry, node='thorax', **scatter_kwargs):

    scatter_defaults = dict(line_width=5, mode='lines', opacity=0.5)
    scatter_defaults.update(scatter_kwargs)


    fig = go.Figure()
    for aa, animal in enumerate(xarry.animal.values):
        toplot = xarry.sel(animal=animal, node=node, drop=True)

        fig.add_trace(go.Scatter(
            x=toplot.sel(poss='x').values,
            y=toplot.sel(poss='y').values,
            line_color=TENCOLOR[aa % 10],
            name=animal,
            hovertext=toplot['time'].values,
            **scatter_defaults
        ))

    fig.update_layout(yaxis_scaleanchor='x')
    return fig


def plot_poss_over_time(xarray):
    fig = go.Figure()
    for animal, color in zip(xarray['animal'].values, [Blue, Green]):
        for poss, dash in zip(xarray['poss'].values, ['solid', 'dash']):
            fig.add_trace(go.Scatter(
                y=xarray.sel(
                    animal=animal, poss=poss, node='thorax', drop=True
                ).values,
                line=dict(color=color, dash=dash), name=f'{animal} {poss}'
            ))
    return fig


def plot_inference(xarray, skeleton, **scatter_kwargs):
    """
    Plots the inference for all animals and nodes for a given time sample.
    :param xarray: inference xarray with dimensions node x poss x animal
    :param skeleton: list of couples specifying edges between node pairs
    :return:
    """

    fig = go.Figure()
    for aa, animal in enumerate(xarray['animal'].values):
        animal_slice = xarray.sel(animal=animal)

        for ee, edge in enumerate(skeleton):
            showlegend = True if ee == 0 else False

            x = animal_slice.sel(poss='x', node=edge).values
            y = animal_slice.sel(poss='y', node=edge).values

            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers+lines', name=str(animal),
                legendgroup=str(animal), text=edge,
                showlegend=showlegend,
                marker=dict(color=TENCOLOR[aa % 10]),
                **scatter_kwargs,
            ))

    fig.update_layout(yaxis=dict(scaleanchor='x', autorange='reversed'))
    return fig