# TODO: Need to remove all matplotlib functions from this file.

from itertools import cycle
from typing import Literal

import numpy as np
from numpy.random import default_rng

from .. import stats
from ..utils import DataHolder, get_transform
from .plot_utils import _bin_data, _calc_hist, process_duplicates, process_jitter
from .types import (
    BW,
    Agg,
    AlphaRange,
    BinType,
    BoxPlotData,
    CountPlotTypes,
    Error,
    Kernels,
    Levels,
    LinePlotData,
    RectanglePlotData,
    JitterPlotData,
    ScatterPlotData,
    SummaryPlotData,
    Transform,
    ViolinPlotData,
    CapStyle,
)

# Reorder the filled matplotlib markers to choose the most different
MARKERS = [
    "o",
    "X",
    "^",
    "s",
    "d",
    "h",
    "p",
    "*",
    "<",
    "H",
    "D",
    "v",
    "P",
    ">",
    "8",
    ".",
]
HATCHES = [
    None,
    "/",
    "o",
    "-",
    "*",
    "+",
    "\\",
    "|",
    "O",
    ".",
    "x",
]


def _jitter(
    data: DataHolder,
    y: str,
    levels: list[str | int],
    loc_dict: dict[str, float],
    width: float,
    color_dict: dict[str, str],
    marker_dict: dict[str, str],
    edgecolor_dict: dict[str, str],
    alpha: AlphaRange = 1.0,
    edge_alpha: AlphaRange = 1.0,
    seed: int = 42,
    markersize: float | int = 2,
    ytransform: Transform = None,
    unique_id: str | None = None,
    *args,
    **kwargs,
) -> JitterPlotData:

    transform = get_transform(ytransform)

    rng = default_rng(seed)

    x_data = []
    y_data = []
    mks = []
    mfcs = []
    mecs = []
    mksizes = []

    groups = data.groups(levels)

    if unique_id is not None:
        unique_groups = data.groups(levels + [unique_id])

    jitter_values = np.zeros(data.shape[0])

    for i, indexes in groups.items():
        temp_jitter = process_jitter(data[indexes, y], loc_dict[i], width, rng=rng)
        jitter_values[indexes] = temp_jitter

    for i, indexes in groups.items():
        if unique_id is None:
            x_data.append(jitter_values[indexes])
            y_data.append(transform(data[indexes, y]))
            mks.append(marker_dict[i])
            mfcs.append(color_dict[i])
            mecs.append(edgecolor_dict[i])
            mksizes.append(markersize)
        else:
            unique_ids_sub = np.unique(data[indexes, unique_id])
            for ui_group, mrk in zip(unique_ids_sub, cycle(MARKERS)):
                sub_indexes = unique_groups[i + (ui_group,)]
                x_data.append(jitter_values[sub_indexes])
                y_data.append(transform(data[sub_indexes, y]))
                mks.append(mrk)
                mfcs.append(color_dict[i])
                mecs.append(edgecolor_dict[i])
                mksizes.append(markersize)

    output = JitterPlotData(
        x_data=x_data,
        y_data=y_data,
        marker=mks,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=mksizes,
        alpha=alpha,
        edge_alpha=edge_alpha,
    )
    return output


def _jitteru(
    data: DataHolder,
    y: str,
    levels: Levels,
    unique_id: str,
    loc_dict: dict[str, float],
    width: float,
    color_dict: dict[str, str],
    marker_dict: dict[str, str],
    edgecolor_dict: dict[str, str],
    alpha: AlphaRange = 1.0,
    edge_alpha: AlphaRange = 1.0,
    duplicate_offset: float = 0.0,
    markersize: int = 2,
    agg_func: Agg | None = None,
    ytransform: Transform = None,
    *args,
    **kwargs,
) -> JitterPlotData:

    transform = get_transform(ytransform)
    temp = width / 2

    x_data = []
    y_data = []
    mks = []
    mfcs = []
    mecs = []
    mksizes = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for i in groups.keys():
        unique_ids_sub = np.unique(data[groups[i], unique_id])
        if len(unique_ids_sub) > 1:
            dist = np.linspace(-temp, temp, num=len(unique_ids_sub) + 1)
            dist = np.round((dist[1:] + dist[:-1]) / 2, 10)
        else:
            dist = [0]
        for index, ui_group in enumerate(unique_ids_sub):
            if i == ("",):
                temp_group = (ui_group,)
            else:
                temp_group = i + (ui_group,)
            sub_indexes = uid_groups[temp_group]
            x = np.full(len(sub_indexes), loc_dict[i]) + dist[index]
            if duplicate_offset > 0.0:
                output = (
                    process_duplicates(data[sub_indexes, y]) * duplicate_offset * temp
                )
                x += output
            if agg_func is None:
                x = get_transform(agg_func)(x)
            else:
                x = x[0]
            x_data.append(x)
            y_data.append(get_transform(agg_func)(transform(data[sub_indexes, y])))
            mks.append(marker_dict[i])
            mfcs.append(color_dict[i])
            mecs.append(edgecolor_dict[i])
            mksizes.append(markersize)
    output = JitterPlotData(
        x_data=x_data,
        y_data=y_data,
        marker=mks,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=mksizes,
        alpha=alpha,
        edge_alpha=edge_alpha,
    )
    return output


def _summary(
    data: DataHolder,
    y: str,
    levels: Levels,
    loc_dict: dict[str, float],
    func: Agg,
    capsize: float,
    capstyle: CapStyle,
    barwidth: float,
    linewidth: float | int,
    color_dict: dict[str, str],
    alpha: AlphaRange,
    err_func: Error | None = None,
    ytransform: Transform = None,
    *args,
    **kwargs,
) -> SummaryPlotData:

    transform = get_transform(ytransform)
    y_data = []
    error_data = []
    colors = []
    x_data = []
    widths = []

    groups = data.groups(levels)
    for i, indexes in groups.items():
        x_data.append(loc_dict[i])
        colors.append(color_dict[i])
        widths.append(barwidth)
        y_data.append(get_transform(func)(transform(data[indexes, y])))
        if err_func is not None:
            error_data.append(get_transform(err_func)(transform(data[indexes, y])))
        else:
            error_data.append(None)
    output = SummaryPlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        widths=widths,
        colors=colors,
        linewidth=linewidth,
        alpha=alpha,
        capstyle=capstyle,
        capsize=capsize,
    )
    return output


def _summaryu(
    data: DataHolder,
    y: str,
    levels: Levels,
    unique_id: str,
    loc_dict: dict[str, float],
    func: Agg,
    capsize: float | int,
    capstyle: CapStyle,
    barwidth: float,
    linewidth: float | int,
    color_dict: dict[str, str],
    alpha: AlphaRange = 1.0,
    agg_func: Agg | None = None,
    err_func: Error = None,
    agg_width: float = 1.0,
    ytransform: Transform = None,
    *args,
    **kwargs,
) -> SummaryPlotData:

    transform = get_transform(ytransform)
    y_data = []
    error_data = []
    colors = []
    x_data = []
    widths = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for i, indexes in groups.items():
        uids = np.unique(data[indexes, unique_id])
        if agg_func is None:
            temp = barwidth / 2
            if len(uids) > 1:
                dist = np.linspace(-temp, temp, num=len(uids) + 1)
                centers = np.round((dist[1:] + dist[:-1]) / 2, 10)
            else:
                centers = [0]
            w = agg_width / len(uids)
            for index, j in enumerate(uids):
                if i == ("",):
                    temp_group = (j,)
                else:
                    temp_group = i + (j,)
                widths.append(w)
                vals = transform(data[uid_groups[temp_group], y])
                x_data.append(loc_dict[i] + centers[index])
                colors.append(color_dict[i])
                y_data.append(get_transform(func)(vals))
                if err_func is not None:
                    error_data.append(get_transform(err_func)(vals))
                else:
                    error_data.append(None)
        else:
            temp_vals = []
            for index, j in enumerate(uids):
                vals = transform(data[uid_groups[i + (j,)], y])
                temp_vals.append(get_transform(func)(vals))
            x_data.append(loc_dict[i])
            colors.append(color_dict[i])
            widths.append(barwidth)
            y_data.append(get_transform(func)(np.array(temp_vals)))
            if err_func is not None:
                error_data.append(get_transform(err_func)(np.array(temp_vals)))
            else:
                error_data.append(None)
    output = SummaryPlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        widths=widths,
        colors=colors,
        linewidth=linewidth,
        alpha=alpha,
        capstyle=capstyle,
        capsize=capsize,
    )
    return output


def _box(
    data: DataHolder,
    y: str,
    levels: Levels,
    loc_dict: dict[str, float],
    color_dict: dict[str, str],
    edgecolor_dict: dict[str, str],
    fliers: str = "",
    width: float = 1.0,
    linewidth: float | int = 1,
    showmeans: bool = False,
    show_ci: bool = False,
    alpha: AlphaRange = 1.0,
    linealpha: AlphaRange = 1.0,
    ytransform: Transform = None,
    *args,
    **kwargs,
):

    transform = get_transform(ytransform)

    y_data = []
    x_data = []
    fcs = []
    ecs = []

    groups = data.groups(levels)
    for key, value in groups.items():
        y_data.append(transform(data[value, y]))
        x_data.append([loc_dict[key]])
        fcs.append(color_dict[key])
        ecs.append(edgecolor_dict[key])
    output = BoxPlotData(
        x_data=x_data,
        y_data=y_data,
        facecolors=fcs,
        edgecolors=ecs,
        alpha=alpha,
        linealpha=linealpha,
        fliers=fliers,
        linewidth=linewidth,
        width=width,
        show_ci=show_ci,
        showmeans=showmeans,
    )
    return output


def _violin(
    data: DataHolder,
    y: str,
    levels: Levels,
    loc_dict: dict[str, float],
    facecolor_dict,
    edgecolor_dict: dict[str, str],
    alpha: AlphaRange = 1.0,
    edge_alpha: AlphaRange = 1.0,
    linewidth: float | int = 1,
    width: float = 1.0,
    ytransform: Transform = None,
    unique_id: str | None = None,
    kernel: Kernels = "gaussian",
    bw: BW = "ISJ",
    kde_length: int | None = None,
    tol: float | int = 1e-3,
    KDEType="fft",
    agg_func: Agg | None = None,
    unique_style: Literal["split", "overlap"] = "overlap",
    *args,
    **kwargs,
):

    transform = get_transform(ytransform)

    groups = data.groups(levels)

    x_data = []
    y_data = []
    loc = []
    ecs = []
    fcs = []
    width = width / 2.0

    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for u, group_indexes in groups.items():
        if unique_id is None:
            y_values = np.asarray(data[group_indexes, y]).flatten()
            x_kde, y_kde = stats.kde(
                get_transform(transform)(y_values),
                bw=bw,
                kernel=kernel,
                tol=tol,
                kde_length=kde_length,
                KDEType=KDEType,
            )
            y_data.append((y_kde / y_kde.max()) * width)
            x_data.append(x_kde)
            ecs.append(edgecolor_dict[u])
            fcs.append(facecolor_dict[u])
            loc.append(loc_dict[u])
        else:
            subgroups = np.unique(data[group_indexes, unique_id])
            if agg_func is not None:
                temp_data = data[group_indexes, y]
                min_data = get_transform(transform)(temp_data.min())
                max_data = get_transform(transform)(temp_data.max())
                min_data = min_data - np.abs((min_data * tol))
                max_data = max_data + np.abs((max_data * tol))
                min_data = min_data if min_data != 0 else -1e-10
                max_data = max_data if max_data != 0 else 1e-10
                x_array = np.linspace(min_data, max_data, num=kde_length)
                y_hold = np.zeros((len(subgroups), kde_length))
            for hi, s in enumerate(subgroups):
                if unique_style == "split":
                    if len(subgroups) > 1:
                        dist = np.linspace(-width, width, num=len(subgroups) + 1)
                        uwidth = (dist[1] - dist[0]) / 2.0
                        dist = np.round((dist[1:] + dist[:-1]) / 2, 10)
                        dist += loc_dict[u]
                    else:
                        dist = [0]
                else:
                    dist = np.full(len(subgroups), loc_dict[u])
                    uwidth = width
                s_indexes = uid_groups[u + (s,)]
                y_values = np.asarray(data[s_indexes, y]).flatten()
                if agg_func is None:
                    x_kde, y_kde = stats.kde(
                        get_transform(transform)(y_values),
                        bw=bw,
                        kernel=kernel,
                        tol=tol,
                        KDEType=KDEType,
                        kde_length=kde_length,
                    )
                    y_data.append((y_kde / y_kde.max()) * uwidth)
                    x_data.append(x_kde)
                    ecs.append(edgecolor_dict[u])
                    fcs.append(facecolor_dict[u])
                    loc.append(dist[hi])
                else:
                    _, y_kde = stats.kde(
                        get_transform(transform)(y_values),
                        bw=bw,
                        kernel=kernel,
                        tol=tol,
                        x=x_array,
                        KDEType=KDEType,
                        kde_length=kde_length,
                    )
                    y_hold[hi, :] = y_kde
            if agg_func is not None:
                x_kde, y_kde = x_array, get_transform(agg_func)(y_hold, axis=0)
                y_data.append((y_kde / y_kde.max()) * width)
                x_data.append(x_kde)
                ecs.append(edgecolor_dict[u])
                fcs.append(facecolor_dict[u])
                loc.append(loc_dict[u])
        output = ViolinPlotData(
            x_data=x_data,
            y_data=y_data,
            location=loc,
            facecolors=fcs,
            edgecolors=ecs,
            alpha=alpha,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
        )
    return output


def _paired_plot():
    pass


def _hist(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    color_dict: dict[str, str],
    facet_dict: dict[str, int],
    hatch: str | dict[str, str] | None = None,
    hist_type: Literal["bar", "step", "stepfilled"] = "bar",
    fillalpha: AlphaRange = 1.0,
    linealpha: AlphaRange = 1.0,
    bin_limits: list[float, float] | None = None,
    nbins=None,
    stat="probability",
    agg_func: Agg | None = None,
    unique_id: str | None = None,
    ytransform: Transform = None,
    xtransfrom=None,
    *args,
    **kwargs,
):
    if hist_type == "bar":
        _bar_histogram(
            data=data,
            y=y,
            x=x,
            levels=levels,
            color_dict=color_dict,
            facet_dict=facet_dict,
            hatch=hatch,
            fillalpha=fillalpha,
            linealpha=linealpha,
            bin_limits=bin_limits,
            nbins=nbins,
            stat=stat,
            agg_func=agg_func,
            unique_id=unique_id,
            ytransform=ytransform,
            xtransfrom=xtransfrom,
            *args,
            **kwargs,
        )


def _bar_histogram(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    color_dict: dict[str, str],
    facet_dict: dict[str, int],
    hatch: str | dict[str, str] | None = None,
    fillalpha: AlphaRange = 1.0,
    linealpha: AlphaRange = 1.0,
    bin_limits: list[float, float] | None = None,
    nbins=None,
    stat="probability",
    agg_func: Agg | None = None,
    unique_id: str | None = None,
    ytransform: Transform = None,
    xtransfrom=None,
    *args,
    **kwargs,
) -> RectanglePlotData:

    y = y if x is None else x
    transform = ytransform if xtransfrom is None else xtransfrom
    axis = "y" if x is None else "x"

    bottom = np.zeros(nbins)
    bw = []
    plot_data = []
    plot_bins = []
    count = 0
    colors = []
    facet = []
    edgec = []
    hatches = []
    linewidth = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            unique_id_indexes = data.groups(levels + [unique_id])
        for i, group_indexes in groups.items():
            if unique_id is not None:
                bins = np.histogram_bin_edges(
                    get_transform(transform)(data[unique_id_indexes[i], y]),
                    bins=nbins,
                    bin_limits=bin_limits,
                )
                x = (bins[1:] + bins[:-1]) / 2
                temp_bw = np.full(nbins, bins[1] - bins[0])
                uids = np.unique(data[group_indexes, unique_id])
                if agg_func is not None:
                    temp_list = np.zeros((len(uids), nbins))
                else:
                    temp_list = []
                for index, j in enumerate(uids):
                    temp_data = np.sort(data[unique_id_indexes[i + (j,)], y])
                    poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                    if agg_func is not None:
                        temp_list[index] = poly
                    else:
                        plot_data.append(poly)
                        colors.append([color_dict[i]] * nbins)
                        edgec.append([color_dict[i]] * nbins)
                        facet.append(facet_dict[i])
                        bw.append(temp_bw)
                        plot_bins.append(bins[:-1])
                        hatches.append([[hatch] * bins.size - 1] * count)
                        count += 1
                if agg_func is not None:
                    plot_data.append(get_transform(agg_func)(temp_list, axis=0))
                    colors.append([color_dict[i]] * nbins)
                    edgec.append([color_dict[i]] * nbins)
                    facet.append(facet_dict[i])
                    bw.append(temp_bw)
                    plot_bins.append(bins[:-1])
                    hatches.append([[hatch] * bins.size - 1] * count)
                    count += 1
            else:
                temp_data = np.sort(data[groups[i], y])
                bins = np.histogram_bin_edges(
                    get_transform(transform)(data[unique_id_indexes[i], y]),
                    bins=nbins,
                    bin_limits=bin_limits,
                )
                bw.append(np.full(nbins, bins[1] - bins[0]))
                x = (bins[1:] + bins[:-1]) / 2
                poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                plot_data.append(poly)
                colors.append([color_dict[i]] * nbins)
                edgec.append([color_dict[i]] * nbins)
                facet.append(facet_dict[i])
                plot_bins.append(bins[:-1])
                hatches.append([[hatch] * bins.size - 1] * count)
                count += 1

    bottom = [bottom for _ in range(count)]
    output = RectanglePlotData(
        tops=plot_data,
        bottoms=bottom,
        bins=plot_bins,
        binwidths=bw,
        fillcolors=colors,
        edgecolors=edgec,
        fill_alpha=fillalpha,
        edge_alpha=linealpha,
        hatches=hatches,
        linewidth=linewidth,
        axis=axis,
    )
    return output


def _scatter(
    data,
    y,
    x,
    levels,
    markers,
    markercolors,
    edgecolors,
    markersizes,
    alpha,
    edge_alpha,
    facetgroup,
    facet_dict=None,
    xtransform=None,
    ytransform=None,
    *args,
    **kwargs,
) -> ScatterPlotData:
    x_data = []
    y_data = []
    mks = []
    mksizes = []
    mfcs = []
    mecs = []
    facet = []

    for key, value in facet_dict.items():
        indexes = np.array([index for index, j in enumerate(facetgroup) if value == j])
        x_data.append(get_transform(xtransform)(data[indexes, x]))
        y_data.append(get_transform(ytransform)(data[indexes, y]))
        mks.append(markers)
        mfcs.append([markercolors[i] for i in indexes])
        mecs.append([edgecolors[i] for i in indexes])
        mksizes.append([markersizes[i] for i in indexes])
        facet.append(facet_dict[key])
    output = ScatterPlotData(
        x_data=x_data,
        y_data=y_data,
        marker=mks,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=mksizes,
        alpha=alpha,
        edge_alpha=edge_alpha,
        facet_index=facet,
    )
    return output


def _aggline(
    data: DataHolder,
    x: str,
    y: str,
    levels: Levels,
    marker: str | dict[str, str],
    markersize: float | int,
    markerfacecolor: str | dict[str, str],
    markeredgecolor: str | dict[str, str],
    linestyle: str | dict[str, str],
    linewidth: float | int,
    linecolor: str | dict[str, str],
    linealpha: float | int,
    facet_dict: dict[str, int],
    func: Agg = None,
    err_func: Error = None,
    fill_between: bool = False,
    fillalpha: AlphaRange = 1.0,
    agg_func: Agg | None = None,
    ytransform: Transform = None,
    xtransform: Transform = None,
    unique_id: str | None = None,
    sort=True,
    *args,
    **kwargs,
) -> LinePlotData:

    x_data = []
    y_data = []
    error_data = []
    facet_index = []
    mks = []
    lcs = []
    lss = []
    mfcs = []
    mecs = []

    err_data = None
    new_levels = (levels + [x]) if unique_id is None else (levels + [x, unique_id])
    new_data = (
        data.groupby(y, new_levels, sort=sort).agg(get_transform(func)).reset_index()
    )
    if unique_id is None:
        if err_func is not None:
            err_data = DataHolder(
                data.groupby(y, new_levels, sort=sort)
                .agg(get_transform(err_func))
                .reset_index()
            )
    else:
        if agg_func is not None:
            if err_func is not None:
                err_data = DataHolder(
                    new_data[levels + [x, y]]
                    .groupby(levels + [x], sort=sort)
                    .agg(get_transform(err_func))
                    .reset_index()
                )
        new_data = (
            new_data[levels + [x, y]]
            .groupby(levels + [x], sort=sort)
            .agg(get_transform(func))
            .reset_index()
        )
    new_data = DataHolder(new_data)
    if unique_id is not None and agg_func is None:
        ugrps = new_data.groups(levels + [unique_id])
    else:
        ugrps = new_data.groups(levels)
    if len(ugrps) != 0:
        for u, indexes in ugrps.items():
            ytemp = get_transform(ytransform)(new_data[indexes, y])
            y_data.append(ytemp)
            xtemp = get_transform(xtransform)(new_data[indexes, x])
            x_data.append(xtemp)
            temp_err = err_data[indexes, y] if err_func is not None else None
            error_data.append(temp_err)
            facet_index.append(facet_dict[u])
            mks.append(marker[u])
            lcs.append(linecolor[u])
            lss.append(linestyle[u])
            mfcs.append(markerfacecolor[u])
            mecs.append(markeredgecolor[u])
    else:
        ytemp = get_transform(ytransform)(new_data[y])
        y_data.append(ytemp)
        xtemp = get_transform(xtransform)(new_data[x])
        x_data.append(xtemp)
        temp_err = err_data[indexes, y] if err_func is not None else None
        error_data.append(temp_err)
        facet_index.append(facet_dict[("",)])
        mks.append(marker[("",)])
        lcs.append(linecolor[("",)])
        lss.append(linestyle[("",)])
        mfcs.append(markerfacecolor[("",)])
        mecs.append(markeredgecolor[("",)])
    output = LinePlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        facet_index=facet_index,
        marker=mks,
        linecolor=lcs,
        linewidth=linewidth,
        linestyle=lss,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=markersize,
        fill_between=fill_between,
        linealpha=linealpha,
        fillalpha=fillalpha,
        fb_direction="y",
    )

    return output


def _kde(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    linecolor: str | dict[str, str],
    facet_dict: dict[str, int],
    linestyle: str | dict[str, str],
    linewidth: float | int,
    linealpha: float | int,
    fillalpha: float | int,
    fill_between: bool,
    fill_under: bool,
    kernel: Kernels = "gaussian",
    bw: BW = "ISJ",
    kde_length: int | None = None,
    tol: float | int = 1e-3,
    common_norm: bool = True,
    unique_id: str | None = None,
    agg_func: Agg | None = None,
    err_func=None,
    xtransform: Transform = None,
    ytransform: Transform = None,
    KDEType="fft",
    *args,
    **kwargs,
) -> LinePlotData:
    size = data.shape[0]

    x_data = []
    y_data = []
    error_data = []
    facet_index = []
    mks = []
    lcs = []
    lss = []
    mfcs = []
    mecs = []

    column = y if x is None else x
    direction = "x" if x is None else "y"
    transform = ytransform if xtransform is None else xtransform

    if len(levels) == 0:
        y_values = np.asarray(data[column]).flatten()
        temp_size = size
        x_kde, y_kde = stats.kde(
            get_transform(transform)(y_values),
            bw=bw,
            kernel=kernel,
            tol=tol,
            kde_length=kde_length,
        )
        if common_norm:
            multiplier = float(temp_size / size)
            y_kde *= multiplier
        if y is not None:
            y_kde, x_kde = x_kde, y_kde
        y_data.append(y_kde)
        x_data.append(x_kde)
        lcs.append(linecolor[("",)])
        lss.append(linestyle[("",)])
        facet_index.append(facet_dict[("",)])
        error_data.append(None)
        mfcs.append(None)
        mecs.append(None)
        mks.append(None)
    else:
        groups = data.groups(levels)

        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for u, group_indexes in groups.items():
            if unique_id is None:
                y_values = np.asarray(data[group_indexes, column]).flatten()
                temp_size = y_values.size
                x_kde, y_kde = stats.kde(
                    get_transform(transform)(y_values),
                    bw=bw,
                    kernel=kernel,
                    tol=tol,
                    kde_length=kde_length,
                )
                if common_norm:
                    multiplier = float(temp_size / size)
                    y_kde *= multiplier
                if y is not None:
                    y_kde, x_kde = x_kde, y_kde
                y_data.append(y_kde)
                x_data.append(x_kde)
                lcs.append(linecolor[u])
                lss.append(linestyle[u])
                facet_index.append(facet_dict[u])
                error_data.append(None)
                mfcs.append(None)
                mecs.append(None)
                mks.append(None)
            else:
                subgroups, count = np.unique(
                    data[group_indexes, unique_id], return_counts=True
                )

                if agg_func is not None:
                    temp_data = data[group_indexes, column]
                    min_data = get_transform(transform)(temp_data.min())
                    max_data = get_transform(transform)(temp_data.max())
                    min_data = min_data - np.abs((min_data * tol))
                    max_data = max_data + np.abs((max_data * tol))
                    min_data = min_data if min_data != 0 else -1e-10
                    max_data = max_data if max_data != 0 else 1e-10
                    if KDEType == "fft":
                        if kde_length is None:
                            kde_length = int(np.ceil(np.log2(len(temp_data))))
                    else:
                        if kde_length is None:
                            max_len = np.max(count)
                            kde_length = int(max_len * 1.5)
                    x_array = np.linspace(min_data, max_data, num=kde_length)
                    y_hold = np.zeros((len(subgroups), x_array.size))
                for hi, s in enumerate(subgroups):
                    s_indexes = uid_groups[u + (s,)]
                    y_values = np.asarray(data[s_indexes, column]).flatten()
                    temp_size = y_values.size
                    if agg_func is None:
                        x_kde, y_kde = stats.kde(
                            get_transform(transform)(y_values),
                            bw=bw,
                            kernel=kernel,
                            tol=tol,
                        )
                        if y is not None:
                            y_kde, x_kde = x_kde, y_kde
                        y_data.append(y_kde)
                        x_data.append(x_kde)
                        lcs.append(linecolor[u])
                        lss.append(linestyle[u])
                        facet_index.append(facet_dict[u])
                        mfcs.append(None)
                        error_data.append(None)
                        mecs.append(None)
                        mks.append(None)
                    else:
                        _, y_kde = stats.kde(
                            get_transform(transform)(y_values),
                            bw=bw,
                            kernel=kernel,
                            tol=tol,
                            x=x_array,
                            KDEType="fft",
                        )
                        y_hold[hi, :] = y_kde
                if agg_func is not None:
                    if y is not None:
                        y_kde, x_kde = x_array, get_transform(agg_func)(y_hold, axis=0)
                    else:
                        x_kde, y_kde = x_array, get_transform(agg_func)(y_hold, axis=0)
                    y_data.append(y_kde)
                    x_data.append(x_kde)
                    lcs.append(linecolor[u])
                    lss.append(linestyle[u])
                    facet_index.append(facet_dict[u])
                    mfcs.append(None)
                    mecs.append(None)
                    mks.append(None)
                    error_data.append(
                        get_transform(err_func)(y_hold, axis=0)
                        if err_func is not None
                        else None
                    )
    output = LinePlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        facet_index=facet_index,
        marker=mks,
        linecolor=lcs,
        linewidth=linewidth,
        linestyle=lss,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=None,
        fill_between=fill_between,
        linealpha=linealpha,
        fillalpha=fillalpha,
        fb_direction=direction,
    )
    return output


def _ecdf(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    marker: str | dict[str, str],
    markersize: float | int,
    markerfacecolor: str | dict[str, str],
    markeredgecolor: str | dict[str, str],
    linewidth: float | int,
    linecolor: str | dict[str, str],
    facet_dict: dict[str, int],
    linestyle: str | dict[str, str],
    linealpha: float | int,
    fill_between: bool = False,
    fillalpha: AlphaRange = 1.0,
    unique_id: str | None = None,
    agg_func: Agg | None = None,
    err_func=None,
    ecdf_type: Literal["spline", "bootstrap"] = "spline",
    ecdf_args=None,
    xtransform: Transform = None,
    ytransform: Transform = None,
    *args,
    **kwargs,
) -> LinePlotData:

    column = y if x is None else x
    transform = ytransform if xtransform is None else xtransform

    x_data = []
    y_data = []
    lss = []
    lcs = []
    facet_index = []
    error_data = []
    mks = []
    mfcs = []
    mecs = []

    etypes = {"spline", "bootstrap"}

    if len(levels) > 0:
        ugroups = data.groups(levels)

        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])

    for u, indexes in ugroups.items():
        if u == ("",):
            y_values = np.asarray(data[column]).flatten()
            x_ecdf, y_ecdf = stats.ecdf(
                get_transform(transform)(y_values), ecdf_type=ecdf_type, **ecdf_args
            )
            y_data.append(y_ecdf)
            x_data.append(x_ecdf)
            lcs.append(linecolor[u])
            lss.append(linestyle[u])
            facet_index.append(facet_dict[u])
            mks.append(marker[u])
            mfcs.append(markerfacecolor[u])
            mecs.append(markeredgecolor[u])
            error_data.append(None)
        elif unique_id is None:
            y_values = np.asarray(data[indexes, column]).flatten()
            x_ecdf, y_ecdf = stats.ecdf(
                get_transform(transform)(y_values), ecdf_type=ecdf_type, **ecdf_args
            )
            y_data.append(y_ecdf)
            x_data.append(x_ecdf)
            lcs.append(linecolor[u])
            lss.append(linestyle[u])
            facet_index.append(facet_dict[u])
            mks.append(marker[u])
            mfcs.append(markerfacecolor[u])
            mecs.append(markeredgecolor[u])
            error_data.append(None)
        else:
            subgroups, counts = np.unique(
                data[ugroups[u], unique_id], return_counts=True
            )
            if agg_func is not None:
                if ecdf_type not in etypes:
                    raise ValueError(
                        "ecdf_type must be spline or bootstrap when using an agg_func"
                    )
                if "size" not in ecdf_args:
                    ecdf_args["size"] = np.max(counts)
                y_ecdf = np.arange(ecdf_args["size"]) / ecdf_args["size"]
                x_hold = np.zeros((len(subgroups), ecdf_args["size"]))
            for hi, s in enumerate(subgroups):
                y_values = np.asarray(data[uid_groups[u + (s,)], column]).flatten()
                if agg_func is None:
                    x_ecdf, y_ecdf = stats.ecdf(
                        get_transform(transform)(y_values),
                        ecdf_type=ecdf_type,
                        **ecdf_args,
                    )
                    y_data.append(y_ecdf)
                    x_data.append(x_ecdf)
                    lcs.append(linecolor[u])
                    lss.append(linestyle[u])
                    facet_index.append(facet_dict[u])
                    mks.append(marker[u])
                    mfcs.append(markerfacecolor[u])
                    mecs.append(markeredgecolor[u])
                    error_data.append(None)
                else:
                    x_ecdf, _ = stats.ecdf(
                        get_transform(transform)(y_values),
                        ecdf_type=ecdf_type,
                        **ecdf_args,
                    )
                    x_hold[hi, :] = x_ecdf
            if agg_func is not None:
                x_data.append(get_transform(agg_func)(x_hold, axis=0))
                y_data.append(y_ecdf)
                lcs.append(linecolor[u])
                lss.append(linestyle[u])
                facet_index.append(facet_dict[u])
                mks.append(marker[u])
                mfcs.append(markerfacecolor[u])
                mecs.append(markeredgecolor[u])
                error_data.append(
                    get_transform(err_func)(x_hold, axis=0)
                    if err_func is not None
                    else None
                )
    output = LinePlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        facet_index=facet_index,
        marker=mks,
        linecolor=lcs,
        linewidth=linewidth,
        linestyle=lss,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=markersize,
        fill_between=fill_between,
        linealpha=linealpha,
        fillalpha=fillalpha,
        fb_direction="x",
    )
    return output


def _poly_hist(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    color_dict: dict[str, str],
    facet_dict: dict[str, int],
    linestyle_dict: dict[str, str],
    linewidth: float | int,
    unique_id: str | None = None,
    density: bool = True,
    bin_limits: list[float, float] | None = None,
    nbins: int = 50,
    func: Agg = "mean",
    err_func: Error = "sem",
    fit_func=None,
    alpha: AlphaRange = 1.0,
    xtransform: Transform = None,
    ytransform: Transform = None,
) -> LinePlotData:
    x_data = []
    y_data = []
    error_data = []
    facet_index = []
    mks = []
    lcs = []
    lss = []
    mfcs = []
    mecs = []

    y = y if x is None else x
    transform = ytransform if xtransform is None else xtransform
    transform = get_transform(transform)
    if bin_limits is None:
        bins = np.linspace(
            transform(data[y]).min(), transform(data[y]).max(), num=nbins + 1
        )
        x = np.linspace(transform(data[y]).min(), transform(data[y]).max(), num=nbins)
    else:
        x = np.linspace(bin[0], bin[1], num=nbins)
        bins = np.linspace(bin[0], bin[1], num=nbins + 1)

    if err_func is not None:
        fill_between = True

    groups = data.groups(levels)
    if unique_id is not None:
        func = get_transform(func)
        if err_func is not None:
            err_func = get_transform(err_func)
        for i, indexes in groups.items():
            uids = np.unique(data[indexes, unique_id])
            temp_list = np.zeros((len(uids), bins))
            for index, j in enumerate(uids):
                temp = np.where(data[unique_id] == j)[0]
                temp_data = np.sort(transform(data[temp, y]))
                poly, _ = np.histogram(temp_data, bins)
                if density:
                    poly = poly / poly.sum()
                if fit_func is not None:
                    poly = fit_func(x, poly)
                temp_list[index] = poly
            mean_data = func(temp_list, axis=0)
            x_data.append(x)
            y_data.append(mean_data)
            facet_index.append(facet_dict[i])
            mks.append(None)
            lcs.append(color_dict[i])
            lss.append(linestyle_dict[i])
            mfcs.append("none")
            mecs.append("none")
            if err_func is not None:
                error_data.append(err_func(temp_list, axis=0))
            else:
                error_data.append(None)
    else:
        for i, indexes in groups.items():
            temp = np.sort(transform(data[indexes, y]))
            poly, _ = np.histogram(temp, bins)
            if fit_func is not None:
                poly = fit_func(x, poly)
            if density:
                poly = poly / poly.sum()
            x_data.append(x)
            y_data.append(poly)
            facet_index.append(facet_dict[i])
            mks.append(None)
            lcs.append(color_dict[i])
            lss.append(linestyle_dict[i])
            mfcs.append("none")
            mecs.append("none")
    output = LinePlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        facet_index=facet_index,
        marker=mks,
        linecolor=lcs,
        linewidth=linewidth,
        linestyle=lss,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=None,
        fill_between=fill_between,
        linealpha=alpha,
        fillalpha=alpha,
        fb_direction="y",
    )
    return output


def _line(
    data: DataHolder,
    y: str,
    x: str,
    levels: Levels,
    color_dict: dict[str, str],
    facet_dict: dict[str, int],
    linestyle_dict: dict[str, str],
    linewidth: float | int = 2,
    unique_id: str | None = None,
    alpha: AlphaRange = 1.0,
    xtransform: Transform = None,
    ytransform: Transform = None,
    *args,
    **kwargs,
) -> LinePlotData:

    x_data = []
    y_data = []
    error_data = []
    facet_index = []
    mks = []
    lcs = []
    lss = []
    mfcs = []
    mecs = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
        for i, indexes in groups.items():
            uids = np.unique(data[indexes])
            for j in uids:
                temp = uid_groups[i + (j,)]
                temp_y = np.asarray(data[temp, y])
                temp_x = np.asarray(data[temp, x])
                x_data.append(get_transform(xtransform)(temp_x))
                y_data.append(get_transform(ytransform)(temp_y))
                facet_index.append(facet_dict[i])
                mks.append(None)
                lcs.append(color_dict[i])
                lss.append(linestyle_dict[i])
                mfcs.append("none")
                mecs.append("none")
                error_data.append(None)
    else:
        for i, indexes in groups.items():
            temp_y = np.asarray(data[indexes, y])
            temp_x = np.asarray(data[indexes, x])
            x_data.append(get_transform(xtransform)(temp_x))
            y_data.append(get_transform(ytransform)(temp_y))
            facet_index.append(facet_dict[i])
            mks.append(None)
            lcs.append(color_dict[i])
            lss.append(linestyle_dict[i])
            mfcs.append("none")
            mecs.append("none")
            error_data.append(None)
    output = LinePlotData(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        facet_index=facet_index,
        marker=mks,
        linecolor=lcs,
        linewidth=linewidth,
        linestyle=lss,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=None,
        fill_between=False,
        linealpha=alpha,
        fillalpha=alpha,
        fb_direction="y",
    )
    return output


# def _biplot(
#     data: DataHolder,
#     columns,
#     group,
#     subgroup=None,
#     group_order=None,
#     subgroup_order=None,
#     plot_pca=False,
#     plot_loadings=True,
#     marker="o",
#     color="black",
#     components=None,
#     alpha=0.8,
#     labelsize=20,
#     axis=True,
# ):
#     if components is None:
#         components = (0, 1)
#     X = preprocessing.scale(data[columns])
#     pca = decomposition.PCA(n_components=np.max(components) + 1)
#     X = pca.fit_transform(X)
#     loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

#     fig, ax = plt.subplots()

#     if plot_pca:
#         if group_order is None:
#             group_order = np.unique(data[group])
#         if subgroup is None:
#             subgroup_order = [""]
#         if subgroup_order is None:
#             subgroup_order = np.unique(data[subgroup])

#         unique_groups = []
#         for i in group_order:
#             for j in subgroup_order:
#                 unique_groups.append(i + j)
#         if subgroup is None:
#             ug_list = data[group]
#         else:
#             ug_list = data[group] + data[subgroup]

#         marker_dict = process_args(marker, group_order, subgroup_order)
#         color_dict = process_args(color, group_order, subgroup_order)

#         if components is None:
#             components = [0, 1]
#         xs = X[:, components[0]]
#         ys = X[:, components[1]]
#         scalex = 1.0 / (xs.max() - xs.min())
#         scaley = 1.0 / (ys.max() - ys.min())
#         for ug in unique_groups:
#             indexes = np.where(ug_list == ug)[0]
#             ax.scatter(
#                 xs[indexes] * scalex,
#                 ys[indexes] * scaley,
#                 alpha=alpha,
#                 marker=marker_dict[ug],
#                 c=color_dict[ug],
#             )
#         ax.legend(
#             marker: str | dict[str, str],
#         )
#     if plot_loadings:
#         width = -0.005 * np.min(
#             [np.subtract(*ax.get_xlim()), np.subtract(*ax.get_ylim())]
#         )
#         for i in range(loadings.shape[0]):
#             ax.arrow(
#                 0,
#                 0,
#                 loadings[i, 0],
#                 loadings[i, 1],
#                 color="grey",
#                 alpha=0.5,
#                 width=width,
#             )
#             ax.text(
#                 loadings[i, 0] * 1.15,
#                 loadings[i, 1] * 1.15,
#                 columns[i],
#                 color="grey",
#                 ha="center",
#                 va="center",
#             )
#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     ax.tick_params(
#         axis="both",
#         which="both",
#         bottom=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False,
#     )
#     ax.set_xlabel(
#         f"PC{components[0]} ({np.round(pca.explained_variance_ratio_[components[0]] * 100,decimals=2)}%)",
#         fontsize=labelsize,
#     )
#     ax.set_ylabel(
#         f"PC{components[1]} ({np.round(pca.explained_variance_ratio_[components[1]] * 100,decimals=2)}%)",
#         fontsize=labelsize,
#     )
#     ax.spines["top"].set_visible(axis)
#     ax.spines["right"].set_visible(axis)
#     ax.spines["left"].set_visible(axis)
#     ax.spines["bottom"].set_visible(axis)


def _percent(
    data: DataHolder,
    y: str,
    levels: Levels,
    loc_dict: dict[str, float],
    color_dict: dict[str, str],
    edgecolor_dict: dict[str, str],
    cutoff: None | float | int | list[float | int],
    include_bins: list[bool],
    barwidth: float = 1.0,
    linewidth: float | int = 1,
    alpha: AlphaRange = 1.0,
    linealpha: AlphaRange = 1.0,
    hatch: str | None = None,
    unique_id: str | None = None,
    transform: Transform = None,
    invert: bool = False,
    axis_type: BinType = "density",
    *args,
    **kwargs,
) -> RectanglePlotData:

    if cutoff is not None:
        bins = np.zeros(len(cutoff) + 2)
        bins[-1] = data[y].max() + 1e-6
        bins[0] = data[y].min() - 1e-6
        for i in range(len(cutoff)):
            bins[i + 1] = cutoff[i]

        if include_bins is None:
            include_bins = [True] * (len(bins) - 1)
    else:
        bins = np.unique(data[y])
        if include_bins is None:
            include_bins = [True] * len(bins)

    plot_bins = sum(include_bins)

    if hatch is True:
        hs = HATCHES[:plot_bins]
    else:
        hs = [None] * plot_bins

    tops = []
    bottoms = []
    edgecolors = []
    fillcolors = []
    x_loc = []
    hatches = []
    bw = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for gr, indexes in groups.items():
        if unique_id is None:
            bw.extend([barwidth] * plot_bins)
            top, bottom = _bin_data(data[indexes, y], bins, axis_type, invert, cutoff)
            tops.extend(top[include_bins])
            bottoms.extend(bottom[include_bins])
            fc = [
                color_dict[gr],
            ] * plot_bins
            fillcolors.extend(fc)
            ec = [
                edgecolor_dict[gr],
            ] * plot_bins
            edgecolors.extend(ec)
            x_s = [loc_dict[gr]] * plot_bins
            x_loc.extend(x_s)
            hatches.extend(hs)
        else:
            unique_ids_sub = np.unique(data[groups[gr], unique_id])
            temp_width = barwidth / len(unique_ids_sub)
            bw.extend([temp_width] * plot_bins * len(unique_ids_sub))
            if len(unique_ids_sub) > 1:
                dist = np.linspace(
                    -barwidth / 2, barwidth / 2, num=len(unique_ids_sub) + 1
                )
                dist = (dist[1:] + dist[:-1]) / 2
            else:
                dist = [0]
            for index, ui_group in enumerate(unique_ids_sub):
                top, bottom = _bin_data(
                    data[uid_groups[gr + (ui_group,)], y],
                    bins,
                    axis_type,
                    invert,
                    cutoff,
                )

                tops.extend(top[include_bins])
                bottoms.extend(bottom[include_bins])
                fc = [
                    color_dict[gr],
                ] * plot_bins
                fillcolors.extend(fc)
                ec = [
                    edgecolor_dict[gr],
                ] * plot_bins
                edgecolors.extend(ec)
                x_s = [loc_dict[gr] + dist[index]] * plot_bins
                x_loc.extend(x_s)
                hatches.extend(hs)
    output = RectanglePlotData(
        tops=tops,
        bottoms=bottoms,
        bins=x_loc,
        binwidths=bw,
        fillcolors=fillcolors,
        edgecolors=edgecolors,
        fill_alpha=alpha,
        edge_alpha=linealpha,
        hatches=hatches,
        linewidth=linewidth,
    )
    return output


def _count(
    data: DataHolder,
    y: str,
    levels: list[str],
    loc_dict: dict,
    color_dict: dict[str, str],
    edgecolor_dict: dict[str, str],
    hatch: str,
    barwidth: float,
    linewidth: float | int,
    alpha: float | int,
    edge_alpha: float | int,
    axis_type: CountPlotTypes,
    unique_id: str | None = None,
    invert: bool = False,
    agg_func: Agg | None = None,
    err_func: Error = None,
    transform: Transform = None,
    *args,
    **kwargs,
) -> RectanglePlotData:

    bw = []
    bottoms = []
    tops = []
    fillcolors = []
    edgecolors = []
    x_loc = []
    hatches = []

    multiplier = 100 if axis_type == "percent" else 1

    groups = data.groups(levels)
    for gr, indexes in groups.items():
        unique_groups_sub, counts = np.unique(data[indexes, y], return_counts=True)
        size = sum(counts)
        temp_width = barwidth / len(unique_groups_sub)
        if len(unique_groups_sub) > 1:
            dist = np.linspace(
                -barwidth / 2, barwidth / 2, num=len(unique_groups_sub) + 1
            )
            dist = (dist[1:] + dist[:-1]) / 2
        else:
            dist = [0]
        bw.extend([temp_width] * len(unique_groups_sub))
        for index, ui_group, count in enumerate(zip(unique_groups_sub, counts)):
            if unique_id is None:
                bottoms.append(0)
                tops.append(
                    (count / size if axis_type != "count" else count) * multiplier
                )
                fillcolors.append(color_dict[str(ui_group)])
                edgecolors.append(edgecolor_dict[str(ui_group)])
                x_loc.append(loc_dict[gr] + dist[index])
                hatches.append(HATCHES[index] if hatch else None)
            else:
                pass
    output = RectanglePlotData(
        tops=tops,
        bottoms=bottoms,
        bins=x_loc,
        binwidths=bw,
        fillcolors=fillcolors,
        edgecolors=edgecolors,
        fill_alpha=alpha,
        edge_alpha=edge_alpha,
        hatches=hatches,
        linewidth=linewidth,
        axis="x",
    )
    return output
