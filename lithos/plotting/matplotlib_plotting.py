# TODO: Need to remove all matplotlib functions from this file.

from itertools import cycle
from typing import Literal, TypedDict, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# import networkx as nx
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.colors import to_rgba

# from matplotlib.container import BarContainer
from numpy.random import default_rng

# from sklearn import decomposition, preprocessing

from ..stats import ecdf, kde
from ..utils import DataHolder, get_transform
from .plot_utils import _bin_data, process_duplicates, process_jitter

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
CB6 = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]
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


def _make_legend_patches(color_dict, alpha, group, subgroup):
    legend_patches = []
    # for j in group:
    #     if j in color_dict:
    #         legend_patches.append(
    #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
    #         )
    # for j in subgroup:
    #     if j in color_dict:
    #         legend_patches.append(
    #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
    #         )
    for key, value in color_dict.items():
        legend_patches.append(
            mpatches.Patch(color=to_rgba(value, alpha=alpha), label=key)
        )
    return legend_patches


def _add_rectangles(
    tops,
    bottoms,
    bins,
    binwidths,
    fillcolors,
    edgecolors,
    fill_alpha,
    edge_alpha,
    hatches,
    linewidth,
    ax,
    axis="x",
):
    for t, b, b, bw, fc, ec, ht, ln, sub_ax in zip(
        tops,
        bottoms,
        bins,
        binwidths,
        fillcolors,
        edgecolors,
        hatches,
        linewidth,
        ax,
    ):
        if axis == "x":
            sub_ax.bar(
                x=bins,
                height=t,
                bottom=b,
                width=bw,
                color=to_rgba(fc, alpha=fill_alpha),
                edgecolor=to_rgba(ec, edge_alpha),
                linewidth=linewidth,
                hatch=ht,
            )
        else:
            sub_ax.barh(
                y=bins,
                width=t,
                left=b,
                height=bw,
                color=to_rgba(fc, alpha=fill_alpha),
                edgecolor=to_rgba(ec, edge_alpha),
                linewidth=linewidth,
                hatch=hatches,
            )
    return ax


def _plot_scatter(
    x_data,
    y_data,
    marker,
    markerfacecolor,
    markeredgecolor,
    markersize,
    alpha,
    edge_alpha,
    ax,
):
    for x, y, mk, mf, me, ms in zip(
        x_data, y_data, marker, markerfacecolor, markeredgecolor, markersize
    ):
        ax.plot(
            x,
            y,
            mk,
            markerfacecolor=to_rgba(mf, alpha=alpha) if mf != "none" else "none",
            markeredgecolor=to_rgba(me, alpha=edge_alpha) if me != "none" else "none",
            markersize=ms,
        )
    return ax


def _jitter_plot(
    data,
    y,
    levels,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    edge_alpha=1,
    seed=42,
    markersize=2,
    transform=None,
    ax=None,
    unique_id=None,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    rng = default_rng(seed)

    x_data = []
    y_data = []
    mks = []
    mfcs = []
    mecs = []
    mksizes = []

    if len(levels) == 0:
        pass
    else:
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

    _plot_scatter(
        x_data=x_data,
        y_data=y_data,
        marker=mks,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=mksizes,
        alpha=alpha,
        edge_alpha=edge_alpha,
        ax=ax,
    )
    return ax


def _jitteru_plot(
    data,
    y,
    levels,
    unique_id,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    edge_alpha=1,
    duplicate_offset=0.0,
    markersize=2,
    agg_func=None,
    transform=None,
    ax=None,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    temp = width / 2

    x_data = []
    y_data = []
    mks = []
    mfcs = []
    mecs = []
    mksizes = []

    if len(levels) == 0:
        pass
    else:
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
                sub_indexes = uid_groups[i + (ui_group,)]
                x = np.full(len(sub_indexes), loc_dict[i]) + dist[index]
                if duplicate_offset > 0.0:
                    output = (
                        process_duplicates(data[sub_indexes, y])
                        * duplicate_offset
                        * temp
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
    _plot_scatter(
        x_data=x_data,
        y_data=y_data,
        marker=mks,
        markerfacecolor=mfcs,
        markeredgecolor=mecs,
        markersize=mksizes,
        alpha=alpha,
        edge_alpha=edge_alpha,
        ax=ax,
    )
    return ax


def _summary_plot(
    data,
    y,
    levels,
    unique_groups,
    loc_dict,
    func,
    capsize,
    capstyle,
    barwidth,
    err_func,
    linewidth,
    color_dict,
    alpha,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    y_data = []
    error_data = []
    colors = []
    x_data = []
    widths = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for i in unique_groups:
            x_data.append(loc_dict[i])
            colors.append(color_dict[i])
            widths.append(barwidth)
            y_data.append(get_transform(func)(transform(data[groups[i], y])))
            if err_func is not None:
                error_data.append(
                    get_transform(err_func)(transform(data[groups[i], y]))
                )
            else:
                error_data.append(None)
        ax = _plot_summary(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            widths=widths,
            colors=colors,
            linewidth=linewidth,
            alpha=alpha,
            capstyle=capstyle,
            capsize=capsize,
            ax=ax,
        )
    return ax


def _summaryu_plot(
    data,
    y,
    levels,
    unique_groups,
    unique_id,
    loc_dict,
    func,
    capsize,
    capstyle,
    barwidth,
    err_func,
    linewidth,
    color_dict,
    alpha,
    agg_func=None,
    agg_width=1,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    y_data = []
    error_data = []
    colors = []
    x_data = []
    widths = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for i in unique_groups:
            uids = np.unique(data[groups[i], unique_id])
            if agg_func is None:
                temp = barwidth / 2
                if len(uids) > 1:
                    dist = np.linspace(-temp, temp, num=len(uids) + 1)
                    centers = np.round((dist[1:] + dist[:-1]) / 2, 10)
                else:
                    centers = [0]
                w = agg_width / len(uids)
                for index, j in enumerate(uids):
                    widths.append(w)
                    vals = transform(data[uid_groups[i + (j,)], y])
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

    ax = _plot_summary(
        x_data=x_data,
        y_data=y_data,
        error_data=error_data,
        widths=widths,
        colors=colors,
        linewidth=linewidth,
        alpha=alpha,
        capstyle=capstyle,
        capsize=capsize,
        ax=ax,
    )
    return ax


def _plot_summary(
    x_data: list,
    y_data: list,
    error_data: list,
    widths: list,
    colors: list,
    linewidth: float,
    alpha: float,
    capstyle: str,
    capsize: float,
    ax,
):
    for xd, yd, e, c, w in zip(x_data, y_data, error_data, colors, widths):
        _, caplines, bars = ax.errorbar(
            x=xd,
            y=yd,
            yerr=e,
            c=to_rgba(c, alpha=alpha),
            fmt="none",
            linewidth=linewidth,
            capsize=capsize,
        )
        for cap in caplines:
            cap.set_solid_capstyle(capstyle)
            cap.set_markeredgewidth(linewidth)
            cap._marker._capstyle = CapStyle(capstyle)
        for b in bars:
            b.set_capstyle(capstyle)
        _, caplines, bars = ax.errorbar(
            y=yd,
            x=xd,
            xerr=w / 2,
            c=to_rgba(c, alpha=alpha),
            fmt="none",
            linewidth=linewidth,
            capsize=0,
        )
        for b in bars:
            b.set_capstyle(capstyle)
    return ax


def _plot_boxplot(
    x_data,
    y_data,
    facecolors,
    edgecolors,
    alpha,
    line_alpha,
    fliers,
    linewidth,
    width,
    show_ci,
    showmeans,
    ax,
):
    for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
        props = {
            "boxprops": {
                "facecolor": to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs,
                "edgecolor": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs,
            },
            "medianprops": {
                "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
            },
            "whiskerprops": {
                "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
            },
            "capprops": {
                "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
            },
        }
        if showmeans:
            props["meanprops"] = {
                "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
            }
        bplot = ax.boxplot(
            y,
            positions=x,
            sym=fliers,
            widths=width,
            notch=show_ci,
            patch_artist=True,
            showmeans=showmeans,
            meanline=showmeans,
            **props,
        )
        for i in bplot["boxes"]:
            i.set_linewidth(linewidth)


def _boxplot(
    data,
    y,
    levels,
    loc_dict,
    color_dict,
    edgecolor_dict,
    fliers="",
    width: float = 1.0,
    linewidth=1,
    showmeans: bool = False,
    show_ci: bool = False,
    alpha: float = 1.0,
    line_alpha=1.0,
    transform=None,
    ax=None,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    y_data = []
    x_data = []
    fcs = []
    ecs = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for key, value in groups.items():
            y_data.append(data[value, y])
            x_data.append([loc_dict[key]])
            fcs.append(color_dict[key])
            ecs.append(edgecolor_dict[key])
    _plot_boxplot(
        x_data=x_data,
        y_data=y_data,
        facecolors=fcs,
        edgecolors=ecs,
        alpha=alpha,
        line_alpha=line_alpha,
        fliers=fliers,
        linewidth=linewidth,
        width=width,
        show_ci=show_ci,
        showmeans=showmeans,
        ax=ax,
    )
    return ax


def _plot_violin(
    x_data,
    y_data,
    facecolors,
    edgecolors,
    alpha,
    edge_alpha,
    linewidth,
    width,
    showmeans,
    showmedians,
    showextrema,
    ax,
):
    for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
        parts = ax.violinplot(
            y,
            positions=x,
            widths=width,
            showmeans=showmeans,
            showmedians=showmedians,
            showextrema=showextrema,
        )
        for body in parts["bodies"]:
            body.set_facecolor(to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs)
            body.set_edgecolor(to_rgba(ecs, alpha=edge_alpha) if ecs != "none" else ecs)
            body.set_linewidth(linewidth)
        if showmeans:
            # Matplotlib seems to divide the alpha by 2 so this needs to be equivalent
            parts["cmeans"].set_color(
                to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
            )
            parts["cmeans"].set_linewidth(linewidth)
        if showmedians:
            parts["cmedians"].set_color(
                to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
            )
            parts["cmedians"].set_linewidth(linewidth)


def _violin_plot(
    data,
    y,
    levels,
    loc_dict,
    facecolor_dict,
    edgecolor_dict,
    alpha=1,
    edge_alpha=1,
    linewidth=1,
    showextrema: bool = False,
    width: float = 1.0,
    showmeans: bool = False,
    showmedians: bool = False,
    transform=None,
    ax=None,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    x_data = []
    y_data = []
    fcs = []
    ecs = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for key, value in groups.items():
            x_data.append([loc_dict[key]])
            y_data.append(transform(data[value, y]))
            fcs.append(facecolor_dict[key])
            ecs.append(edgecolor_dict[key])
    _plot_violin(
        x_data=x_data,
        y_data=y_data,
        facecolors=fcs,
        edgecolors=ecs,
        alpha=alpha,
        edge_alpha=edge_alpha,
        linewidth=linewidth,
        width=width,
        showmeans=showmeans,
        showmedians=showmedians,
        showextrema=showextrema,
        ax=ax,
    )
    return ax


def paired_plot():
    pass


def _calc_hist(data, bins, stat):
    if stat == "probability":
        data, _ = np.histogram(data, bins)
        return data / data.sum()
    elif stat == "count":
        data, _ = np.histogram(data, bins)
        return data
    elif stat == "density":
        data, _ = np.histogram(data, bins, density=True)
        return data


def _hist_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    hatch=None,
    hist_type: Literal["bar", "step", "stepfilled"] = "bar",
    fillalpha=1.0,
    linealpha=1.0,
    bin_limits=None,
    nbins=None,
    stat="probability",
    ax=None,
    agg_func=None,
    projection="rectilinear",
    unique_id=None,
    ytransform=None,
    xtransfrom=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    y = y if x is None else x
    transform = ytransform if xtransfrom is None else xtransfrom
    axis = "y" if x is None else "x"

    if bin_limits is None:
        bins = np.linspace(
            get_transform(transform)(data[y].min()),
            get_transform(transform)(data[y].max()),
            num=nbins + 1,
        )
        # x = np.linspace(data[y].min(), data[y].max(), num=nbins)
        x = (bins[1:] + bins[:-1]) / 2
    else:
        # x = np.linspace(bin_limits[0], bin_limits[1], num=nbins)
        bins = np.linspace(
            get_transform(transform)(bin_limits[0]),
            get_transform(transform)(bin_limits[1]),
            num=nbins + 1,
        )
        x = (bins[1:] + bins[:-1]) / 2
    bottom = np.zeros(nbins)
    bw = np.full(
        nbins,
        bins[1] - bins[0],
    )
    plot_data = []
    count = 0
    colors = []
    axes1 = []
    edgec = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            unique_id_indexes = data.groups(levels + [unique_id])
        for i in unique_groups:
            if unique_id is not None:
                uids = np.unique(data[unique_groups == i, unique_id])
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
                        axes1.append(ax[facet_dict[i]])
                        count += 1
                if agg_func is not None:
                    plot_data.append(get_transform(agg_func)(temp_list, axis=0))
                    colors.append([color_dict[i]] * nbins)
                    edgec.append([color_dict[i]] * nbins)
                    axes1.append(ax[facet_dict[i]])
                    count += 1
            else:
                temp_data = np.sort(data[groups[i], y])
                poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                plot_data.append(poly)
                colors.append([color_dict[i]] * nbins)
                edgec.append([color_dict[i]] * nbins)
                axes1.append(ax[facet_dict[i]])
                count += 1
    bottom = [bottom for _ in range(count)]
    bins = [bins[:-1] for _ in range(count)]
    bw = [bw for _ in range(count)]
    hatches = [[hatch] * nbins] * count
    linewidth = [np.full(nbins, 0) for _ in range(count)]
    _add_rectangles(
        tops=plot_data,
        bottoms=bottom,
        bins=bins,
        binwidths=bw,
        fillcolors=colors,
        edgecolors=edgec,
        fill_alpha=fillalpha,
        edge_alpha=linealpha,
        hatches=hatches,
        linewidth=linewidth,
        ax=axes1,
        axis=axis,
    )
    return ax


def _scatter_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    markers,
    markercolors,
    edgecolors,
    markersizes,
    facetgroup,
    ax=None,
    facet_dict=None,
    xtransform=None,
    ytransform=None,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    for key, value in facet_dict.items():
        indexes = np.array([index for index, j in enumerate(facetgroup) if value == j])
        ax[value].scatter(
            get_transform(xtransform)(data[indexes, x]),
            get_transform(ytransform)(data[indexes, y]),
            marker=markers,
            color=[markercolors[i] for i in indexes],
            edgecolors=[edgecolors[i] for i in indexes],
            s=[markersizes[i] for i in indexes],
        )
    return ax


def _plot_error_line(
    x_data,
    y_data,
    error_data,
    facet_index,
    marker=None,
    linecolor=None,
    linewidth=None,
    linestyle=None,
    markerfacecolor=None,
    markeredgecolor=None,
    fill_between=False,
    fb_direction="y",
    markersize=None,
    fillalpha=None,
    linealpha=None,
    ax=None,
):
    for x, y, err, ls, lc, mf, me, mk, fc in zip(
        x_data,
        y_data,
        error_data,
        linestyle,
        linecolor,
        markerfacecolor,
        markeredgecolor,
        marker,
        facet_index,
    ):
        if not fill_between:
            if fb_direction == "x":
                ax[fc].errorbar(
                    x,
                    y,
                    xerr=err,
                    marker=mk,
                    color=lc,
                    elinewidth=linewidth,
                    linewidth=linewidth,
                    linestyle=ls,
                    markerfacecolor=mf,
                    markeredgecolor=me,
                    markersize=markersize,
                    alpha=linealpha,
                )
            else:
                ax[fc].errorbar(
                    x,
                    y,
                    yerr=err,
                    marker=mk,
                    color=lc,
                    elinewidth=linewidth,
                    linewidth=linewidth,
                    linestyle=ls,
                    markerfacecolor=mf,
                    markeredgecolor=me,
                    markersize=markersize,
                    alpha=linealpha,
                )
        else:
            if err is not None:
                if fb_direction == "y":
                    ax[fc].fill_between(
                        x,
                        y - err,
                        y + err,
                        color=to_rgba(lc, alpha=fillalpha),
                        linewidth=0,
                        edgecolor="none",
                    )
                else:
                    ax[fc].fill_betweenx(
                        y,
                        x - err,
                        x + err,
                        color=to_rgba(lc, alpha=fillalpha),
                        linewidth=0,
                        edgecolor="none",
                    )
            ax[fc].plot(
                x,
                y,
                linestyle=ls,
                linewidth=linewidth,
                color=lc,
                alpha=linealpha,
            )


def _agg_line(
    data,
    x,
    y,
    levels,
    marker,
    markersize,
    markerfacecolor,
    markeredgecolor,
    linestyle,
    linewidth,
    linecolor,
    linealpha,
    func,
    err_func,
    facet_dict,
    fill_between=False,
    fillalpha=1.0,
    agg_func=None,
    ytransform=None,
    xtransform=None,
    unique_id=None,
    ax=None,
    sort=True,
    *args,
    **kwargs,
):
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
    _plot_error_line(
        x_data,
        y_data,
        error_data,
        facet_index,
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
        ax=ax,
        fb_direction="y",
    )

    return ax


class LinePlotter(TypedDict):
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    error_data: list[np.ndarray | None]
    facet_index: list[int]
    marker: list[str]
    linecolor: list[str]
    linewidth: list[float]
    linestyle: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    fill_between: bool
    linealpha: float
    fillalpha: float


def _kde_plot(
    data,
    y,
    x,
    levels,
    linecolor,
    facet_dict,
    linestyle,
    linewidth,
    linealpha,
    fillalpha,
    fill_between,
    kernel: Literal[
        "gaussian",
        "exponential",
        "box",
        "tri",
        "epa",
        "biweight",
        "triweight",
        "tricube",
        "cosine",
    ] = "gaussian",
    bw: Literal["ISJ", "silverman", "scott"] = "ISJ",
    tol: Union[float, int] = 1e-3,
    common_norm: bool = True,
    unique_id=None,
    ax=None,
    agg_func=None,
    err_func=None,
    xtransform=None,
    ytransform=None,
    KDEType="fft",
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]
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
        x_kde, y_kde = kde(
            get_transform(transform)(y_values), bw=bw, kernel=kernel, tol=tol
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
                x_kde, y_kde = kde(
                    get_transform(transform)(y_values), bw=bw, kernel=kernel, tol=tol
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
                        power2 = int(np.ceil(np.log2(len(temp_data))))
                        x_array = np.linspace(min_data, max_data, num=(1 << power2))
                    else:
                        max_len = np.max(count)
                        x_array = np.linspace(
                            min_data, max_data, num=int(max_len * 1.5)
                        )
                    y_hold = np.zeros((len(subgroups), x_array.size))

                for hi, s in enumerate(subgroups):
                    s_indexes = uid_groups[u + (s,)]
                    y_values = np.asarray(data[s_indexes, column]).flatten()
                    temp_size = y_values.size
                    if agg_func is None:
                        x_kde, y_kde = kde(
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
                        _, y_kde = kde(
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
    _plot_error_line(
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
        ax=ax,
        fb_direction=direction,
    )
    return ax


def _ecdf(
    data,
    y,
    x,
    levels,
    marker,
    markersize,
    markerfacecolor,
    markeredgecolor,
    linewidth,
    linecolor,
    facet_dict,
    linestyle,
    linealpha,
    fill_between=False,
    fillalpha=1.0,
    unique_id=None,
    agg_func=None,
    err_func=None,
    ecdf_type: Literal["spline", "bootstrap"] = "spline",
    ecdf_args=None,
    ax=None,
    xtransform=None,
    ytransform=None,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]

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
            x_ecdf, y_ecdf = ecdf(
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
            x_ecdf, y_ecdf = ecdf(
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
                    x_ecdf, y_ecdf = ecdf(
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
                    x_ecdf, _ = ecdf(
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
    _plot_error_line(
        x_data,
        y_data,
        error_data,
        facet_index,
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
        ax=ax,
        fb_direction="x",
    )
    return ax


def _poly_hist(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    linestyle_dict,
    linewidth,
    unique_id=None,
    density=True,
    bin_limits=None,
    nbins=50,
    func="mean",
    err_func="sem",
    fit_func=None,
    alpha=1,
    ax=None,
    xtransform=None,
    ytransform=None,
):
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
    if ax is None:
        ax = plt.gca()
        ax = [ax]

    if unique_id is not None:
        func = get_transform(func)
        if err_func is not None:
            err_func = get_transform(err_func)
        ugrp = np.unique(unique_groups)
        for i in ugrp:
            indexes = np.where(unique_groups == i)[0]
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
            ax[facet_dict[i]].plot(
                x,
                mean_data,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
                alpha=alpha,
                linewidth=linewidth,
            )
            if err_func is not None:
                ax[facet_dict[i]].fill_between(
                    x=x,
                    y1=mean_data - err_func(temp_list, axis=0),
                    y2=mean_data + err_func(temp_list, axis=0),
                    alpha=alpha / 2,
                    color=color_dict[i],
                    edgecolor="none",
                )
    else:
        ugrp = np.unique(unique_groups)
        for i in ugrp:
            indexes = np.where(unique_groups == i)[0]
            temp = np.sort(transform(data[indexes, y]))
            poly, _ = np.histogram(temp, bins)
            if fit_func is not None:
                poly = fit_func(x, poly)
            if density:
                poly = poly / poly.sum()
            ax[facet_dict[i]].plot(
                x,
                poly,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
            )
    return ax


def _line_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    linestyle_dict,
    linewidth=2,
    unique_id=None,
    func="mean",
    err_func="sem",
    fit_func=None,
    alpha=1,
    ax=None,
    xtransform=None,
    ytransform=None,
):
    if ax is None:
        ax = plt.gca()
    if ax is None:
        ax = plt.gca()
        ax = [ax]

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
        func = get_transform(func)
        if err_func is not None:
            err_func = get_transform(err_func)
        for i in unique_groups:
            indexes = groups[i]
            uids = np.unique(data[indexes])
            temp_list_y = None
            temp_list_x = None
            for index, j in enumerate(uids):
                temp = uid_groups[i + (j,)]
                temp_y = np.asarray(data[temp, y])
                temp_x = np.asarray(data[temp, x])
                if temp_list_y is None:
                    temp_list_y = np.zeros((len(uids), temp_x.size))
                if temp_list_x is None:
                    temp_list_x = np.zeros((len(uids), temp_x.size))
                if fit_func is not None:
                    poly = fit_func(temp_x, temp_y)
                    temp_list_y[index] = poly
                else:
                    temp_list_y[index] = temp_y
                temp_list_x[index] = temp_x
            mean_x = np.nanmean(temp_list_x, axis=0)
            mean_y = func(temp_list_y, axis=0)
            ax[facet_dict[i]].plot(
                mean_x,
                mean_y,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
                linewidth=linewidth,
                alpha=alpha,
            )
            if err_func is not None:
                mean_y = func(temp_list_y)
                err = err_func(temp_list_y, axis=0)
                ax[facet_dict[i]].fill_between(
                    x=mean_x,
                    y1=mean_y - err,
                    y2=mean_y + err,
                    alpha=alpha / 2,
                    color=color_dict[i],
                    edgecolor="none",
                )
    else:
        for i in unique_groups:
            indexes = groups[i]
            temp_y = np.asarray(data[indexes, y])
            temp_x = np.asarray(data[indexes, x])
            if fit_func is not None:
                temp_y = fit_func(temp_x, temp_y)
            ax[facet_dict[i]].plot(
                temp_x, temp_y, c=color_dict[i], linestyle=linestyle_dict[i]
            )
    return ax


# def biplot(
#     data,
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
#             marker,
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


def _percent_plot(
    data: DataHolder,
    y,
    levels,
    unique_groups,
    loc_dict,
    color_dict,
    edgecolor_dict,
    cutoff: Union[None, float, int, list[Union[float, int]]],
    include_bins: list[bool],
    barwidth: float = 1.0,
    linewidth=1,
    alpha: float = 1.0,
    line_alpha=1.0,
    hatch=None,
    unique_id=None,
    ax=None,
    transform=None,
    invert=False,
    axis_type="density",
):
    if ax is None:
        ax = plt.gca()

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
    lw = []
    edgecolors = []
    fillcolors = []
    x_loc = []
    hatches = []
    bw = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for gr in unique_groups:
        if unique_id is None:
            bw.extend([barwidth] * plot_bins)
            lw.extend([linewidth] * plot_bins)
            top, bottom = _bin_data(
                data[groups[gr], y], bins, axis_type, invert, cutoff
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
            x_s = [loc_dict[gr]] * plot_bins
            x_loc.extend(x_s)
            hatches.extend(hs)
        else:
            unique_ids_sub = np.unique(data[groups[gr], unique_id])
            temp_width = barwidth / len(unique_ids_sub)
            bw.extend([temp_width] * plot_bins * len(unique_ids_sub))
            lw.extend([linewidth] * plot_bins * len(unique_ids_sub))
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
    ax = _add_rectangles(
        tops=tops,
        bottoms=bottoms,
        bins=x_loc,
        binwidths=bw,
        fillcolors=fillcolors,
        edgecolors=edgecolors,
        fill_alpha=alpha,
        edge_alpha=line_alpha,
        hatches=hatches,
        linewidth=lw,
        ax=ax,
    )
    return ax


def _count_plot(
    data: DataHolder,
    y: str,
    levels: list[str],
    unique_groups: dict,
    loc_dict: dict,
    color_dict,
    edgecolor_dict,
    hatch,
    barwidth,
    linewidth,
    alpha,
    line_alpha,
    axis_type,
    unique_id=None,
    invert=False,
    agg_func=None,
    err_func=None,
    ax=None,
    transform=None,
):

    bw = []
    bottoms = []
    tops = []
    fillcolors = []
    edgecolors = []
    x_loc = []
    hatches = []
    lws = []

    multiplier = 100 if axis_type == "percent" else 1

    groups = data.groups(levels)
    for gr in unique_groups:
        unique_groups_sub, counts = np.unique(data[groups[gr], y], return_counts=True)
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
                lws.append(linewidth)
            else:
                pass

    ax = _add_rectangles(
        tops=tops,
        bottoms=bottoms,
        bins=x_loc,
        binwidths=bw,
        fillcolors=fillcolors,
        edgecolors=edgecolors,
        fill_alpha=alpha,
        edge_alpha=alpha,
        hatches=hatches,
        linewidth=lws,
        ax=ax,
    )
    ax.autoscale()
    return ax


# def _plot_network(
#     graph,
#     marker_alpha: float = 0.8,
#     line_alpha: float = 0.1,
#     markersize: int = 2,
#     marker_scale: int = 1,
#     linewidth: int = 1,
#     edge_color: str = "k",
#     marker_color: str = "red",
#     marker_attr: Optional[str] = None,
#     cmap: str = "gray",
#     seed: int = 42,
#     scale: int = 50,
#     plot_max_degree: bool = False,
#     layout: Literal["spring", "circular", "communities"] = "spring",
# ):

#     if isinstance(cmap, str):
#         cmap = plt.colormaps[cmap]
#     _, ax = plt.subplots()
#     Gcc = graph.subgraph(
#         sorted(nx.connected_components(graph), key=len, reverse=True)[0]
#     )
#     if layout == "spring":
#         pos = nx.spring_layout(Gcc, seed=seed, scale=scale)
#     elif layout == "circular":
#         pos = nx.circular_layout(Gcc, scale=scale)
#     elif layout == "random":
#         pos = nx.random_layout(Gcc, seed=seed)
#     elif layout == "communities":
#         communities = nx.community.greedy_modularity_communities(Gcc)
#         # Compute positions for the node clusters as if they were themselves nodes in a
#         # supergraph using a larger scale factor
#         _ = nx.cycle_graph(len(communities))
#         superpos = nx.spring_layout(Gcc, scale=scale, seed=seed)

#         # Use the "supernode" positions as the center of each node cluster
#         centers = list(superpos.values())
#         pos = {}
#         for center, comm in zip(centers, communities):
#             pos.update(
#                 nx.spring_layout(nx.subgraph(Gcc, comm), center=center, seed=seed)
#             )

#     nodelist = list(Gcc)
#     markersize = np.array([Gcc.degree(i) for i in nodelist])
#     markersize = markersize * marker_scale
#     xy = np.asarray([pos[v] for v in nodelist])

#     edgelist = list(Gcc.edges(data=True))
#     edge_pos = np.asarray([(pos[e0], pos[e1]) for (e0, e1, _) in edgelist])
#     _, _, data = edgelist[0]
#     if edge_color in data:
#         edge_color = [data["weight"] for (_, _, data) in edgelist]
#         edge_vmin = min(edge_color)
#         edge_vmax = max(edge_color)
#         color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
#         edge_color = [cmap(color_normal(e)) for e in edge_color]
#     edge_collection = LineCollection(
#         edge_pos,
#         colors=edge_color,
#         linewidths=linewidth,
#         antialiaseds=(1,),
#         linestyle="solid",
#         alpha=line_alpha,
#     )
#     edge_collection.set_cmap(cmap)
#     edge_collection.set_clim(edge_vmin, edge_vmax)
#     edge_collection.set_zorder(0)  # edges go behind nodes
#     edge_collection.set_label("edges")
#     ax.add_collection(edge_collection)

#     if isinstance(marker_color, dict):
#         if marker_attr is not None:
#             mcolor = [
#                 marker_color[data[marker_attr]] for (_, data) in Gcc.nodes(data=True)
#             ]
#         else:
#             mcolor = "red"
#     else:
#         mcolor = marker_color

#     path_collection = ax.scatter(
#         xy[:, 0], xy[:, 1], s=markersize, alpha=marker_alpha, c=mcolor
#     )
#     path_collection.set_zorder(1)
#     ax.axis("off")
