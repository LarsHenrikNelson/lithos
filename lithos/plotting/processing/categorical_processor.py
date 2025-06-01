# TODO: Need to remove all matplotlib functions from this file.

from itertools import cycle
from typing import Literal

import numpy as np
from numpy.random import default_rng

from ... import stats
from ...utils import DataHolder, get_transform
from ..plot_utils import (
    _bin_data,
    process_duplicates,
    process_jitter,
    _process_positions,
    create_dict,
    _create_groupings,
)
from ..types import (
    BW,
    Agg,
    AlphaRange,
    BinType,
    BoxPlotData,
    Error,
    Kernels,
    Levels,
    RectanglePlotData,
    JitterPlotData,
    SummaryPlotData,
    Transform,
    ViolinPlotData,
    CapStyle,
)

from .base_processor import BaseProcessor


class CategoricalProcessor(BaseProcessor):
    def __init__(self, markers, hatches):
        super().__init__(markers, hatches)
        self.PLOTS = {
            "box": self._box,
            "jitter": self._jitter,
            "jitteru": self._jitteru,
            "summary": self._summary,
            "summaryu": self._summaryu,
            "violin": self._violin,
            "percent": self._percent,
            "bar": self._bar,
        }

    def process_groups(
        self, data, group, subgroup, group_order, subgroup_order, group_spacing
    ):
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            data, group, subgroup, group_order, subgroup_order
        )
        if group is not None:
            loc_dict, width = _process_positions(
                group_order=group_order,
                subgroup_order=subgroup_order,
                group_spacing=group_spacing,
            )
        else:
            group_order = [""]
            subgroup_order = [""]
            loc_dict = {("",): 0.0}
            width = 1.0

        zgroup = group_order if subgroup_order is None else subgroup_order
        zorder_dict = create_dict(zgroup, unique_groups)

        x_ticks = [index for index, _ in enumerate(group_order)]

        self._plot_dict = {
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "loc_dict": loc_dict,
            "levels": levels,
            "zorder_dict": zorder_dict,
            "ticks": x_ticks,
            "width": width,
        }

    def _jitter(
        self,
        data: DataHolder,
        y: str,
        levels: list[str | int],
        loc_dict: dict[str, float],
        width: float,
        markercolor: dict[str, str],
        marker: dict[str, str],
        edgecolor: dict[str, str],
        zorder_dict: dict[str, int],
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
        seed: int = 42,
        markersize: float | int = 2,
        x: str = None,
        ytransform: Transform = None,
        unique_id: str | None = None,
        *args,
        **kwargs,
    ) -> JitterPlotData:
        transform = get_transform(ytransform)

        rng = default_rng(seed)
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        x_data = []
        y_data = []
        markers = []
        group_labels = []

        groups = data.groups(levels)
        unique_groups = None

        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))

        jitter_values = np.zeros(data.shape[0])

        for group_key, indexes in groups.items():
            temp_jitter = process_jitter(
                data[indexes, column], loc_dict[group_key], width, rng=rng
            )
            jitter_values[indexes] = temp_jitter

        for group_key, indexes in groups.items():
            if unique_id is None:
                x_data.append(jitter_values[indexes])
                y_data.append(transform(data[indexes, column]))
                group_labels.append(group_key)
                markers.append(marker[group_key])
            else:
                subgroups = np.unique(data[indexes, unique_id])
                for ui_group, mrk in zip(subgroups, cycle(self.MARKERS)):
                    sub_indexes = unique_groups[group_key + (ui_group,)]
                    x_data.append(jitter_values[sub_indexes])
                    y_data.append(transform(data[sub_indexes, column]))
                    group_labels.append(group_key)
                    markers.append(mrk)

        output = JitterPlotData(
            x_data=x_data,
            y_data=y_data,
            marker=markers,
            markerfacecolor=self._process_dict(groups, markercolor, unique_groups),
            markeredgecolor=self._process_dict(groups, edgecolor, unique_groups),
            markersize=[markersize] * len(y_data),
            alpha=alpha,
            edge_alpha=edge_alpha,
            group_labels=group_labels,
            direction=direction,
            zorder=self._process_dict(groups, zorder_dict, unique_groups),
        )
        return output

    def _jitteru(
        self,
        data: DataHolder,
        y: str,
        levels: Levels,
        unique_id: str,
        loc_dict: dict[str, float],
        width: float,
        markercolor: dict[str, str],
        marker: str,
        edgecolor: dict[str, str],
        zorder_dict: dict[str, int],
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
        duplicate_offset: float = 0.0,
        markersize: int = 2,
        x: str = None,
        agg_func: Agg | None = None,
        ytransform: Transform = None,
        *args,
        **kwargs,
    ) -> JitterPlotData:
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        transform = get_transform(ytransform)
        temp = width / 2

        x_data = []
        y_data = []
        group_labels = []
        unique_groups = None

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))

        for group_key in groups.keys():
            unique_ids_sub = np.unique(data[groups[group_key], unique_id])
            if len(unique_ids_sub) > 1:
                left = loc_dict[group_key] - width / 2
                right = loc_dict[group_key] + width / 2
                vals = len(unique_ids_sub) * 2 + 1
                dist = np.linspace(left, right, num=vals)
                dist = dist[1::2]
            else:
                dist = [0]
            for index, ui_group in enumerate(unique_ids_sub):
                if group_key == ("",):
                    temp_group = (ui_group,)
                else:
                    temp_group = group_key + (ui_group,)
                sub_indexes = unique_groups[temp_group]
                x = np.full(len(sub_indexes), dist[index])
                if duplicate_offset > 0.0:
                    output = (
                        process_duplicates(data[sub_indexes, column])
                        * duplicate_offset
                        * temp
                    )
                    x += output
                if agg_func is None:
                    x = get_transform(agg_func)(x)
                else:
                    x = x[0]
                x_data.append(x)
                y_data.append(
                    get_transform(agg_func)(transform(data[sub_indexes, column]))
                )
                group_labels.append(group_key)
        output = JitterPlotData(
            x_data=x_data,
            y_data=y_data,
            marker=self._process_dict(groups, marker, unique_groups, None),
            markerfacecolor=self._process_dict(
                groups, markercolor, unique_groups, None
            ),
            markeredgecolor=self._process_dict(groups, edgecolor, unique_groups, None),
            markersize=[markersize] * len(y_data),
            alpha=alpha,
            edge_alpha=edge_alpha,
            group_labels=group_labels,
            direction=direction,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, None),
        )
        return output

    def _summary(
        self,
        data: DataHolder,
        y: str,
        levels: Levels,
        loc_dict: dict[str, float],
        func: Agg,
        capsize: float,
        capstyle: CapStyle,
        barwidth: float,
        linewidth: float | int,
        zorder_dict: dict[str, int],
        color: dict[str, str],
        alpha: AlphaRange,
        x: str = None,
        err_func: Error | None = None,
        ytransform: Transform = None,
        *args,
        **kwargs,
    ) -> SummaryPlotData:
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        transform = get_transform(ytransform)
        y_data = []
        error_data = []
        x_data = []
        group_labels = []

        groups = data.groups(levels)
        for i, indexes in groups.items():
            x_data.append(loc_dict[i])
            y_data.append(get_transform(func)(transform(data[indexes, column])))
            group_labels.append(i)
            if err_func is not None:
                error_data.append(
                    get_transform(err_func)(transform(data[indexes, column]))
                )
            else:
                error_data.append(None)
        output = SummaryPlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            widths=[barwidth] * len(y_data),
            colors=self._process_dict(groups, color),
            linewidth=linewidth,
            alpha=alpha,
            capstyle=capstyle,
            capsize=capsize,
            group_labels=group_labels,
            direction=direction,
            zorder=self._process_dict(groups, zorder_dict),
        )
        return output

    def _summaryu(
        self,
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
        color: dict[str, str],
        zorder_dict: dict[str, int],
        alpha: AlphaRange = 1.0,
        x: str = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        agg_width: float = 1.0,
        ytransform: Transform = None,
        *args,
        **kwargs,
    ) -> SummaryPlotData:
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        transform = get_transform(ytransform)
        y_data = []
        error_data = []
        x_data = []
        widths = []
        group_labels = []
        unique_groups = None

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, indexes in groups.items():
            unique_ids_sub = np.unique(data[indexes, unique_id])
            if agg_func is None:
                if len(unique_ids_sub) > 1:
                    left = loc_dict[group_key] - barwidth / 2
                    right = loc_dict[group_key] + barwidth / 2
                    vals = len(unique_ids_sub) * 2 + 1
                    dist = np.linspace(left, right, num=vals)
                    dist = dist[1::2]
                    w = (dist[1] - dist[0]) * agg_width
                else:
                    dist = [0]
                    w = barwidth
                for index, ui_group in enumerate(unique_ids_sub):
                    if group_key == ("",):
                        temp_group = (ui_group,)
                    else:
                        temp_group = group_key + (ui_group,)
                    widths.append(w)
                    vals = transform(data[unique_groups[temp_group], column])
                    x_data.append(dist[index])
                    y_data.append(get_transform(func)(vals))
                    group_labels.append(group_key)
                    if err_func is not None:
                        error_data.append(get_transform(err_func)(vals))
                    else:
                        error_data.append(None)
            else:
                temp_vals = []
                for index, j in enumerate(unique_ids_sub):
                    vals = transform(data[unique_groups[group_key + (j,)], column])
                    temp_vals.append(get_transform(func)(vals))
                x_data.append(loc_dict[group_key])
                widths.append(barwidth)
                y_data.append(get_transform(func)(np.array(temp_vals)))
                group_labels.append(group_key)
                if err_func is not None:
                    error_data.append(get_transform(err_func)(np.array(temp_vals)))
                else:
                    error_data.append(None)
        output = SummaryPlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            widths=widths,
            colors=self._process_dict(groups, color, unique_groups, agg_func),
            linewidth=linewidth,
            alpha=alpha,
            capstyle=capstyle,
            capsize=capsize,
            group_labels=group_labels,
            direction=direction,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
        )
        return output

    def _box(
        self,
        data: DataHolder,
        y: str,
        levels: Levels,
        loc_dict: dict[str, float],
        facecolor: dict[str, str],
        edgecolor: dict[str, str],
        zorder_dict: dict[str, int],
        fliers: str = "",
        width: float = 1.0,
        linewidth: float | int = 1,
        x: str = None,
        showmeans: bool = False,
        show_ci: bool = False,
        alpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        ytransform: Transform = None,
        *args,
        **kwargs,
    ):
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"
        transform = get_transform(ytransform)

        y_data = []
        x_data = []
        group_labels = []

        groups = data.groups(levels)
        for group_key, value in groups.items():
            y_data.append(transform(data[value, column]))
            x_data.append([loc_dict[group_key]])
            group_labels.append(group_key)
        output = BoxPlotData(
            x_data=x_data,
            y_data=y_data,
            facecolors=self._process_dict(groups, facecolor),
            edgecolors=self._process_dict(groups, edgecolor),
            alpha=alpha,
            linealpha=linealpha,
            fliers=fliers,
            linewidth=linewidth,
            width=width,
            show_ci=show_ci,
            showmeans=showmeans,
            group_labels=group_labels,
            direction=direction,
            zorder=self._process_dict(groups, zorder_dict),
        )
        return output

    def _violin(
        self,
        data: DataHolder,
        y: str,
        levels: Levels,
        loc_dict: dict[str, float],
        facecolor,
        edgecolor: dict[str, str],
        zorder_dict: dict[str, int],
        style: str,
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
        x: str = None,
        KDEType="fft",
        agg_func: Agg | None = None,
        unique_style: Literal["split", "overlap"] = "overlap",
        *args,
        **kwargs,
    ):
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        transform = get_transform(ytransform)

        groups = data.groups(levels)
        unique_groups = None

        x_data = []
        y_data = []
        loc = []
        if style not in {"left", "right", "alternate"}:
            width = width / 2.0
        group_labels = []

        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, group_indexes in groups.items():
            if unique_id is None:
                y_values = np.asarray(data[group_indexes, column]).flatten()
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
                loc.append(loc_dict[group_key])
                group_labels.append(group_key)
            else:
                subgroups = np.unique(data[group_indexes, unique_id])
                if agg_func is not None:
                    temp_data = data[group_indexes, column]
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
                            dist = (dist[1:] + dist[:-1]) / 2.0
                            dist += loc_dict[group_key]
                        else:
                            dist = [loc_dict[group_key]]
                    else:
                        dist = np.full(len(subgroups), loc_dict[group_key])
                        uwidth = width
                    s_indexes = unique_groups[group_key + (s,)]
                    y_values = np.asarray(data[s_indexes, column]).flatten()
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
                        loc.append(dist[hi])
                        group_labels.append(group_key)
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
                    loc.append(loc_dict[group_key])
                    group_labels.append(group_key)

            output = ViolinPlotData(
                x_data=x_data,
                y_data=y_data,
                location=loc,
                facecolors=self._process_dict(
                    groups, facecolor, unique_groups, agg_func
                ),
                edgecolors=self._process_dict(
                    groups, edgecolor, unique_groups, agg_func
                ),
                alpha=alpha,
                edge_alpha=edge_alpha,
                linewidth=linewidth,
                group_labels=group_labels,
                style=style,
                direction=direction,
                zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
            )
        return output

    def _paired_plot():
        pass

    def _bar(
        self,
        data: DataHolder,
        y: str,
        levels: list[str],
        loc_dict: dict,
        facecolor: dict[str, str],
        edgecolor: dict[str, str],
        hatch: str,
        barwidth: float,
        linewidth: float | int,
        alpha: float | int,
        linealpha: float | int,
        zorder_dict: dict[str, int],
        func: Agg,
        x: str = None,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        ytransform: Transform = None,
        *args,
        **kwargs,
    ) -> RectanglePlotData:
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        bw = []
        bottoms = []
        heights = []
        x_loc = []
        hatches = []
        group_labels = []

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        else:
            unique_groups = None
        for group_key, indexes in groups.items():
            if unique_id is None:
                y_values = get_transform(ytransform)(data[indexes, column])
                bw.append(barwidth)
                heights.append(get_transform(func)(y_values))
                bottoms.append(0)
                x_loc.append(loc_dict[group_key])
                hatches.append(hatch[group_key])
                group_labels.append(group_key)
            else:
                unique_ids_sub = np.unique(data[indexes, unique_id])
                if agg_func is None:
                    if len(unique_ids_sub) > 1:
                        left = loc_dict[group_key] - barwidth / 2
                        right = loc_dict[group_key] + barwidth / 2
                        vals = len(unique_ids_sub) * 2 + 1
                        dist = np.linspace(left, right, num=vals)
                        dist = dist[1::2]
                        w = dist[1] - dist[0]
                    else:
                        dist = [0]
                        w = barwidth
                else:
                    output = np.zeros(len(unique_ids_sub))

                for index, ui_group in enumerate(unique_ids_sub):
                    s_indexes = unique_groups[group_key + (ui_group,)]
                    y_values = get_transform(ytransform)(data[s_indexes, column])
                    if agg_func is None:
                        bw.append(w)
                        heights.append(get_transform(func)(y_values))
                        bottoms.append(0)
                        x_loc.append(dist[index])
                        hatches.append(hatch[group_key])
                        group_labels.append(group_key)
                    else:
                        output[index] = get_transform(func)(y_values)
                if agg_func is not None:
                    bw.append(barwidth)
                    heights.append(get_transform(agg_func)(output))
                    bottoms.append(0)
                    x_loc.append(loc_dict[group_key])
                    hatches.append(hatch[group_key])
                    group_labels.append(group_key)
        output = RectanglePlotData(
            heights=heights,
            bottoms=bottoms,
            bins=x_loc,
            binwidths=bw,
            fillcolors=self._process_dict(groups, facecolor, unique_groups, agg_func),
            edgecolors=self._process_dict(groups, edgecolor, unique_groups, agg_func),
            fill_alpha=alpha,
            edge_alpha=linealpha,
            hatches=hatches,
            linewidth=linewidth,
            direction=direction,
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
        )
        return output

    def _percent(
        self,
        data: DataHolder,
        y: str,
        levels: Levels,
        loc_dict: dict[str, float],
        facecolor: dict[str, str],
        edgecolor: dict[str, str],
        cutoff: None | float | int | list[float | int],
        include_bins: list[bool],
        zorder_dict: dict[str, int],
        barwidth: float = 1.0,
        linewidth: float | int = 1,
        alpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        hatch: str | None = None,
        x: str = None,
        unique_id: str | None = None,
        invert: bool = False,
        axis_type: BinType = "density",
        *args,
        **kwargs,
    ) -> RectanglePlotData:
        column = y if x is None else x
        direction = "vertical" if x is None else "horizontal"

        if cutoff is not None:
            bins = np.zeros(len(cutoff) + 2)
            bins[-1] = data[column].max() + 1e-6
            bins[0] = data[column].min() - 1e-6
            for i in range(len(cutoff)):
                bins[i + 1] = cutoff[i]

            if include_bins is None:
                include_bins = [True] * (len(bins) - 1)
            if len(include_bins) < (len(bins) - 1):
                include_bins.extend([True] * ((len(bins) - 1) - len(include_bins)))
        else:
            bins = np.unique(data[column])
            if include_bins is None:
                include_bins = [True] * len(bins)

            if len(include_bins) < len(bins):
                include_bins.extend([True] * (len(bins) - len(include_bins)))

        plot_bins = sum(include_bins)

        if list(hatch.values())[0] is True:
            hs = self.HATCHES[:plot_bins]
        else:
            hs = [None] * plot_bins

        heights = []
        bottoms = []
        x_loc = []
        hatches = []
        bw = []
        group_labels = []
        unique_groups = None

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, indexes in groups.items():
            if unique_id is None:
                bw.append([barwidth] * plot_bins)
                top, bottom = _bin_data(
                    data[indexes, column], bins, axis_type, invert, cutoff
                )
                heights.append(top[include_bins])
                bottoms.append(bottom[include_bins])
                x_s = [loc_dict[group_key]] * plot_bins
                x_loc.append(x_s)
                hatches.append(hs)
                group_labels.append(group_key)
            else:
                unique_ids_sub = np.unique(data[groups[group_key], unique_id])
                temp_width = barwidth / len(unique_ids_sub)
                if len(unique_ids_sub) > 1:
                    dist = np.linspace(
                        -barwidth / 2, barwidth / 2, num=len(unique_ids_sub) + 1
                    )
                    dist = (dist[1:] + dist[:-1]) / 2
                else:
                    dist = [0]
                for index, ui_group in enumerate(unique_ids_sub):
                    top, bottom = _bin_data(
                        data[unique_groups[group_key + (ui_group,)], column],
                        bins,
                        axis_type,
                        invert,
                        cutoff,
                    )
                    heights.append(top[include_bins])
                    bottoms.append(bottom[include_bins])
                    bw.append(temp_width)
                    x_s = [loc_dict[group_key] + dist[index]] * plot_bins
                    x_loc.append(x_s)
                    hatches.append(hs)
                    group_labels.append(group_key)
        fillcolors = self._process_dict(groups, facecolor, unique_groups)
        edgecolors = self._process_dict(groups, edgecolor, unique_groups)
        output = RectanglePlotData(
            heights=heights,
            bottoms=bottoms,
            bins=x_loc,
            binwidths=bw,
            fillcolors=fillcolors,
            edgecolors=edgecolors,
            fill_alpha=alpha,
            edge_alpha=linealpha,
            hatches=hatches,
            linewidth=linewidth,
            direction=direction,
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups),
        )
        return output
