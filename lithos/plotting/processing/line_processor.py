from collections import defaultdict
from typing import Literal

import numpy as np

from ... import stats
from ...stats import hist
from ...types.basic_types import (
    BW,
    BinType,
    FitFunc,
    HistBinLimits,
    HistStat,
    HistType,
    JitterType,
    Kernels,
    Levels,
    NBins,
    Transform,
)
from ...types.plot_input import (
    Agg,
    AlphaRange,
    Error,
    Grouping,
    Subgrouping,
    UniqueGrouping,
)
from ...types.plot_types import LinePlotData, RectanglePlotData, ScatterPlotData
from ...utils import DataHolder, get_transform
from ..plot_utils import _create_groupings, create_dict
from .base_processor import BaseProcessor


class LineProcessor(BaseProcessor):
    def __init__(self, markers, hatches):
        super().__init__(markers, hatches)
        self.PLOTS = {
            "hist": self._hist,
            "line": self._line,
            "kde": self._kde,
            "ecdf": self._ecdf,
            "scatter": self._scatter,
            "aggline": self._aggline,
            "fit": self._fit,
        }

    def process_groups(
        self,
        data: DataHolder,
        group: int | str,
        subgroup: int | str,
        group_order: Grouping,
        subgroup_order: Subgrouping,
        facet: bool = False,
        facet_title: bool = False,
        **kwargs,
    ):
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            data, group, subgroup, group_order, subgroup_order
        )

        if facet:
            loc_dict = create_dict(group_order, unique_groups)
        else:
            loc_dict = create_dict(0, unique_groups)

        zgroup = group_order if subgroup_order is None else subgroup_order
        zorder_dict = create_dict(zgroup, unique_groups)

        self._plot_dict = {
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "loc_dict": loc_dict,
            "levels": levels,
            "zorder_dict": zorder_dict,
            "facet_title": facet_title,
            "facet": facet,
        }

    def _post_process_density(
        self, plot_data, hist_type: HistType, facet_index: list[int] | np.ndarray
    ):
        plot_data = np.asarray(plot_data)
        output = np.zeros(plot_data.shape)
        bottoms = np.zeros(plot_data.shape)
        indexes = np.unique(facet_index)
        facet_index = np.array(facet_index)
        for i in indexes:
            t_indexes = np.where(facet_index == i)[0]
            p, b = self._post_process_density_type(plot_data[t_indexes, :], hist_type)
            output[t_indexes, :] = p[:, :]
            bottoms[t_indexes, :] = b[:, :]
        return output, bottoms

    def _post_process_density_type(self, data, hist_type: HistType):
        if hist_type == "fill":
            og = np.asarray(data)
            t_sum = og.sum(axis=0)
            output = np.divide(og, t_sum, where=t_sum > 0)
            output[og == 0] = 0
            bottoms = np.zeros(output.shape)
            bottoms[1:, :] = output.cumsum(axis=0)[:-1, :]
            bottoms[output == 0] = 0
        elif hist_type == "stack":
            output = np.asarray(data)
            bottoms = np.zeros(output.shape)
            bottoms[1:, :] = output.cumsum(axis=0)[:-1, :]
            bottoms[output == 0] = 0
        else:
            output = np.asarray(data)
            bottoms = np.zeros(output.shape)
        return output, bottoms

    def _hist(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        facecolor: dict[str, str],
        edgecolor: dict[str, str],
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        hatch: dict[str, str],
        hist_type: HistType = "bar",
        fillalpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        bin_limits: HistBinLimits = None,
        linewidth: float | int = 2,
        nbins: NBins = 50,
        stat: HistStat = "density",
        agg_func: Agg | None = None,
        unique_id: str | None = None,
        ytransform: Transform = None,
        xtransfrom=None,
        *args,
        **kwargs,
    ):
        axis = "vertical" if y is None else "horizontal"
        column = y if x is None else x
        transform = ytransform if xtransfrom is None else xtransfrom

        plot_data = []
        plot_bins = []
        count = 0
        group_labels = []

        unique_groups = None

        if agg_func is not None and isinstance(nbins, str):
            raise ValueError("nbins must be integer if agg_func is not None.")

        bins = None
        if bin_limits == "common":
            bins = np.histogram_bin_edges(
                get_transform(transform)(data[column]), bins=nbins
            )

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, group_indexes in groups.items():
            if unique_id is not None:
                if bins is None:
                    t = get_transform(transform)(data[group_indexes, column])
                    if bin_limits is None:
                        limits = (t.min(), t.max())
                    bins = np.histogram_bin_edges(
                        t,
                        bins=nbins,
                        range=limits,
                    )
                subgroup = np.unique(data[group_indexes, unique_id])
                if agg_func is not None:
                    if isinstance(nbins, str):
                        nbins = 50
                    temp_list = np.zeros((len(subgroup), nbins))
                else:
                    temp_list = []
                for index, j in enumerate(subgroup):
                    if unique_groups is None:
                        raise ValueError("unique_groups must not be None")
                    temp_data = np.sort(data[unique_groups[group_key + (j,)], column])
                    poly = hist(get_transform(transform)(temp_data), bins, stat)
                    if agg_func is not None:
                        temp_list[index] = poly
                    else:
                        plot_data.append(poly)
                        plot_bins.append(bins[:-1])
                        group_labels.append(group_key)
                        count += 1
                if agg_func is not None:
                    plot_data.append(get_transform(agg_func)(temp_list, axis=0))
                    plot_bins.append(bins)
                    group_labels.append(group_key)
                    count += 1
            else:
                temp_data = np.sort(data[groups[group_key], column])
                if bins is None:
                    t = get_transform(transform)(data[group_indexes, column])
                    if bin_limits is None:
                        limits = (t.min(), t.max())
                    bins = np.histogram_bin_edges(
                        t,
                        bins=nbins,
                        range=limits,
                    )
                poly = hist(get_transform(transform)(temp_data), bins, stat)
                plot_data.append(poly)
                plot_bins.append(bins)
                group_labels.append(group_key)
                count += 1
        facet_index = self._process_dict(groups, loc_dict, unique_groups, agg_func)
        if hist_type != "step":
            plot_data, bottoms = self._post_process_density(
                plot_data, hist_type, facet_index
            )
            output = RectanglePlotData(
                heights=plot_data,
                bottoms=bottoms,
                bins=[i[:-1] for i in plot_bins],
                binwidths=[np.full(len(i) - 1, i[1] - i[0]) for i in plot_bins],
                fillcolors=self._process_dict(
                    groups, facecolor, unique_groups, agg_func
                ),
                edgecolors=self._process_dict(
                    groups, edgecolor, unique_groups, agg_func
                ),
                fill_alpha=fillalpha,
                edge_alpha=linealpha,
                hatches=self._process_dict(groups, hatch, unique_groups, agg_func),
                linewidth=linewidth,
                facet_index=facet_index,
                direction=axis,
                group_labels=group_labels,
                zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
            )
        else:
            nones = [None for _ in plot_data]
            for index in range(len(plot_data)):
                p = np.zeros(len(plot_data[index]) * 2 + 2)
                p[1:-1] = np.repeat(plot_data[index], 2)
                plot_data[index] = p
                plot_bins[index] = np.repeat(plot_bins[index], 2)
            output = LinePlotData(
                x_data=plot_bins,
                y_data=plot_data,
                error_data=[None for _ in plot_data],
                facet_index=self._process_dict(
                    groups, loc_dict, unique_groups, agg_func
                ),
                marker=[None for _ in plot_data],
                linecolor=self._process_dict(
                    groups, edgecolor, unique_groups, agg_func
                ),
                fillcolor=self._process_dict(
                    groups, facecolor, unique_groups, agg_func
                ),
                linewidth=[linewidth for _ in plot_data],
                linestyle=["-" for _ in plot_data],
                markerfacecolor=nones,
                markeredgecolor=nones,
                markersize=0,
                fill_between=False,
                fill_under=True,
                linealpha=linealpha,
                fillalpha=fillalpha,
                direction=axis,
                group_labels=group_labels,
                zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
            )
        return output

    def _scatter(
        self,
        data,
        y,
        x,
        marker,
        markercolor,
        edgecolor,
        markersize,
        alpha,
        edge_alpha,
        linewidth,
        facetgroup,
        zorder_dict: dict[str, int],
        loc_dict: dict[str, int],
        xtransform: Transform = None,
        ytransform: Transform = None,
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
        group_labels = []
        zorder = []

        for key, value in loc_dict.items():
            indexes = np.array(
                [index for index, j in enumerate(facetgroup) if value == j]
            )
            x_data.append(get_transform(xtransform)(data[indexes, x]))
            y_data.append(get_transform(ytransform)(data[indexes, y]))
            mks.append(marker)
            mfcs.append([markercolor[i] for i in indexes])
            mecs.append([edgecolor[i] for i in indexes])
            mksizes.append([markersize[i] for i in indexes])
            facet.append(loc_dict[key])
            group_labels.append(key)
            zorder.append(zorder_dict[key])
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
            linewidth=linewidth,
            group_labels=group_labels,
            direction="vertical",
            zorder=zorder,
        )
        return output

    def _aggline(
        self,
        data: DataHolder,
        x: str,
        y: str,
        levels: Levels,
        marker: dict,
        markersize: float | int,
        markerfacecolor: dict,
        markeredgecolor: dict,
        linestyle: dict,
        linewidth: float | int,
        linecolor: dict,
        fillcolor: dict,
        linealpha: float | int,
        loc_dict: dict,
        zorder_dict: dict,
        func: Agg | None = None,
        err_func: Error = None,
        fill_between: bool = False,
        fillalpha: AlphaRange = 1.0,
        agg_func: Agg | None = None,
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
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
        fcs = []
        lss = []
        mfcs = []
        mecs = []
        group_labels = []
        zorder = []

        err_data = None
        new_levels = (levels + (x,)) if unique_id is None else (levels + (x, unique_id))
        ytransform = get_transform(ytransform)
        func = get_transform(func)
        agg_dict = {col: (y, lambda x: func(ytransform(x))) for col in [y]}
        new_data = data.groupby(y, new_levels, sort=sort).agg(**agg_dict)
        if unique_id is None:
            if err_func is not None:
                agg_dict = {
                    col: (y, lambda x: get_transform(err_func)((ytransform(x))))
                    for col in [y]
                }
                err_data = DataHolder(
                    data.groupby(y, new_levels, sort=sort).agg(**agg_dict)
                )
        else:
            if agg_func is not None:
                if err_func is not None:
                    err_data = DataHolder(
                        new_data[list(levels + (x, y))]
                        .groupby(list(levels + (x,)), sort=sort, as_index=False)
                        .agg(get_transform(err_func))
                    )
                new_data = (
                    new_data[list(levels + (x, y))]
                    .groupby(list(levels + (x,)), sort=sort, as_index=False)
                    .agg(get_transform(agg_func))
                )
        new_data = DataHolder(new_data)
        if unique_id is not None and agg_func is None:
            ugrps = new_data.groups(levels + (unique_id,))
        else:
            ugrps = new_data.groups(levels)
        for u, indexes in ugrps.items():
            u = u if len(u) == len(levels) else u[: len(levels)]
            u = ("",) if len(u) == 0 else u
            ytemp = new_data[indexes, y]
            y_data.append(ytemp)
            xtemp = get_transform(xtransform)(new_data[indexes, x])
            x_data.append(xtemp)
            temp_err = err_data[indexes, y] if err_data is not None else None
            error_data.append(temp_err)
            facet_index.append(loc_dict[u])
            mks.append(marker[u])
            lcs.append(linecolor[u])
            fcs.append(fillcolor[u])
            lss.append(linestyle[u])
            mfcs.append(markerfacecolor[u])
            mecs.append(markeredgecolor[u])
            group_labels.append(u)
            zorder.append(zorder_dict[u])
        output = LinePlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            facet_index=facet_index,
            marker=mks,
            linecolor=lcs,
            fillcolor=fcs,
            linewidth=[linewidth for _ in y_data],
            linestyle=lss,
            markerfacecolor=mfcs,
            markeredgecolor=mecs,
            markersize=markersize,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
            direction="vertical",
            group_labels=group_labels,
            zorder=zorder,
        )

        return output

    def _kde(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        linecolor: str | dict[str, str],
        fillcolor: str | dict[str, str],
        loc_dict: dict[str, int],
        linestyle: str | dict[str, str],
        linewidth: float | int,
        linealpha: float | int,
        fillalpha: float | int,
        fill_between: bool,
        fill_under: bool,
        zorder_dict: dict[str, int],
        kernel: Kernels = "gaussian",
        bw: BW = "ISJ",
        kde_length: int | None = None,
        tol: float | int | tuple[float, float] = 1e-3,
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
        group_labels = []
        unique_groups = None

        column = y if x is None else x
        direction = direction = "vertical" if y is None else "horizontal"
        transform = ytransform if xtransform is None else xtransform

        groups = data.groups(levels)

        if isinstance(tol, tuple):
            min_data, max_data = tol
            if min_data >= data[column].min():
                raise ValueError(
                    f"tol[0] must be less than the minimum value of {column}."
                )
            if max_data <= data[column].max():
                raise ValueError(
                    f"tol[1] must be greater than the maximum value of {column}."
                )

        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, group_indexes in groups.items():
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
                y_data.append(y_kde)
                x_data.append(x_kde)
                error_data.append(None)
                group_labels.append(group_key)
            else:
                subgroups, count = np.unique(
                    data[group_indexes, unique_id], return_counts=True
                )

                if agg_func is not None:
                    if not isinstance(tol, tuple):
                        temp_data = data[group_indexes, column]
                        min_data = get_transform(transform)(temp_data.min())
                        max_data = get_transform(transform)(temp_data.max())
                        min_data = min_data - np.abs((min_data * tol))
                        max_data = max_data + np.abs((max_data * tol))
                        min_data = min_data if min_data != 0 else -1e-10
                        max_data = max_data if max_data != 0 else 1e-10
                    elif isinstance(tol, tuple) and len(tol) == 2:
                        min_data, max_data = tol[0], tol[1]
                    else:
                        raise ValueError("Cannot determine min and max for linspace.")
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
                    if unique_groups is None:
                        raise ValueError("unique_groups must not be None.")
                    s_indexes = unique_groups[group_key + (s,)]
                    y_values = np.asarray(data[s_indexes, column]).flatten()
                    temp_size = y_values.size
                    if agg_func is None:
                        x_kde, y_kde = stats.kde(
                            get_transform(transform)(y_values),
                            bw=bw,
                            kernel=kernel,
                            tol=tol,
                            kde_length=kde_length,
                        )
                        y_data.append(y_kde)
                        x_data.append(x_kde)
                        error_data.append(None)
                        group_labels.append(group_key)
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
                    y_kde = get_transform(agg_func)(y_hold, axis=0)
                    y_data.append(y_kde)
                    x_data.append(x_array)
                    group_labels.append(group_key)
                    error_data.append(
                        get_transform(err_func)(y_hold, axis=0)
                        if err_func is not None
                        else None
                    )
        nones = [None] * len(y_data)
        output = LinePlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            facet_index=self._process_dict(groups, loc_dict, unique_groups, agg_func),
            marker=nones,
            linecolor=self._process_dict(groups, linecolor, unique_groups, agg_func),
            fillcolor=self._process_dict(groups, fillcolor, unique_groups, agg_func),
            linewidth=[linewidth] * len(y_data),
            linestyle=self._process_dict(groups, linestyle, unique_groups, agg_func),
            markerfacecolor=nones,
            markeredgecolor=nones,
            markersize=None,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
            fill_under=fill_under,
            direction=direction,
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
        )
        return output

    def _ecdf(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        linewidth: float | int,
        linecolor: str | dict[str, str],
        fillcolor: str | dict[str, str],
        loc_dict: dict[str, int],
        linestyle: str | dict[str, str],
        linealpha: float | int,
        zorder_dict: dict[str, int],
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

        if ecdf_args is None:
            ecdf_args = {}
        x_data = []
        y_data = []
        error_data = []
        group_labels = []
        unique_groups = None

        groups = data.groups(levels)

        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))

        for group_key, indexes in groups.items():
            if unique_id is None:
                y_values = np.asarray(data[indexes, column]).flatten()
                x_ecdf, y_ecdf = stats.ecdf(
                    get_transform(transform)(y_values), ecdf_type=ecdf_type, **ecdf_args
                )
                y_data.append(y_ecdf)
                x_data.append(x_ecdf)
                error_data.append(None)
                group_labels.append(group_key)
            else:
                subgroups, counts = np.unique(
                    data[indexes, unique_id], return_counts=True
                )
                if agg_func is not None:
                    if "size" not in ecdf_args:
                        ecdf_args["size"] = np.max(counts)
                    y_ecdf = np.arange(ecdf_args["size"]) / ecdf_args["size"]
                    x_hold = np.zeros((len(subgroups), ecdf_args["size"]))
                for hi, s in enumerate(subgroups):
                    if unique_groups is None:
                        raise ValueError("unique_groups must not be None.")
                    y_values = np.asarray(
                        data[unique_groups[group_key + (s,)], column]
                    ).flatten()
                    if agg_func is None:
                        x_ecdf, y_ecdf = stats.ecdf(
                            get_transform(transform)(y_values),
                            ecdf_type=ecdf_type,
                            **ecdf_args,
                        )
                        y_data.append(y_ecdf)
                        x_data.append(x_ecdf)
                        error_data.append(None)
                        group_labels.append(group_key)
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
                    group_labels.append(group_key)
                    error_data.append(
                        get_transform(err_func)(x_hold, axis=0)
                        if err_func is not None
                        else None
                    )
        nones = [None] * len(y_data)
        output = LinePlotData(
            x_data=y_data,
            y_data=x_data,
            error_data=error_data,
            facet_index=self._process_dict(groups, loc_dict, unique_groups, agg_func),
            marker=nones,
            linecolor=self._process_dict(groups, linecolor, unique_groups, agg_func),
            fillcolor=self._process_dict(groups, fillcolor, unique_groups, agg_func),
            linewidth=[linewidth] * len(y_data),
            linestyle=self._process_dict(groups, linestyle, unique_groups, agg_func),
            markerfacecolor=nones,
            markeredgecolor=nones,
            markersize=None,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
            direction="horizontal",
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
        )
        return output

    def _line(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        linecolor: dict[str, str],
        loc_dict: dict[str, int],
        linestyle: dict[str, str],
        zorder_dict: dict[str, int],
        linewidth: float | int = 2,
        unique_id: str | None = None,
        linealpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 0.5,
        xtransform: Transform = None,
        ytransform: Transform = None,
        func: Agg = "mean",
        err_func: Error = "sem",
        index: str | None = None,
        *args,
        **kwargs,
    ) -> LinePlotData:
        x_data = []
        y_data = []
        group_labels = []
        err_data = []
        unique_groups = None

        if index is None and x is not None:
            index = x

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, indexes in groups.items():
            if unique_id is None:
                temp_y = np.asarray(data[indexes, y])
                if x is not None:
                    temp_x = np.asarray(data[indexes, x])
                    x_data.append(get_transform(xtransform)(temp_x))
                else:
                    x_data.append(get_transform(xtransform)(np.arange(len(temp_y))))
                y_data.append(get_transform(ytransform)(temp_y))
                group_labels.append(group_key)
                err_data.append(None)
            else:
                uids = np.unique(data[indexes, unique_id])
                if func is not None:
                    seen = set()
                    seq = data[indexes, x]
                    if x is None:
                        raise ValueError("x must be passed if you want to aggregate y")
                    x_temp = [x for x in seq if not (x in seen or seen.add(x))]
                    x_output = np.zeros((len(uids), len(x_temp)))
                    y_output = np.zeros((len(uids), len(x_temp)))
                for uid_index, j in enumerate(uids):
                    if unique_groups is None:
                        raise ValueError("unique_groups must not be None")
                    sub_indexes = unique_groups[group_key + (j,)]
                    temp_y = np.asarray(data[sub_indexes, y])
                    if func is None:
                        y_data.append(get_transform(ytransform)(temp_y))
                        group_labels.append(group_key)
                        if x is not None:
                            temp_x = np.asarray(data[sub_indexes, x])
                            x_data.append(get_transform(xtransform)(temp_x))
                        else:
                            x_data.append(
                                get_transform(xtransform)(np.arange(len(temp_y)))
                            )
                        err_data.append(None)
                    else:
                        temp_x = np.asarray(data[sub_indexes, x])
                        y_output[uid_index, :] = get_transform(ytransform)(temp_y)
                        x_output[uid_index, :] = get_transform(ytransform)(temp_x)
                if func is not None:
                    y_data.append(get_transform(func)(y_output, axis=0))
                    x_data.append(get_transform(func)(x_output, axis=0))
                    group_labels.append(group_key)
                if err_func is not None:
                    err_data.append(get_transform(err_func)(y_output, axis=0))
                else:
                    err_data.append(None)
        nones = [None] * len(y_data)
        if err_func is not None:
            fillcolor = self._process_dict(groups, linecolor, unique_groups, func)
            fill_between = True
        else:
            fillcolor = nones
            fill_between = False
        output = LinePlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=err_data,
            facet_index=self._process_dict(groups, loc_dict, unique_groups, func),
            marker=nones,
            linecolor=self._process_dict(groups, linecolor, unique_groups, func),
            fillcolor=fillcolor,
            linewidth=[linewidth] * len(y_data),
            linestyle=self._process_dict(groups, linestyle, unique_groups, func),
            markerfacecolor=nones,
            markeredgecolor=nones,
            markersize=None,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
            direction="vertical",
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, func),
        )
        return output

    def _fit(
        self,
        data: DataHolder,
        y: str,
        x: str,
        fit_func: FitFunc,
        levels: Levels,
        linecolor: dict[str, str],
        loc_dict: dict[str, int],
        linestyle: dict[str, str],
        zorder_dict: dict[str, int],
        linewidth: float | int = 2,
        unique_id: str | None = None,
        linealpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 0.5,
        xtransform: Transform = None,
        ytransform: Transform = None,
        fit_args: dict | None = None,
        fill_between: bool = False,
        ci_func: Literal["ci", "pi"] = "ci",
        agg_func: Agg | None = None,
        err_func: Agg | None = None,
        **kwargs,
    ):
        x_data = []
        y_data = []
        error_data = []
        group_labels = []
        unique_groups = None

        if fit_args is None:
            fit_args = {}

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, indexes in groups.items():
            if unique_id is None:
                temp_y = get_transform(ytransform)(np.asarray(data[indexes, y]))
                temp_x = get_transform(xtransform)(np.asarray(data[indexes, x]))
                fit_output = stats.fit(
                    fit_func=fit_func, x=temp_x, y=temp_y, ci_func=ci_func, **fit_args
                )
                y_data.append(fit_output[1])
                x_data.append(fit_output[2])
                error_data.append(fit_output[3])
                group_labels.append(group_key)
            else:
                uids = np.unique(data[indexes, unique_id])
                if agg_func is not None:
                    temp_data = data[indexes, x]
                    min_data = get_transform(xtransform)(temp_data.min())
                    max_data = get_transform(xtransform)(temp_data.max())
                    x_array = np.linspace(min_data, max_data, num=100)
                    y_hold = np.zeros((len(uids), 100))
                    x_data.append(x_array)
                for uindex, j in enumerate(uids):
                    if unique_groups is None:
                        raise ValueError("unique_groups must not be None.")
                    sub_indexes = unique_groups[group_key + (j,)]
                    if agg_func is None:
                        temp_y = get_transform(ytransform)(
                            np.asarray(data[sub_indexes, y])
                        )
                        temp_x = get_transform(xtransform)(
                            np.asarray(data[sub_indexes, x])
                        )
                        fit_output = stats.fit(
                            fit_func=fit_func,
                            x=temp_x,
                            y=temp_y,
                            ci_func=ci_func,
                            **fit_args,
                        )
                        y_data.append(fit_output[1])
                        x_data.append(fit_output[2])
                        error_data.append(None)
                        group_labels.append(group_key)
                    else:
                        temp_y = get_transform(ytransform)(
                            np.asarray(data[sub_indexes, y])
                        )
                        temp_x = get_transform(xtransform)(
                            np.asarray(data[sub_indexes, x])
                        )
                        fit_output = stats.fit(
                            fit_func=fit_func,
                            x=temp_x,
                            y=temp_y,
                            fit_x=x_array,
                            ci_func=ci_func,
                            **fit_args,
                        )
                        y_hold[uindex, :] = fit_output[1]
                if agg_func is not None:
                    y_values = get_transform(agg_func)(y_hold, axis=0)
                    y_data.append(y_values)
                    error_values = get_transform(err_func)(y_hold, axis=0)
                    error_data.append(error_values)
                    group_labels.append(group_key)
        nones = [None] * len(y_data)
        output = LinePlotData(
            x_data=x_data,
            y_data=y_data,
            error_data=error_data,
            facet_index=self._process_dict(groups, loc_dict, unique_groups, agg_func),
            marker=nones,
            linecolor=self._process_dict(groups, linecolor, unique_groups, agg_func),
            fillcolor=self._process_dict(groups, linecolor, unique_groups, agg_func),
            linewidth=[linewidth] * len(y_data),
            linestyle=self._process_dict(groups, linestyle, unique_groups, agg_func),
            markerfacecolor=nones,
            markeredgecolor=nones,
            markersize=None,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
            direction="vertical",
            group_labels=group_labels,
            zorder=self._process_dict(groups, zorder_dict, unique_groups, agg_func),
        )
        return output
