from typing import Literal

import numpy as np

from ... import stats
from ...utils import DataHolder, get_transform
from ..plot_utils import _calc_hist, create_dict, _create_groupings
from ..types import (
    BW,
    Agg,
    AlphaRange,
    Error,
    Kernels,
    Levels,
    RectanglePlotData,
    ScatterPlotData,
    MarkerLinePlotData,
    FillBetweenPlotData,
    LinePlotData,
    FillUnderPlotData,
    Transform,
    FitFunc,
)

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

    def post_process_line(self, plot_data, style):
        plot_type = style.pop("plot_type")
        temp = {}
        for key, value in style.items():
            temp[key] = self._process_output(plot_data["group_labels"], value)
        temp.update(plot_data)
        if plot_type == "marker":
            output = MarkerLinePlotData(**temp)
        elif plot_type == "fill_between":
            output = FillBetweenPlotData(**temp)
        elif plot_type == "fill_under":
            output = FillUnderPlotData(**temp)
        if plot_type == "line" or temp["error_data"] is None:
            _ = temp.pop("error_data")
            output = LinePlotData(**temp)
        return output

    def process_groups(
        self, data, group, subgroup, group_order, subgroup_order, facet, facet_title
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
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        fillalpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        bin_limits: list[float, float] | None = None,
        linewidth: float | int = 2,
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
            output = self._bar_histogram(
                data=data,
                y=y,
                x=x,
                levels=levels,
                facecolor=facecolor,
                edgecolor=edgecolor,
                loc_dict=loc_dict,
                zorder_dict=zorder_dict,
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
                linewidth=linewidth,
                *args,
                **kwargs,
            )
        return output

    def _bar_histogram(
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
        fillalpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        linewidth: float | int = 2,
        bin_limits: list[float, float] | Literal["common"] | None = None,
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
        group_labels = []

        unique_groups = None

        bins = None
        if bin_limits == "common":
            bins = np.histogram_bin_edges(get_transform(transform)(data[y]), bins=nbins)

        groups = data.groups(levels)
        if unique_id is not None:
            unique_groups = data.groups(levels + (unique_id,))
        for group_key, group_indexes in groups.items():
            if unique_id is not None:
                if bins is None:
                    bins = np.histogram_bin_edges(
                        get_transform(transform)(data[group_indexes[group_key], y]),
                        bins=nbins,
                        range=bin_limits,
                    )
                temp_bw = np.full(nbins, bins[1] - bins[0])
                subgroup = np.unique(data[group_indexes, unique_id])
                if agg_func is not None:
                    temp_list = np.zeros((len(subgroup), nbins))
                else:
                    temp_list = []
                for index, j in enumerate(subgroup):
                    temp_data = np.sort(data[unique_groups[group_key + (j,)], y])
                    poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                    if agg_func is not None:
                        temp_list[index] = poly
                    else:
                        plot_data.append(poly)
                        bw.append(temp_bw)
                        plot_bins.append(bins[:-1])
                        group_labels.append(group_key)
                        count += 1
                if agg_func is not None:
                    plot_data.append(get_transform(agg_func)(temp_list, axis=0))
                    bw.append(temp_bw)
                    plot_bins.append(bins[:-1])
                    group_labels.append(group_key)
                    count += 1
            else:
                temp_data = np.sort(data[groups[group_key], y])
                if bins is None:
                    bins = np.histogram_bin_edges(
                        get_transform(transform)(data[group_indexes, y]),
                        bins=nbins,
                        range=bin_limits,
                    )
                bw.append(np.full(nbins, bins[1] - bins[0]))
                poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                plot_data.append(poly)
                plot_bins.append(bins[:-1])
                group_labels.append(group_key)
                count += 1

        bottoms = [bottom for _ in plot_bins]
        output = RectanglePlotData(
            heights=plot_data,
            bottoms=bottoms,
            bins=plot_bins,
            binwidths=bw,
            fillcolors=self._process_output(group_labels, facecolor),
            edgecolors=self._process_output(group_labels, edgecolor),
            fill_alpha=fillalpha,
            edge_alpha=linealpha,
            hatches=self._process_output(group_labels, hatch),
            linewidth=linewidth,
            facet_index=self._process_output(group_labels, loc_dict),
            axis=axis,
            group_labels=group_labels,
            zorder=self._process_output(group_labels, zorder_dict),
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
            zorder=zorder,
        )
        return output

    def _aggline(
        self,
        data: DataHolder,
        x: str,
        y: str,
        levels: Levels,
        style: dict,
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        func: Agg = None,
        err_func: Error = None,
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
        group_labels = []
        zorder = []

        error_data = None
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
                error_df = DataHolder(
                    data.groupby(y, new_levels, sort=sort).agg(**agg_dict)
                )
        else:
            if agg_func is not None:
                if err_func is not None:
                    error_df = DataHolder(
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
            group_labels.append(u)
            y_data.append(new_data[indexes, y])
            x_data.append(get_transform(xtransform)(new_data[indexes, x]))
            temp_err = error_df[indexes, y] if err_func is not None else None
            error_data.append(temp_err)
            facet_index.append(loc_dict[u])
            zorder.append(zorder_dict[u])
        output = {
            "x_data": x_data,
            "y_data": y_data,
            "error_data": error_data,
            "facet_index": facet_index,
            "direction": "y",
            "group_labels": group_labels,
            "zorder": zorder,
        }
        output = self.post_process_line(output, style)
        return output

    def _kde(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        style: dict,
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
        error_data = [] if err_func is not None else None
        group_labels = []
        unique_groups = None

        column = y if x is None else x
        direction = "x" if x is None else "y"
        transform = ytransform if xtransform is None else xtransform

        groups = data.groups(levels)

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
                if y is not None:
                    y_kde, x_kde = x_kde, y_kde
                y_data.append(y_kde)
                x_data.append(x_kde)
                group_labels.append(group_key)
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
                        if y is not None:
                            y_kde, x_kde = x_kde, y_kde
                        y_data.append(y_kde)
                        x_data.append(x_kde)
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
                    if y is not None:
                        y_kde, x_kde = x_array, get_transform(agg_func)(y_hold, axis=0)
                    else:
                        x_kde, y_kde = x_array, get_transform(agg_func)(y_hold, axis=0)
                    y_data.append(y_kde)
                    x_data.append(x_kde)
                    group_labels.append(group_key)
                    if err_func is not None:
                        error_data.append(get_transform(err_func)(y_hold, axis=0))
        output = {
            "x_data": x_data,
            "y_data": y_data,
            "error_data": error_data,
            "facet_index": self._process_output(group_labels, loc_dict),
            "direction": direction,
            "group_labels": group_labels,
            "zorder": self._process_output(group_labels, zorder_dict),
        }
        output = self.post_process_line(output, style)
        return output

    def _ecdf(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        style: dict,
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
        error_data = [] if err_func is not None else None
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
                    if err_func is not None:
                        error_data.append(get_transform(err_func)(x_hold, axis=0))
        output = {
            "x_data": x_data,
            "y_data": y_data,
            "error_data": error_data,
            "facet_index": self._process_output(group_labels, loc_dict),
            "direction": "x",
            "group_labels": group_labels,
            "zorder": self._process_output(group_labels, zorder_dict),
        }
        output = self.post_process_line(output, style)
        return output

    def _line(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: Levels,
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        style: dict,
        unique_id: str | None = None,
        xtransform: Transform = None,
        ytransform: Transform = None,
        func: Agg = "mean",
        err_func: Error = None,
        index: str | None = None,
        *args,
        **kwargs,
    ) -> LinePlotData:
        x_data = []
        y_data = []
        group_labels = []
        error_data = [] if err_func is not None else None
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
                for index, j in enumerate(uids):
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
                    else:
                        temp_x = np.asarray(data[sub_indexes, x])
                        y_output[index, :] = get_transform(ytransform)(temp_y)
                        x_output[index, :] = get_transform(ytransform)(temp_x)
                if func is not None:
                    y_data.append(get_transform(func)(y_output, axis=0))
                    x_data.append(get_transform(func)(x_output, axis=0))
                    group_labels.append(group_key)
                if err_func is not None:
                    error_data.append(get_transform(err_func)(y_output, axis=0))
        output = {
            "x_data": x_data,
            "y_data": y_data,
            "error_data": error_data,
            "facet_index": self._process_output(group_labels, loc_dict),
            "direction": "y",
            "group_labels": group_labels,
            "zorder": self._process_output(group_labels, zorder_dict),
        }
        output = self.post_process_line(output, style)
        return output

    def _fit(
        self,
        data: DataHolder,
        y: str,
        x: str,
        fit_func: FitFunc,
        levels: Levels,
        loc_dict: dict[str, int],
        zorder_dict: dict[str, int],
        style: dict,
        unique_id: str | None = None,
        xtransform: Transform = None,
        ytransform: Transform = None,
        fit_args: dict | None = None,
        func: Agg = None,
        **kwargs,
    ):
        x_data = []
        y_data = []
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
                    fit_func=fit_func, x=temp_x, y=temp_y, **fit_args
                )
                y_data.append(fit_output[1])
                x_data.append(temp_x)
                group_labels.append(group_key)
            else:
                uids = np.unique(data[indexes, unique_id])
                for j in uids:
                    sub_indexes = unique_groups[group_key + (j,)]
                    temp_y = get_transform(ytransform)(np.asarray(data[sub_indexes, y]))
                    temp_x = get_transform(xtransform)(np.asarray(data[sub_indexes, x]))
                    fit_output = stats.fit(
                        fit_func=fit_func, x=temp_x, y=temp_y, **fit_args
                    )
                    y_data.append(fit_output[1])
                    x_data.append(temp_x)
                    group_labels.append(group_key)
        output = {
            "x_data": x_data,
            "y_data": y_data,
            "error_data": None,
            "facet_index": self._process_output(group_labels, loc_dict),
            "direction": "y",
            "group_labels": group_labels,
            "zorder": self._process_output(group_labels, zorder_dict),
        }
        output = self.post_process_line(output, style)
        return output
