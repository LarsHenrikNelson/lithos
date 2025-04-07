from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..plot_utils import (
    _process_colors,
    create_dict,
    process_args,
    process_scatter_args,
)
from ..types import (
    BW,
    Agg,
    AlphaRange,
    ColorParameters,
    Error,
    KDEType,
    SavePath,
)
from .. import matplotlib_plotter as mpl
from .base_class import BasePlot
from ..processing import LineProcessor


class LinePlot(BasePlot):
    ecdf_args = {
        "spline": {"size": 1000, "bc_type": "natural"},
        "bootstrap": {"size": 1000, "repititions": 1000, "seed": 42},
    }

    def __init__(self, data: pd.DataFrame, inplace: bool = False):
        super().__init__(data, inplace)

        if not self.inplace:
            self.inplace = True
            self.grouping()
            self.inplace = False
        else:
            self.grouping()

    def grouping(
        self,
        group: str | int | None = None,
        subgroup: str | int | None = None,
        group_order: list[str | int | float] | None = None,
        subgroup_order: list[str | int | float] | None = None,
        facet: bool = False,
        facet_title: bool = False,
    ):
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "facet": facet,
            "facet_title": facet_title,
        }

        if not self.inplace:
            return self

    def line(
        self,
        linecolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
    ):
        self._plot_methods.append("line")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "unique_id": unique_id,
            }
        )
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(linecolor, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])
        line_plot = {
            "linecolor_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "unique_id": unique_id,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("line", line_plot))

        if not self.inplace:
            return self

    def aggline(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameters = "glasbey_category10",
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float = 1.0,
        func: Agg = "mean",
        err_func: Error = "sem",
        agg_func: Agg | None = None,
        fill_between: bool = False,
        fillalpha: AlphaRange = 1.0,
        sort=True,
        unique_id=None,
    ):
        self._plot_methods.append("aggline")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "linealpha": linealpha,
                "func": func,
                "err_func": err_func,
                "agg_func": agg_func,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "sort": sort,
                "unique_id": None,
            }
        )
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(
            linecolor,
            self._plot_dict["unique_groups"],
        )
        markerfacecolor_dict = create_dict(
            markerfacecolor,
            self._plot_dict["unique_groups"],
        )
        markeredgecolor_dict = create_dict(
            markeredgecolor,
            self._plot_dict["unique_groups"],
        )

        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])

        line_plot = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "linealpha": linealpha,
            "fill_between": fill_between,
            "fillalpha": fillalpha,
            "sort": sort,
            "marker": marker_dict,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "markersize": markersize,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("aggline", line_plot))

        if not self.inplace:
            return self

    def kde(
        self,
        kernel: KDEType = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        common_norm: bool = False,
        linecolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        fill_between: bool = False,
        alpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 1.0,
        kde_length: int | None = None,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        KDEType: Literal["tree", "fft"] = "fft",
    ):
        self._plot_methods.append("kde")
        self._plot_prefs.append(
            {
                "kernel": kernel,
                "bw": bw,
                "tol": tol,
                "common_norm": common_norm,
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "fill_between": fill_between,
                "fill_under": False,
                "alpha": alpha,
                "fillalpha": fillalpha,
                "kde_length": kde_length,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "KDEType": KDEType,
            }
        )

        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor = create_dict(
            linecolor,
            self._plot_dict["unique_groups"],
        )

        linestyle = create_dict(
            linestyle,
            self._plot_dict["unique_groups"],
        )

        kde_plot = {
            "linecolor": linecolor,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "linealpha": alpha,
            "fill_between": fill_between,
            "fill_under": False,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "err_func": err_func,
            "kde_length": kde_length,
            "KDEType": KDEType,
            "fillalpha": fillalpha,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("kde", kde_plot))

        if not self.inplace:
            return self

    def polyhist(
        self,
        color: ColorParameters = None,
        linestyle: str = "-",
        linewidth: int = 2,
        bin_limits=None,
        density=True,
        nbins=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
    ):
        self._plot_methods.append("polyhist")
        self._plot_pref.append(
            {
                "linestyle": linestyle,
                "linewidth": linewidth,
                "bin_limits": bin_limits,
                "density": density,
                "nbins": nbins,
                "func": func,
                "err_func": err_func,
                "fit_func": fit_func,
                "alpha": alpha,
                "unique_id": unique_id,
            }
        )
        color_dict = process_args(
            _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            ),
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linestyle_dict = process_args(
            linestyle, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )

        if bin_limits is not None and len(bin_limits) != 2:
            raise AttributeError("bin_limits must be length 2")

        poly_hist = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "density": density,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "unique_id": unique_id,
            "alpha": alpha,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("poly_hist", poly_hist))

        if not self.inplace:
            return self

    def hist(
        self,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        color: ColorParameters = "glasbey_category10",
        linecolor: ColorParameters = None,
        linewidth: float | int = 2,
        hatch=None,
        fillalpha: AlphaRange = 1.0,
        linealpha: float = 1.0,
        bin_limits=None,
        stat: Literal["density", "probability", "count"] = "density",
        nbins=50,
        err_func: Error = None,
        agg_func: Agg | None = None,
        unique_id=None,
    ):
        self._plot_methods.append("hist")
        self._plot_prefs.append(
            {
                "hist_type": hist_type,
                "color": color,
                "linecolor": linecolor,
                "linewidth": linewidth,
                "hatch": hatch,
                "bin_limits": bin_limits,
                "fillalpha": fillalpha,
                "linealpha": linealpha,
                "nbins": nbins,
                "err_func": err_func,
                "agg_func": agg_func,
                "stat": stat,
                "unique_id": unique_id,
            }
        )
        color = _process_colors(
            color,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])
        hatch_dict = create_dict(hatch, self._plot_dict["unique_groups"])
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(linecolor, self._plot_dict["unique_groups"])

        hist = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "linewidth": linewidth,
            "hatch_dict": hatch_dict,
            "stat": stat,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "agg_func": agg_func,
            "err_func": err_func,
            "unique_id": unique_id,
            "fillalpha": fillalpha,
            "linealpha": linealpha,
            "projection": self.plot_format["figure"]["projection"],
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("hist", hist))

        if self.plot_format["figure"]["projection"] == "polar":
            self.plot_format["grid"]["ygrid"] = True
            self.plot_format["grid"]["xgrid"] = True

        if not self.inplace:
            return self

    def ecdf(
        self,
        linecolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        linealpha: AlphaRange = 1.0,
        fill_between: bool = True,
        fillalpha: AlphaRange = 0.5,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        ecdf_type: Literal["spline", "bootstrap", "none"] = "none",
        ecdf_args=None,
    ):
        if ecdf_args is None and agg_func is not None:
            ecdf_args = {"size": 1000, "repititions": 1000, "seed": 42}
            ecdf_type = "bootstrap"
        self._plot_methods.append("ecdf")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "ecdf_type": ecdf_type,
                "agg_func": agg_func,
                "err_func": err_func,
                "ecdf_args": ecdf_args,
            }
        )
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(
            linecolor,
            self._plot_dict["unique_groups"],
        )

        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])

        ecdf = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "linealpha": linealpha,
            "unique_id": unique_id,
            "ecdf_type": ecdf_type,
            "ecdf_args": ecdf_args if ecdf_args is not None else {},
            "agg_func": agg_func,
            "err_func": err_func,
            "fillalpha": fillalpha,
            "fill_between": fill_between,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("ecdf", ecdf))

        self.plot_format["axis"]["ylim"] = [0.0, 1.0]

        if not self.inplace:
            return self

    def scatter(
        self,
        marker: str = ".",
        markercolor: ColorParameters | tuple[str, str] = "glasbey_category10",
        edgecolor: ColorParameters = "white",
        markersize: float | str = 36,
        linewidth: float = 1.5,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
    ):
        self._plot_methods.append("scatter")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markercolor": markercolor,
                "edgecolor": edgecolor,
                "markersize": markersize,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "linewidth": linewidth,
            }
        )

        if isinstance(markercolor, tuple):
            markercolor0 = markercolor[0]
            markercolor1 = markercolor[1]
        else:
            markercolor0 = markercolor
            markercolor1 = None

        if isinstance(edgecolor, tuple):
            edgecolor0 = edgecolor[0]
            edgecolor1 = edgecolor[1]
        else:
            edgecolor0 = edgecolor
            edgecolor1 = None

        colors = process_scatter_args(
            markercolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            markercolor1,
        )
        edgecolors = process_scatter_args(
            edgecolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            edgecolor1,
        )
        if isinstance(markersize, tuple):
            column = markersize[0]
            start, stop = markersize[1].split(":")
            start, stop = int(start) * 4, int(stop) * 4
            vmin = self.data.min(column)
            vmax = self.data.max(column)
            vals = self.data[column]
            markersize = (np.array(vals) - vmin) * (stop - start) / (
                vmax - vmin
            ) + start
        else:
            markersize = [markersize * 4] * self.data.shape[0]
        facetgroup = process_scatter_args(
            self._plot_dict["facet"],
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )
        scatter = {
            "markers": marker,
            "markercolors": colors,
            "edgecolors": edgecolors,
            "markersizes": markersize,
            "facetgroup": facetgroup,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "linewidth": linewidth,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("scatter", scatter))

        if not self.inplace:
            return self

    def fit(
        self,
        fit_func: Callable,
        linecolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
        agg_func: Agg = None,
        err_func: Error = None,
    ):
        self._plot_methods.append("fit")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "err_func": err_func,
                "fit_func": fit_func,
            }
        )

    def process_data(self):
        processor = LineProcessor(mpl.MARKERS, mpl.HATCHES)
        self.processed_data = processor(
            data=self.data,
            plot_list=self.plot_list,
            levels=self._plot_dict["levels"],
            y=self._plot_data["y"],
            x=self._plot_data["x"],
            facet_dict=self._plot_dict["facet_dict"],
            transforms=self._plot_transforms,
        )

    def _plot_processed_data(
        self,
        savefig: bool = False,
        path: SavePath = None,
        filename: str = "",
        filetype: str = "svg",
        **kwargs,
    ):
        self.plotter = mpl.LinePlotter(
            plot_data=self.processed_data,
            plot_dict=self._plot_dict,
            metadata=self.metadata(),
            savefig=savefig,
            path=path,
            filename=filename,
            filetype=filetype,
            **kwargs,
        )
        self.plotter.plot()
