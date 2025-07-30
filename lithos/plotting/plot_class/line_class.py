from typing import Literal

import pandas as pd

from ..types import (
    BW,
    Agg,
    AlphaRange,
    ColorParameters,
    Error,
    KDEType,
    SavePath,
    FitFunc,
    Kernels,
    HistTypes
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
        marker: str = "none",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters | None = None,
        fill_between: bool = False,
        linestyle: str = "-",
        linewidth: int = 2,
        linealpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 0.5,
        unique_id: str | None = None,
        func: Agg | None = None,
        err_func: Error | None = None,
        index: str | None = None,
    ):
        self._plot_methods.append("line")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "fillcolor": fillcolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "unique_id": unique_id,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "func": func,
                "err_func": err_func,
                "index": index,
            }
        )

        if not self.inplace:
            return self

    def aggline(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters | None = None,
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float = 1.0,
        func: Agg = "mean",
        err_func: Error = "sem",
        agg_func: Agg | None = None,
        fill_between: bool = False,
        fillalpha: AlphaRange = 0.5,
        sort=True,
        unique_id=None,
    ):
        if fillcolor is None:
            fillcolor = linecolor
        self._plot_methods.append("aggline")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "fillcolor": fillcolor,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "linealpha": linealpha,
                "func": func,
                "err_func": err_func,
                "agg_func": agg_func,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "sort": sort,
                "unique_id": unique_id,
            }
        )

        if not self.inplace:
            return self

    def kde(
        self,
        kernel: Kernels = "gaussian",
        bw: BW = "silverman",
        tol: float | int | tuple = 1e-3,
        common_norm: bool = False,
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters | None = None,
        linestyle: str = "-",
        linewidth: int = 2,
        fill_under: bool = False,
        fill_between: bool = False,
        linealpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 1.0,
        kde_length: int | None = None,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        KDEType: KDEType = "fft",
    ):
        if fill_under and fill_between:
            raise AttributeError("Cannot fill under and between at the same time")
        if fillcolor is None:
            fillcolor = linecolor
        self._plot_methods.append("kde")
        self._plot_prefs.append(
            {
                "kernel": kernel,
                "bw": bw,
                "tol": tol,
                "common_norm": common_norm,
                "linecolor": linecolor,
                "fillcolor": fillcolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "fill_between": fill_between,
                "fill_under": fill_under,
                "linealpha": linealpha,
                "fillalpha": fillalpha,
                "kde_length": kde_length,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "err_func": err_func,
                "KDEType": KDEType,
            }
        )

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
        if bin_limits is not None and len(bin_limits) != 2:
            raise AttributeError("bin_limits must be length 2")
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

        if not self.inplace:
            return self

    def hist(
        self,
        hist_type: HistTypes = "bar",
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        linewidth: float | int = 2,
        hatch=None,
        fillalpha: AlphaRange = 0.5,
        linealpha: float = 1.0,
        bin_limits=None,
        stat: Literal["density", "probability", "count"] = "count",
        nbins=50,
        err_func: Error = None,
        agg_func: Agg | None = None,
        unique_id=None,
    ):
        self._plot_methods.append("hist")
        self._plot_prefs.append(
            {
                "hist_type": hist_type,
                "facecolor": facecolor,
                "edgecolor": edgecolor,
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

        if self.plot_format["figure"]["projection"] == "polar":
            self.plot_format["grid"]["ygrid"] = True
            self.plot_format["grid"]["xgrid"] = True

        if not self.inplace:
            return self

    def ecdf(
        self,
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters | None = None,
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
        else:
            ecdf_args
        if fillcolor is None:
            fillcolor = linecolor
        self._plot_methods.append("ecdf")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "fillcolor": fillcolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "ecdf_type": ecdf_type,
                "agg_func": agg_func,
                "err_func": err_func,
                "ecdf_args": ecdf_args,
                "unique_id": unique_id,
            }
        )

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

        if not self.inplace:
            return self

    def fit(
        self,
        fit_func: FitFunc= "linear",
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        fillalpha: AlphaRange = 0.5,
        fill_between=True,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
        fit_args: dict = None,
        ci_func: Literal["ci", "pi", "bootstrap_ci"] = "ci",
        agg_func: Agg = "mean",
        err_func: Error = "sem",
    ):
        self._plot_methods.append("fit")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "fillcolor": fillcolor,
                "fillalpha": fillalpha,
                "alpha": alpha,
                "unique_id": unique_id,
                "fit_func": fit_func,
                "fit_args": fit_args,
                "agg_func": agg_func,
                "err_func": err_func,
                "ci_func": ci_func,
                "fill_between": fill_between,
            }
        )

        if not self.inplace:
            return self

    def process_data(self):
        processor = LineProcessor(mpl.MARKERS, mpl.HATCHES)
        return processor(data=self.data, plot_metadata=self.metadata())

    def _plot_processed_data(
        self,
        savefig: bool = False,
        path: SavePath = None,
        filename: str = "",
        filetype: str = "svg",
        **kwargs,
    ):
        self.processed_data, plot_dict = self.process_data()
        self.plotter = mpl.LinePlotter(
            plot_data=self.processed_data,
            plot_dict=plot_dict,
            metadata=self.metadata(),
            savefig=savefig,
            path=path,
            filename=filename,
            filetype=filetype,
            **kwargs,
        )
        self.plotter.plot()
