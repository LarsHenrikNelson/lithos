from typing import Literal

import pandas as pd
from typing_extensions import Self

from ...types.basic_types import (
    BW,
    FitFunc,
    HistBinLimits,
    HistStat,
    HistType,
    InputData,
    KDEType,
    Kernels,
    NBins,
    SavePath,
)
from ...types.plot_input import (
    Agg,
    AlphaRange,
    ColorParameters,
    Error,
    Grouping,
    Subgrouping,
    UniqueGrouping,
)
from .. import matplotlib_plotter as mpl
from ..processing import LineProcessor
from .base_class import BasePlot


class LinePlot(BasePlot):
    ecdf_args = {
        "spline": {"size": 1000, "bc_type": "natural"},
        "bootstrap": {"size": 1000, "repititions": 1000, "seed": 42},
    }

    def __init__(self, data: InputData):
        super().__init__(data)

    def grouping(
        self,
        group: str | int | None = None,
        subgroup: str | int | None = None,
        group_order: Grouping = None,
        subgroup_order: Subgrouping = None,
        facet: bool = False,
        facet_title: bool = False,
        **kwargs,
    ) -> Self:
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "facet": facet,
            "facet_title": facet_title,
        }

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
        linewidth: float | int = 2,
        linealpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 0.5,
        unique_id: str | None = None,
        func: Agg | None = None,
        err_func: Error | None = None,
        index: str | None = None,
    ) -> Self:
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
    ) -> Self:
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
    ) -> Self:
        if fill_under and fill_between:
            raise ValueError("Cannot fill under and between at the same time")

        if kde_length is None and agg_func is not None:
            kde_length = 256

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

        return self

    def hist(
        self,
        hist_type: HistType = "bar",
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        linewidth: float | int = 2,
        hatch=None,
        fillalpha: AlphaRange = 0.5,
        linealpha: float = 1.0,
        bin_limits: HistBinLimits = None,
        stat: HistStat = "count",
        nbins: NBins = 50,
        err_func: Error = None,
        agg_func: Agg | None = None,
        unique_id=None,
    ) -> Self:
        if agg_func is not None and isinstance(nbins, str):
            raise ValueError("nbins must be int if agg_func is not None.")

        if isinstance(bin_limits, (tuple, list)):
            if bin_limits[1] < bin_limits[0]:
                raise ValueError("bin_limits[1] must be greater than bin_limits[0]")

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
    ) -> Self:
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

        return self

    def scatter(
        self,
        marker: str = ".",
        markercolor: ColorParameters | tuple[str, str] = "glasbey_category10",
        edgecolor: ColorParameters = "white",
        markersize: float | str | tuple[str | int, str] = 36,
        linewidth: float = 1.5,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
    ) -> Self:
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

        return self

    def fit(
        self,
        fit_func: FitFunc = "linear",
        linecolor: ColorParameters = "glasbey_category10",
        fillcolor: ColorParameters = "glasbey_category10",
        linestyle: str = "-",
        linewidth: int = 2,
        fillalpha: AlphaRange = 0.5,
        fill_between=True,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
        fit_args: dict | None = None,
        ci_func: Literal["ci", "pi", "bootstrap_ci"] = "ci",
        agg_func: Agg | None = "mean",
        err_func: Error = "sem",
    ) -> Self:
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

        return self

    def process_data(self):
        processor = LineProcessor(mpl.MARKERS, mpl.HATCHES)
        return processor(data=self.data, plot_metadata=self.metadata())

    def _plot_processed_data(
        self,
        savefig: bool = False,
        path: SavePath = "",
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
