from typing import Callable, Literal

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
    FillType,
    MarkerLine,
    FillBetweenLine,
    FillUnderLine,
    Line,
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
        style: Line | MarkerLine | FillBetweenLine = Line(),
        unique_id: str | None = None,
        index: str | None = None,
    ):
        self._plot_methods.append("line")
        prefs = {
            "unique_id": unique_id,
            "index": index,
        }
        prefs["style"] = style._asdict()
        self._plot_prefs.append(prefs)

        if not self.inplace:
            return self

    def aggline(
        self,
        style: Line | MarkerLine | FillBetweenLine | None = None,
        func: Agg = "mean",
        sort=True,
        unique_id=None,
    ):
        self._plot_methods.append("aggline")
        if style is None:
            style = FillBetweenLine()
        prefs = {
            "func": func,
            "style": style,
            "sort": sort,
            "unique_id": unique_id,
        }
        prefs["style"] = style._asdict()
        self._plot_prefs.append(prefs)

        if not self.inplace:
            return self

    def kde(
        self,
        kernel: Kernels = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        common_norm: bool = False,
        style: Line | MarkerLine | FillBetweenLine | FillUnderLine | None = None,
        kde_length: int | None = None,
        unique_id: str | None = None,
        KDEType: KDEType = "fft",
    ):
        self._plot_methods.append("kde")
        if style is None:
            style = FillUnderLine()
        prefs = {
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "kde_length": kde_length,
            "unique_id": unique_id,
            "KDEType": KDEType,
        }
        prefs["style"] = style._asdict()
        self._plot_prefs.append(prefs)

        if not self.inplace:
            return self

    def hist(
        self,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        linewidth: float | int = 2,
        hatch=None,
        fillalpha: AlphaRange = 0.5,
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
        style: Line | MarkerLine | FillBetweenLine | None = None,
        unique_id: str | None = None,
        ecdf_type: Literal["spline", "bootstrap", "none"] = "none",
        ecdf_args=None,
    ):
        if ecdf_args is None:
            ecdf_args = {"size": 1000, "repititions": 1000, "seed": 42}
        else:
            ecdf_args
        self._plot_methods.append("ecdf")
        if style is None:
            style = Line()
        prefs = {
            "ecdf_type": ecdf_type,
            "ecdf_args": ecdf_args,
            "unique_id": unique_id,
        }
        prefs["style"] = style._asdict()
        self._plot_prefs.append()

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
        fit_func: FitFunc,
        style: Line | MarkerLine | FillBetweenLine | None = None,
        unique_id: str | None = None,
        fit_args: dict = None,
    ):
        self._plot_methods.append("fit")
        if style is None:
            style = Line()
        prefs = {
            "unique_id": unique_id,
            "fit_func": fit_func,
            "fit_args": fit_args,
        }
        prefs["style"] = style._asdict()
        self._plot_prefs.append()

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
