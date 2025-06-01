from typing import Literal

import numpy as np
import pandas as pd

from ..plot_utils import _process_colors, create_dict
from .. import matplotlib_plotter as mpl
from ..types import (
    BW,
    Agg,
    AlphaRange,
    BinType,
    CapStyle,
    ColorParameters,
    CountPlotTypes,
    Error,
    KDEType,
    SavePath,
)
from .base_class import BasePlot
from ..processing import CategoricalProcessor


class CategoricalPlot(BasePlot):
    def __init__(self, data: pd.DataFrame | np.ndarray | dict, inplace: bool = False):
        super().__init__(data, inplace)

        if not self.inplace:
            self.inplace = True
            self.grouping()
            self.inplace = False
        else:
            self.grouping()

    def grouping(
        self,
        group: str | int | float = None,
        subgroup: str | int | float = None,
        group_order: list[str | int | float] | None = None,
        subgroup_order: list[str | int | float] | None = None,
        group_spacing: float | int = 1.0,
    ):
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "group_spacing": group_spacing,
        }

        if not self.inplace:
            return self

    def jitter(
        self,
        markercolor: ColorParameters = "glasbey_category10",
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "white",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 0.9,
        seed: int = 42,
        markersize: float = 5.0,
        unique_id: str | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitter")
        self._plot_prefs.append(
            {
                "markercolor": markercolor,
                "marker": marker,
                "edgecolor": edgecolor,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "markersize": markersize,
                "seed": seed,
                "unique_id": unique_id,
                "legend": legend,
            }
        )

        if not self.inplace:
            return self

    def jitteru(
        self,
        unique_id: str | int | float,
        markercolor: ColorParameters = "glasbey_category10",
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 0.9,
        duplicate_offset=0.0,
        markersize: float = 5.0,
        agg_func: Agg | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitteru")
        self._plot_prefs.append(
            {
                "unique_id": unique_id,
                "markercolor": markercolor,
                "marker": marker,
                "edgecolor": edgecolor,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "duplicate_offset": duplicate_offset,
                "markersize": markersize,
                "agg_func": agg_func,
                "legend": legend,
            }
        )

        if not self.inplace:
            return self

    def summary(
        self,
        func: Agg = "mean",
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 0.9,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_methods.append("summary")
        self._plot_prefs.append(
            {
                "func": func,
                "capsize": capsize,
                "capstyle": capstyle,
                "barwidth": barwidth,
                "err_func": err_func,
                "linewidth": linewidth,
                "color": color,
                "alpha": alpha,
                "legend": legend,
            }
        )

        if not self.inplace:
            return self

    def summaryu(
        self,
        unique_id,
        func: Agg = "mean",
        agg_func: Agg = None,
        agg_width: float = 1.0,
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 0.9,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "glasbey_category10",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_methods.append("summaryu")
        self._plot_prefs.append(
            {
                "func": func,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "agg_width": agg_width,
                "capsize": capsize,
                "capstyle": capstyle,
                "barwidth": barwidth,
                "err_func": err_func,
                "linewidth": linewidth,
                "color": color,
                "alpha": alpha,
                "legend": legend,
            }
        )

        if not self.inplace:
            return self

    def box(
        self,
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        fliers="",
        width: float = 0.9,
        linewidth=1,
        alpha: AlphaRange = 0.5,
        linealpha: AlphaRange = 1.0,
        showmeans: bool = False,
        show_ci: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("box")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "fliers": fliers,
                "width": width,
                "alpha": alpha,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "showmeans": showmeans,
                "show_ci": show_ci,
                "legend": legend,
            }
        )

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        linewidth=1,
        alpha: AlphaRange = 0.5,
        edge_alpha: AlphaRange = 1.0,
        width: float = 0.9,
        kde_length: int = 128,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        kernel: KDEType = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        KDEType: Literal["tree", "fft"] = "fft",
        style: Literal["left", "right", "alternate", "full"] = "full",
        unique_style: Literal["split", "overlap"] = "overlap",
        legend: bool = False,
    ):
        if unique_id is not None and agg_func is None:
            style = "full"
        self._plot_methods.append("violin")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "legend": legend,
                "kde_length": kde_length,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "KDEType": KDEType,
                "kernel": kernel,
                "bw": bw,
                "tol": tol,
                "style": style,
                "unique_style": unique_style,
            }
        )

        if not self.inplace:
            return self

    def percent(
        self,
        cutoff: None | float | int | list[float | int] = None,
        unique_id=None,
        facecolor="glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        hatch: bool = False,
        barwidth: float = 0.9,
        linewidth=1,
        alpha: float = 0.5,
        linealpha=1.0,
        axis_type: BinType = "density",
        include_bins: list[bool] | None = None,
        invert: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("percent")
        if isinstance(cutoff, (float, int)):
            cutoff = [cutoff]
        self._plot_prefs.append(
            {
                "cutoff": cutoff,
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "linewidth": linewidth,
                "barwidth": barwidth,
                "alpha": alpha,
                "linealpha": linealpha,
                "axis_type": axis_type,
                "invert": invert,
                "include_bins": include_bins,
                "unique_id": unique_id,
                "legend": legend,
            }
        )

        if axis_type == "density":
            self.plot_format["axis"]["ylim"] = [0.0, 1.0]
        else:
            self.plot_format["axis"]["ylim"] = [0, 100]

        if not self.inplace:
            return self

    def bar(
        self,
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        hatch=None,
        barwidth: float = 0.9,
        linewidth=1,
        alpha: float = 0.5,
        linealpha=1.0,
        func: Agg = "mean",
        agg_func: Agg | None = None,
        unique_id: str | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("bar")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "barwidth": barwidth,
                "linewidth": linewidth,
                "alpha": alpha,
                "linealpha": linealpha,
                "func": func,
                "legend": legend,
                "unique_id": unique_id,
                "agg_func": agg_func,
            }
        )

        if not self.inplace:
            return self

    def process_data(self):
        processor = CategoricalProcessor(mpl.MARKERS, mpl.HATCHES)
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
        self.plotter = mpl.CategoricalPlotter(
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
