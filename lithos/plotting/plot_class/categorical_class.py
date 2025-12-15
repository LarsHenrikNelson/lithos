import warnings
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import Self

from ...types.basic_types import (
    BW,
    BinType,
    CapStyle,
    CategoricalLabels,
    CountPlotTypes,
    InputData,
    JitterType,
    KDEType,
    Kernels,
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
from ..plot_utils import _process_colors, create_dict
from ..processing import CategoricalProcessor
from .base_class import BasePlot


class CategoricalPlot(BasePlot):
    def __init__(self, data: InputData):
        super().__init__(data)

    def grouping(
        self,
        group: str | int | float | None = None,
        subgroup: str | int | float | None = None,
        group_order: Grouping = None,
        subgroup_order: Subgrouping = None,
        group_spacing: float | int = 1.0,
        labels: CategoricalLabels = "style1",
        **kwargs,
    ) -> Self:
        if subgroup is None and labels in {"style2", "style3"}:
            raise ValueError("Cannot pass labels as style2, style3 if subgroup is None")
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "group_spacing": group_spacing,
            "labels": labels,
        }

        return self

    def jitter(
        self,
        markercolor: ColorParameters = "glasbey_category10",
        marker: str | dict[str | int, str] = "o",
        edgecolor: ColorParameters = "white",
        markeredgewidth: float | int = 1.0,
        jitter_type: JitterType = "fill",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange | None = None,
        width: float | int = 0.9,
        seed: int = 42,
        markersize: float = 5.0,
        unique_id: str | int | None = None,
        legend: bool = False,
    ) -> Self:
        self._plot_methods.append("jitter")
        self._plot_prefs.append(
            {
                "markercolor": markercolor,
                "markeredgewidth": markeredgewidth,
                "marker": marker,
                "edgecolor": edgecolor,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "markersize": markersize,
                "seed": seed,
                "unique_id": unique_id,
                "legend": legend,
                "jitter_type": jitter_type,
            }
        )

        return self

    def jitteru(
        self,
        unique_id: str | int | float,
        markercolor: ColorParameters = "glasbey_category10",
        marker: str | dict[str | int, str] = "o",
        edgecolor: ColorParameters = "none",
        markeredgewidth: float | int = 1.0,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange | None = None,
        width: float | int = 0.9,
        duplicate_offset=0.0,
        markersize: float = 5.0,
        agg_func: Agg | None = None,
        legend: bool = False,
    ) -> Self:
        self._plot_methods.append("jitteru")
        self._plot_prefs.append(
            {
                "unique_id": unique_id,
                "markercolor": markercolor,
                "markeredgewidth": markeredgewidth,
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
    ) -> Self:
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

        return self

    def summaryu(
        self,
        unique_id,
        func: Agg = "mean",
        agg_func: Agg | None = None,
        agg_width: float = 1.0,
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 0.9,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "glasbey_category10",
        alpha: float = 1.0,
        legend: bool = False,
    ) -> Self:
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

        return self

    def box(
        self,
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        fliers="",
        width: float = 0.9,
        linewidth=1,
        alpha: AlphaRange = 0.5,
        edge_alpha: AlphaRange = 1.0,
        showmeans: bool = False,
        show_ci: bool = False,
        legend: bool = False,
    ) -> Self:
        self._plot_methods.append("box")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "fliers": fliers,
                "width": width,
                "alpha": alpha,
                "linewidth": linewidth,
                "edge_alpha": edge_alpha,
                "showmeans": showmeans,
                "show_ci": show_ci,
                "legend": legend,
            }
        )

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
        unique_id: str | int | None = None,
        agg_func: Agg | None = None,
        kernel: Kernels = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        KDEType: KDEType = "fft",
        style: Literal["left", "right", "alternate", "full"] = "full",
        unique_style: Literal["split", "overlap"] = "overlap",
        legend: bool = False,
    ) -> Self:
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
        edge_alpha=1.0,
        axis_type: BinType = "density",
        include_bins: list[bool] | None = None,
        invert: bool = False,
        legend: bool = False,
    ) -> Self:
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
                "edge_alpha": edge_alpha,
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

        return self

    def bar(
        self,
        facecolor: ColorParameters = "glasbey_category10",
        edgecolor: ColorParameters = "glasbey_category10",
        hatch=None,
        barwidth: float = 0.9,
        linewidth=1,
        alpha: float = 0.5,
        edge_alpha=1.0,
        func: Agg = "mean",
        agg_func: Agg | None = None,
        unique_id: str | int | None = None,
        legend: bool = False,
    ) -> Self:
        self._plot_methods.append("bar")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "barwidth": barwidth,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "func": func,
                "legend": legend,
                "unique_id": unique_id,
                "agg_func": agg_func,
            }
        )

        return self

    def paired(
        self,
        unique_id: str | int,
        index: str | int,
        order: list[str | int] | tuple[str | int] | None = None,
        width: float | int = 0.9,
        marker: str = "o",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markeredgewidth: float = 1.0,
        markersize: float | str = 5,
        alpha: AlphaRange = 1.0,
        linecolor: ColorParameters = "glasbey_category10",
        linealpha: AlphaRange = 1.0,
        linestyle: str = "-",
        linewidth: int = 2,
        legend: bool = False,
        agg_func: Agg | None = None,
    ):
        self._plot_methods.append("paired")
        self._plot_prefs.append(
            {
                "marker": marker,
                "order": order,
                "width": width,
                "index": index,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markeredgewidth": markeredgewidth,
                "markersize": markersize,
                "alpha": alpha,
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "unique_id": unique_id,
                "linealpha": linealpha,
                "agg_func": agg_func,
            }
        )
        return self

    def process_data(self):
        processor = CategoricalProcessor(mpl.MARKERS, mpl.HATCHES)
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
