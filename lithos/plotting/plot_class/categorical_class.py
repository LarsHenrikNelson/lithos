from typing import Literal

import numpy as np
import pandas as pd

from ..plot_utils import _process_colors, create_dict
from . import matplotlib_plotter as mpl
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
        color: ColorParameters = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        seed: int = 42,
        markersize: float = 2.0,
        unique_id: str | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitter")
        self._plot_prefs.append(
            {
                "color": color,
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
        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor == "color":
            edgecolor_dict = color_dict
        else:
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        jitter_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "seed": seed,
            "markersize": markersize,
            "unique_id": unique_id,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("jitter", jitter_plot))

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def jitteru(
        self,
        unique_id: str | int | float,
        color: ColorParameters = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        duplicate_offset=0.0,
        markersize: float = 2.0,
        agg_func: Agg | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitteru")
        self._plot_prefs.append(
            {
                "unique_id": unique_id,
                "color": color,
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
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor is None:
            edgecolor_dict = color_dict
        else:
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        if edge_alpha is None:
            edge_alpha = alpha

        jitteru_plot = {
            "color_dict": color_dict,
            "marker": marker,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "markersize": markersize,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("jitteru", jitteru_plot))

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def summary(
        self,
        func: Agg = "mean",
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 1.0,
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
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        summary_plot = {
            "func": func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self._plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("summary", summary_plot))

        if legend:
            d = _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

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
        barwidth: float = 1.0,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "black",
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
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        summary_plot = {
            "func": func,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self._plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
            "agg_width": agg_width,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("summaryu", summary_plot))

        if legend:
            d = _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def box(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = None,
        fliers="",
        width: float = 1.0,
        linewidth=1,
        alpha: AlphaRange = 1.0,
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
        if facecolor != "edgecolor":
            color = _process_colors(
                facecolor,
                self._plot_dict["group_order"],
                self._plot_dict["subgroup_order"],
            )
            color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor != "facecolor":
            edgecolor = _process_colors(
                edgecolor,
                self._plot_dict["group_order"],
                self._plot_dict["subgroup_order"],
            )
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        if facecolor == "edgecolor":
            color_dict = edgecolor_dict

        if edgecolor == "facecolor":
            edgecolor_dict = color_dict

        box = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "fliers": fliers,
            "width": width * self._plot_dict["width"],
            "showmeans": showmeans,
            "show_ci": show_ci,
            "linewidth": linewidth,
            "alpha": alpha,
            "linealpha": linealpha,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("box", box))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = None,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
        width: float = 1.0,
        kde_length: int = 128,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        kernel: KDEType = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        KDEType: Literal["tree", "fft"] = "fft",
        unique_style: Literal["split", "overlap"] = "overlap",
        legend: bool = False,
    ):
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
                "unique_style": unique_style,
            }
        )
        color = _process_colors(
            facecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])
        edgecolor = _process_colors(
            edgecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        edge_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])
        violin = {
            "facecolor_dict": color_dict,
            "edgecolor_dict": edge_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "linewidth": linewidth,
            "kde_length": kde_length,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "KDEType": KDEType,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "unique_style": unique_style,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("violin", violin))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def percent(
        self,
        cutoff: None | float | int | list[float | int] = None,
        unique_id=None,
        facecolor=None,
        edgecolor: ColorParameters = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        linealpha=1.0,
        axis_type: BinType = "density",
        include_bins: list[bool] | None = None,
        invert: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("percent")
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
                "legend": legend,
            }
        )
        if isinstance(cutoff, (float, int)):
            cutoff = [cutoff]
        facecolor = _process_colors(
            facecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        color_dict = create_dict(facecolor, self._plot_dict["unique_groups"])

        edgecolor = _process_colors(
            edgecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        percent_plot = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "cutoff": cutoff,
            "hatch": hatch,
            "barwidth": barwidth * self._plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "linealpha": linealpha,
            "include_bins": include_bins,
            "unique_id": unique_id,
            "invert": invert,
            "axis_type": axis_type,
            "zorder_dict": self._set_zorder(),
        }

        self.plot_list.append(("percent", percent_plot))

        if axis_type == "density":
            self.plot_format["axis"]["ylim"] = [0.0, 1.0]
        else:
            self.plot_format["axis"]["ylim"] = [0, 100]

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def count(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        edge_alpha=1.0,
        axis_type: CountPlotTypes = "count",
        legend: bool = False,
    ):
        self._plot_methods.append("count")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "barwidth": barwidth,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "axis_type": axis_type,
                "legend": legend,
            }
        )
        facecolor = _process_colors(
            facecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(facecolor, self._plot_dict["unique_groups"])

        edgecolor = _process_colors(
            edgecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        count_plot = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "hatch": hatch,
            "barwidth": barwidth * self._plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "axis_type": axis_type,
            "zorder_dict": self._set_zorder(),
        }
        self.plot_list.append(("count", count_plot))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def process_data(self):
        processor = CategoricalProcessor(mpl.MARKERS, mpl.HATCHES)
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
        self.plotter = mpl.CategoricalPlotter(
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
