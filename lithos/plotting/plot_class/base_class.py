from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import Self

from ...types.basic_types import (
    InputData,
    SavePath,
    Transform,
)
from ...types.plot_input import Agg, Error, Grouping, Subgrouping
from ...utils import (
    DataHolder,
    metadata_utils,
)


class BasePlot:
    aggregating_funcs = Agg
    error_funcs = Error
    transform_funcs = Transform

    def __init__(self, data: InputData):
        self.plot_list = []
        self._plot_methods = []
        self._plot_prefs = []
        self._grouping = {}
        self.data = DataHolder(data)
        self.processed_data = []
        self.plotter = None

        self.plot_format = {}
        self._plot_data = {}

        self.grouping()
        self.labels()
        self.axis()
        self.axis_format()
        self.figure()
        self.grid()
        self.transform()

    def grouping(
        self,
        group: str | int | None = None,
        subgroup: str | int | None = None,
        group_order: Grouping = None,
        subgroup_order: Subgrouping = None,
        **kwargs,
    ):
        raise NotImplementedError("Must implement grouping method.")

    def add_axline(
        self,
        linetype: Literal["hline", "vline"],
        lines: list,
        linestyle="solid",
        linealpha=1,
        linecolor="black",
        linewidth=1.5,
        zorder=1,
    ) -> Self:
        if linetype not in ["hline", "vline"]:
            raise AttributeError("linetype must by hline or vline")
        if isinstance(lines, (float, int)):
            lines = [lines]
        self.plot_format[linetype] = {
            "linetype": linetype,
            "lines": lines,
            "linestyle": linestyle,
            "linealpha": linealpha,
            "linecolor": linecolor,
            "linewidth": linewidth,
            "zorder": zorder,
        }

        return self

    def labels(
        self,
        labelsize: float = 20,
        titlesize: float = 22,
        ticklabel_size: int = 12,
        font: str = "DejaVu Sans",
        fontweight: None | str | float = None,
        title_fontweight: str | float = "regular",
        label_fontweight: str | float = "regular",
        tick_fontweight: str | float = "regular",
        xlabel_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ylabel_rotation: Literal["horizontal", "vertical"] | float = "vertical",
        xtick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ytick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
    ) -> Self:
        if fontweight is not None:
            title_fontweight = fontweight
            label_fontweight = fontweight
            tick_fontweight = fontweight

        label_props = {
            "labelsize": labelsize,
            "titlesize": titlesize,
            "font": font,
            "ticklabel_size": ticklabel_size,
            "title_fontweight": title_fontweight,
            "label_fontweight": label_fontweight,
            "tick_fontweight": tick_fontweight,
            "xlabel_rotation": xlabel_rotation,
            "ylabel_rotation": ylabel_rotation,
            "xtick_rotation": xtick_rotation,
            "ytick_rotation": ytick_rotation,
        }
        self.plot_format["labels"] = label_props
        return self

    def axis(
        self,
        ylim: list | tuple | None = None,
        xlim: list | tuple | None = None,
        yaxis_lim: list | tuple | None = None,
        xaxis_lim: list | tuple | None = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        ydecimals: int | None = None,
        xdecimals: int | None = None,
        xformat: Literal["f", "e"] = "f",
        yformat: Literal["f", "e"] = "f",
        yunits: Literal["degree", "radian", "wradian"] | None = None,
        xunits: Literal["degree", "radian", "wradian"] | None = None,
    ) -> Self:
        if ylim is None:
            ylim = (None, None)
        if xlim is None:
            xlim = (None, None)

        axis_settings = {
            "yscale": yscale,
            "xscale": xscale,
            "ylim": ylim,
            "xlim": xlim,
            "yaxis_lim": yaxis_lim,
            "xaxis_lim": xaxis_lim,
            "ydecimals": ydecimals,
            "xdecimals": xdecimals,
            "xunits": xunits,
            "yunits": yunits,
            "xformat": xformat,
            "yformat": yformat,
        }
        self.plot_format["axis"] = axis_settings

        return self

    def axis_format(
        self,
        linewidth: float | dict[str, float] = 2,
        tickwidth: float = 2,
        ticklength: float = 5.0,
        minor_tickwidth: float = 1.5,
        minor_ticklength: float = 2.5,
        yminorticks: int = 0,
        xminorticks: int = 0,
        ysteps: int | tuple[int, int, int] = 5,
        xsteps: int | tuple[int, int, int] = 5,
        truncate_xaxis: bool = False,
        truncate_yaxis: bool = False,
        style: Literal["default", "lithos"] = "lithos",
    ) -> Self:
        if isinstance(ysteps, int):
            ysteps = (ysteps, 0, ysteps)
        if isinstance(xsteps, int):
            xsteps = (xsteps, 0, xsteps)
        if isinstance(linewidth, (int, float)):
            linewidth = {"left": linewidth, "bottom": linewidth, "top": 0, "right": 0}
        elif isinstance(linewidth, dict):
            temp_lw = {"left": 0, "bottom": 0, "top": 0, "right": 0}
            for key, value in linewidth:
                temp_lw[key] = value
            linewidth = temp_lw

        axis_format = {
            "tickwidth": tickwidth,
            "ticklength": ticklength,
            "linewidth": linewidth,
            "minor_tickwidth": minor_tickwidth,
            "minor_ticklength": minor_ticklength,
            "yminorticks": yminorticks,
            "xminorticks": xminorticks,
            "xsteps": xsteps,
            "ysteps": ysteps,
            "style": style,
            "truncate_xaxis": truncate_xaxis,
            "truncate_yaxis": truncate_yaxis,
        }

        self.plot_format["axis_format"] = axis_format

        return self

    def figure(
        self,
        margins=0.05,
        aspect: int | float | None = None,
        figsize: None | tuple[int, int] = None,
        gridspec_kw: dict[str, str | int | float] | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        projection: Literal["rectilinear", "polar"] = "rectilinear",
    ) -> Self:
        figure = {
            "gridspec_kw": gridspec_kw,
            "margins": margins,
            "aspect": aspect if projection == "rectilinear" else None,
            "figsize": figsize,
            "nrows": nrows,
            "ncols": ncols,
            "projection": projection,
        }

        self.plot_format["figure"] = figure

        return self

    def grid(
        self,
        ygrid: int | float = 0,
        xgrid: int | float = 0,
        yminor_grid: int | float = 0,
        xminor_grid: int | float = 0,
        linestyle: str | tuple = "solid",
        minor_linestyle: str | tuple = "solid",
    ) -> Self:
        grid = {
            "ygrid": ygrid,
            "xgrid": xgrid,
            "yminor_grid": yminor_grid,
            "xminor_grid": xminor_grid,
            "linestyle": linestyle,
            "minor_linestyle": minor_linestyle,
        }
        self.plot_format["grid"] = grid

        return self

    def clear_plots(self) -> Self:
        self._plot_methods = []
        self._plot_prefs = []

        return self

    def plot(
        self,
        savefig: bool = False,
        path: SavePath = "",
        filename: str = "",
        filetype: str = "svg",
        backend: str = "matplotlib",
        save_metadata: bool = False,
        **kwargs,
    ) -> Self:
        if path == "" or path is None:
            path = Path().cwd()
        elif isinstance(path, str):
            path = Path(path)
        filename_output = self._plot_data["y"] if filename == "" else filename
        filename_output = filename_output if filename_output is not None else ""
        self._plot_processed_data(savefig, path, filename_output, filetype, **kwargs)
        if save_metadata and isinstance(path, (Path)):
            path = path / f"{filename}.txt"
            self.save_metadata(path)

        return self

    def transform(
        self,
        ytransform: Transform | None = None,
        back_transform_yticks: bool = False,
        xtransform: Transform | None = None,
        back_transform_xticks: bool = False,
    ) -> Self:
        self._plot_transforms = {}
        self._plot_transforms["ytransform"] = ytransform
        if callable(ytransform):
            self._plot_transforms["back_transform_yticks"] = False
        else:
            self._plot_transforms["back_transform_yticks"] = back_transform_yticks

        self._plot_transforms["xtransform"] = xtransform
        if callable(xtransform):
            self._plot_transforms["back_transform_xticks"] = False
        else:
            self._plot_transforms["back_transform_xticks"] = back_transform_xticks

        return self

    def get_format(self) -> dict:
        return self.plot_format

    def plot_data(
        self,
        y: str | None = None,
        x: str | None = None,
        ylabel: str = "",
        xlabel: str = "",
        title: str = "",
        figure_title: str = "",
    ) -> Self:
        if x is None and y is None:
            raise ValueError("Must specify either x or y")

        self._plot_data = {
            "y": y,
            "x": x,
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
            "figure_title": figure_title,
        }

        return self

    def metadata(self):
        output = {
            "grouping": self._grouping,
            "data": self._plot_data,
            "format": self.plot_format,
            "transforms": self._plot_transforms,
            "plot_methods": self._plot_methods,
            "plot_prefs": self._plot_prefs,
        }
        return output

    def save_metadata(self, file_path: str | Path):
        metadata = self.metadata()
        metadata_utils.save_metadata(metadata, file_path)

    def _load_plot_prefs(self, plot_dict: dict, meta_dict: dict):
        for key, value in meta_dict.items():
            if key in plot_dict:
                plot_dict[key] = value

    def _set_metadata_from_dict(self, metadata: dict) -> Self:
        self._plot_data = metadata["data"]
        for key in metadata["format"]:
            self._load_plot_prefs(self.plot_format[key], metadata["format"][key])
        self._load_plot_prefs(self._plot_transforms, metadata["transforms"])

        # Not super happy with this code but it works for not
        self = self.grouping(**metadata["grouping"])
        # Must loop through metadata and not set class variables otherwise it will
        # overwrite the instance and get stuck before the loop even runs.
        for pfunc, ppref in zip(metadata["plot_methods"], metadata["plot_prefs"]):
            method = getattr(self, pfunc)
            self = method(**ppref)
        return self

    def load_metadata(self, metadata_path: str | dict | Path) -> Self:
        metadata = metadata_utils.load_metadata(metadata_path)
        self = self._set_metadata_from_dict(metadata)
        return self

    def set_metadata_directory(self, metadata_dir: str | Path):
        metadata_utils.set_metadata_dir(metadata_dir)

    def _plot_processed_data(
        self,
        savefig: bool = False,
        path: SavePath = "",
        filename: str = "",
        filetype: str = "svg",
        **kwargs,
    ):
        raise NotImplementedError("Subclasses must implement _plot_processed_data()")


class GraphPlot:
    def __init__(self, graph):
        self._plot_dict = {}
        self._plot_dict["graph"] = graph
        self.plots = []

    def graphplot(
        self,
        marker_alpha: float = 0.8,
        linealpha: float = 0.1,
        markersize: int = 2,
        markerscale: int = 1,
        linewidth: int = 1,
        edgecolor: str = "k",
        markercolor: str = "red",
        marker_attr: str | None = None,
        cmap: str = "gray",
        seed: int = 42,
        scale: int = 50,
        plot_max_degree: bool = False,
        layout: Literal["spring", "circular", "communities"] = "spring",
    ):
        graph_plot = {
            "marker_alpha": marker_alpha,
            "linealpha": linealpha,
            "markersize": markersize,
            "markerscale": markerscale,
            "linewidth": linewidth,
            "edgecolor": edgecolor,
            "markercolor": markercolor,
            "marker_attr": marker_attr,
            "cmap": cmap,
            "seed": seed,
            "scale": scale,
            "layout": layout,
            "plot_max_degree": plot_max_degree,
        }
        self.plots.append(graph_plot)
