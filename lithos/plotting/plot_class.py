from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import (
    AGGREGATE,
    ERROR,
    TRANSFORM,
    DataHolder,
    get_backtransform,
    get_transform,
    metadata_utils,
)
from . import matplotlib_plotting as mp
from .plot_utils import (
    _decimals,
    _process_colors,
    _process_positions,
    create_dict,
    get_ticks,
    process_args,
    process_scatter_args,
    radian_ticks,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


@dataclass
class ValueRange:
    lo: float
    hi: float


AlphaRange = Annotated[float, ValueRange(0.0, 1.0)]
ColorParameter = str | dict[str, str] | None

MP_PLOTS = {
    "boxplot": mp._boxplot,
    "hist": mp._hist_plot,
    "jitter": mp._jitter_plot,
    "jitteru": mp._jitteru_plot,
    "line_plot": mp._line_plot,
    "poly_hist": mp._poly_hist,
    "summary": mp._summary_plot,
    "summaryu": mp._summaryu_plot,
    "violin": mp._violin_plot,
    "kde": mp._kde_plot,
    "percent": mp._percent_plot,
    "ecdf": mp._ecdf,
    "count": mp._count_plot,
    "scatter": mp._scatter_plot,
    "aggline": mp._agg_line,
}

MPL_SAVE_TYPES = {"svg", "png", "jpeg"}
PLOTLY_SAVE_TYPES = {"html"}


class BasePlot:
    aggregating_funcs = AGGREGATE
    error_funcs = ERROR
    transform_funcs = TRANSFORM

    def __init__(self, data: pd.DataFrame, inplace: bool = False):
        self.inplace = inplace
        self.plots = []
        self.plot_list = []
        self._plot_funcs = []
        self._plot_prefs = []
        self._grouping = {}
        self._plot_settings_run = False
        self.data = DataHolder(data)

        plt.rcParams["svg.fonttype"] = "none"

        self.plot_format = {}
        self._plot_dict = {}
        self._plot_data = {}

        if not self.inplace:
            self.inplace = True
            self.labels()
            self.axis()
            self.axis_format()
            self.figure()
            self.grid_settings()
            self.transform()
            self.inplace = False
        else:
            self.labels()
            self.axis()
            self.axis_format()
            self.figure()
            self.grid_settings()
            self.transform()

    def _set_grid(self, sub_ax):
        if self.plot_format["grid"]["ygrid"]:
            sub_ax.yaxis.grid(
                linewidth=self.plot_format["grid"]["ylinewidth"],
                linestyle=self.plot_format["grid"]["linestyle"],
            )

        if self.plot_format["grid"]["xgrid"]:
            sub_ax.xaxis.grid(
                linewidth=self.plot_format["grid"]["xlinewidth"],
                linestyle=self.plot_format["grid"]["linestyle"],
            )

    def add_axline(
        self,
        linetype: Literal["hline", "vline"],
        lines: list,
        linestyle="solid",
        linealpha=1,
        linecolor="black",
    ):
        if linetype not in ["hline", "vline"]:
            raise AttributeError("linetype must by hline or vline")
        if isinstance(lines, (float, int)):
            lines = [lines]
        self._plot_dict[linetype] = {
            "linetype": linetype,
            "lines": lines,
            "linestyle": linestyle,
            "linealpha": linealpha,
            "linecolor": linecolor,
        }

        if not self.inplace:
            return self

    def _plot_axlines(self, line_dict, ax):
        for ll in line_dict["lines"]:
            if line_dict["linetype"] == "vline":
                ax.axvline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                )
            else:
                ax.axhline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                )

    def _set_lims(self, ax, decimals, axis="x"):
        if axis == "y":
            if self.plot_format["axis"]["yscale"] not in ["log", "symlog"]:
                ticks = ax.get_yticks()
                lim, _, ticks = get_ticks(
                    lim=self.plot_format["axis"]["ylim"],
                    axis_lim=self.plot_format["axis"]["yaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["ysteps"],
                )
                ax.set_ylim(bottom=lim[0], top=lim[1])
                if (
                    "back_transform_yticks" in self._plot_transforms
                    and self._plot_transforms["back_transform_yticks"]
                ):
                    tick_labels = get_backtransform(
                        self._plot_transforms["ytransform"]
                    )(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        # This does not work with scientific format
                        tick_labels = np.round(tick_labels, decimals=decimals)
                        dformat = self.plot_format["axis"]["yformat"]
                        tick_labels = [
                            f"{value:.{decimals}{dformat}}" for value in tick_labels
                        ]
                ax.set_yticks(
                    ticks,
                    labels=tick_labels,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
                truncate = (
                    self.plot_format["axis_format"]["ysteps"][1] != 0
                    or self.plot_format["axis_format"]["ysteps"][2]
                    != self.plot_format["axis_format"]["ysteps"][0]
                )
                if truncate or self.plot_format["axis"]["yaxis_lim"] is not None:
                    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
            else:
                ax.set_yscale(self.plot_format["axis"]["yscale"])
                ticks = ax.get_yticks()
                lim, _, _ = get_ticks(
                    lim=self.plot_format["axis"]["ylim"],
                    axis_lim=self.plot_format["axis"]["yaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["ysteps"],
                )
                ax.set_ylim(bottom=lim[0], top=lim[1])
        else:
            if self.plot_format["axis"]["xscale"] not in ["log", "symlog"]:
                ticks = ax.get_xticks()
                lim, _, ticks = get_ticks(
                    lim=self.plot_format["axis"]["xlim"],
                    axis_lim=self.plot_format["axis"]["xaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["xsteps"],
                )
                ax.set_xlim(left=lim[0], right=lim[1])
                if (
                    "back_transform_xticks" in self._plot_transforms
                    and self._plot_transforms["back_transform_xticks"]
                ):
                    tick_labels = get_backtransform(
                        self._plot_transforms["xtransform"]
                    )(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        # This does not work with scientific format
                        tick_labels = np.round(tick_labels, decimals=decimals)
                        dformat = self.plot_format["axis"]["xformat"]
                        tick_labels = [
                            f"{value:.{decimals}{dformat}}" for value in tick_labels
                        ]
                ax.set_xticks(
                    ticks,
                    labels=tick_labels,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["xtick_rotation"],
                )
                truncate = (
                    self.plot_format["axis_format"]["xsteps"][1] != 0
                    or self.plot_format["axis_format"]["xsteps"][2]
                    != self.plot_format["axis_format"]["xsteps"][0]
                )
                if truncate or self.plot_format["axis"]["xaxis_lim"] is not None:
                    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
            else:
                ax.set_xscale(self.plot_format["axis"]["xscale"])
                ticks = ax.get_xticks()
                lim, _, _ = get_ticks(
                    lim=self.plot_format["axis"]["xlim"],
                    axis_lim=self.plot_format["axis"]["xaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["xsteps"],
                )
                ax.set_xlim(left=lim[0], right=lim[1])

    def _set_minorticks(self, ax, nticks, transform: str, ticks: Literal["y", "x"]):
        if ticks == "y":
            yticks = ax.get_yticks()
        else:
            yticks = ax.get_xticks()
        yticks = get_backtransform(transform)(yticks)
        mticks = np.zeros((len(yticks) - 1) * nticks)
        for index in range(yticks.size - 1):
            vals = np.linspace(
                yticks[index], yticks[index + 1], num=nticks, endpoint=False
            )
            start = index * nticks
            end = index * nticks + nticks
            mticks[start:end] = vals
        if ticks == "y":
            ax.set_yticks(
                get_transform(transform)(mticks),
                minor=True,
            )
            ax.tick_params(
                axis="y",
                which="minor",
                width=self.plot_format["axis_format"]["minor_tickwidth"],
                length=self.plot_format["axis_format"]["minor_ticklength"],
                labelfontfamily=self.plot_format["labels"]["font"],
            )

        else:
            ax.set_xticks(
                get_transform(transform)(mticks),
                minor=True,
            )
            ax.tick_params(
                axis="x",
                which="minor",
                width=self.plot_format["axis_format"]["minor_tickwidth"],
                length=self.plot_format["axis_format"]["minor_ticklength"],
                labelfontfamily=self.plot_format["labels"]["font"],
            )

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
    ):
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
        if not self.inplace:
            return self

    def axis(
        self,
        ylim: list | None = None,
        xlim: list | None = None,
        yaxis_lim: list | None = None,
        xaxis_lim: list | None = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        ydecimals: int = None,
        xdecimals: int = None,
        xformat: Literal["f", "e"] = "f",
        yformat: Literal["f", "e"] = "f",
        yunits: Literal["degree", "radian" "wradian"] | None = None,
        xunits: Literal["degree", "radian" "wradian"] | None = None,
    ):
        if ylim is None:
            ylim = [None, None]
        if xlim is None:
            xlim = [None, None]

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

        if not self.inplace:
            return self

    def axis_format(
        self,
        linewidth: float = 2,
        tickwidth: float = 2,
        ticklength: float = 5.0,
        minor_tickwidth: float = 1.5,
        minor_ticklength: float = 2.5,
        yminorticks: int = 0,
        xminorticks: int = 0,
        ysteps: int | tuple[int, int, int] = 5,
        xsteps: int | tuple[int, int, int] = 5,
    ):
        if isinstance(ysteps, int):
            ysteps = (ysteps, 0, ysteps)
        if isinstance(xsteps, int):
            xsteps = (xsteps, 0, xsteps)
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
        }

        self.plot_format["axis_format"] = axis_format

        if not self.inplace:
            return self

    def figure(
        self,
        margins=0.05,
        aspect: int | float = 1.0,
        figsize: None | tuple[int, int] = None,
        gridspec_kw: dict[str, str | int | float] = None,
        nrows: int = None,
        ncols: int = None,
        projection: Literal["rectilinear", "polar"] = "rectilinear",
    ):

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

        if not self.inplace:
            return self

    def grid_settings(
        self,
        ygrid: bool = False,
        xgrid: bool = False,
        linestyle: str | tuple = "solid",
        ylinewidth: float | int = 1,
        xlinewidth: float | int = 1,
    ):

        grid_settings = {
            "ygrid": ygrid,
            "xgrid": xgrid,
            "linestyle": linestyle,
            "xlinewidth": xlinewidth,
            "ylinewidth": ylinewidth,
        }
        self.plot_format["grid"] = grid_settings

        if not self.inplace:
            return self

    def clear_plots(self):
        self.plots = []
        self.plot_list = []

        if not self.inplace:
            return self

    def plot(
        self,
        savefig: bool = False,
        path: str | Path = None,
        filename: str = "",
        filetype: str = "svg",
        backend: str = "matplotlib",
        save_metadata: bool = False,
    ):

        if backend == "matplotlib":
            output = self._matplotlib_backend(
                savefig=savefig, path=path, filetype=filetype, filename=filename
            )
        else:
            raise AttributeError("Backend not implemented")
        if save_metadata:
            path = Path(path)
            filename = self._plot_data["y"] if filename == "" else filename
            path = path / f"{filename}.txt"
            self.save_metadata(path)
        return output

    def plot_legend(self):
        fig, ax = plt.subplots()

        handles = mp._make_legend_patches(
            color_dict=self._plot_dict["legend_dict"][0],
            alpha=self._plot_dict["legend_dict"][1],
            group=self._plot_dict["group_order"],
            subgroup=self._plot_dict["subgroup_order"],
        )
        ax.plot()
        ax.axis("off")
        ax.legend(handles=handles, frameon=False)
        return fig, ax

    def transform(
        self,
        ytransform: TRANSFORM | None = None,
        back_transform_yticks: bool = False,
        xtransform: TRANSFORM | None = None,
        back_transform_xticks: bool = False,
    ):
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

        if not self.inplace:
            return self

    def get_format(self):
        return self.plot_format

    def plot_data(
        self,
        y: str | None = None,
        x: str | None = None,
        ylabel: str = "",
        xlabel: str = "",
        title: str = "",
    ):
        if x is None and y is None:
            raise AttributeError("Must specify either x or y")
        self._plot_data = {
            "y": y,
            "x": x,
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
        }

        if not self.inplace:
            return self

    def _create_groupings(self, group, subgroup, group_order, subgroup_order):
        if group is None:
            unique_groups = [("",)]
            group_order = [""]
            levels = []
        elif subgroup is None:
            if group_order is None:
                group_order = np.unique(self.data[group])
            unique_groups = [(g,) for g in group_order]
            levels = [group]
        else:
            if group_order is None:
                group_order = np.unique(self.data[group])
            if subgroup_order is None:
                subgroup_order = np.unique(self.data[subgroup])
            unique_groups = list(set(zip(self.data[group], self.data[subgroup])))
            levels = [group, subgroup]
        return group_order, subgroup_order, unique_groups, levels

    def metadata(self):
        output = {
            "grouping": self._grouping,
            "data": self._plot_data,
            "format": self.plot_format,
            "transforms": self._plot_transforms,
            "plot_methods": self._plot_funcs,
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

    def _set_metadata_from_dict(self, metadata: dict):
        self._plot_data = metadata["data"]
        for key in metadata["format"]:
            self._load_plot_prefs(self.plot_format[key], metadata["format"][key])
        self._load_plot_prefs(self._plot_transforms, metadata["transforms"])

        # Not super happy with this code but it works for now
        if not self.inplace:
            self.inplace = True
            self.grouping(**metadata["grouping"])

            # Must loop through metadata and not set class variables otherwise it will
            # overwrite the instance and get stuck before the loop even runs.
            for pfunc, ppref in zip(metadata["plot_methods"], metadata["plot_prefs"]):
                method = getattr(self, pfunc)
                method(**ppref)
        else:
            self.grouping(**metadata["grouping"])
            # Must loop through metadata and not set class variables otherwise it will
            # overwrite the instance and get stuck before the loop even runs.
            for pfunc, ppref in zip(metadata["plot_methods"], metadata["plot_prefs"]):
                method = getattr(self, pfunc)
                method(**ppref)
        self.inplace = False
        return self

    def load_metadata(self, metadata_path: str | dict | Path):
        metadata = metadata_utils.load_metadata(metadata_path)
        if not self.inplace:
            self = self._set_metadata_from_dict(metadata)
            return self

    def set_metadata_directory(self, metadata_dir: str | dict | Path):
        metadata_utils.set_metadata_dir(metadata_dir)


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
        group_order, subgroup_order, unique_groups, levels = self._create_groupings(
            group, subgroup, group_order, subgroup_order
        )

        if facet:
            facet_dict = create_dict(group_order, unique_groups)
        else:
            facet_dict = create_dict(0, unique_groups)

        self._plot_dict = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "facet": facet,
            "facet_dict": facet_dict,
            "facet_title": facet_title,
            "levels": levels,
        }

        if not self.inplace:
            return self

    def line(
        self,
        linecolor: ColorParameter = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        func: str = "mean",
        err_func: str = "sem",
        fit_func: Callable | np.ndarray | None = None,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
    ):
        self._plot_funcs.append("line")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "func": func,
                "err_func": err_func,
                "fit_func": fit_func,
                "alpha": alpha,
                "unique_id": unique_id,
            }
        )

        linecolor_dict = create_dict(linecolor, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])
        line_plot = {
            "color_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "alpha": alpha,
            "unique_id": unique_id,
        }
        self.plots.append(line_plot)
        self.plot_list.append("line_plot")

        if not self.inplace:
            return self

    def aggline(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameter | tuple[str, str] = None,
        markeredgecolor: ColorParameter | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameter = None,
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float = 1.0,
        func="mean",
        err_func="sem",
        agg_func=None,
        fill_between=False,
        fillalpha: AlphaRange = 1.0,
        sort=True,
        unique_id=None,
    ):
        self._plot_funcs.append("aggline")
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
        }
        self.plots.append(line_plot)
        self.plot_list.append("aggline")

        if not self.inplace:
            return self

    def kde(
        self,
        kernel: Literal[
            "gaussian",
            "exponential",
            "box",
            "tri",
            "epa",
            "biweight",
            "triweight",
            "tricube",
            "cosine",
        ] = "gaussian",
        bw: Literal["ISJ", "silverman", "scott"] = "silverman",
        tol: float | int = 1e-3,
        common_norm: bool = True,
        linecolor: ColorParameter = None,
        linestyle: str = "-",
        linewidth: int = 2,
        fill_between: bool = False,
        alpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 1.0,
        unique_id: str | None = None,
        agg_func=None,
        err_func=None,
        kde_type: Literal["tree", "fft"] = "fft",
    ):
        self._plot_funcs.append("kde")
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
                "alpha": alpha,
                "fillalpha": fillalpha,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "kde_type": kde_type,
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
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "err_func": err_func,
            "kde_type": kde_type,
            "fillalpha": alpha / 2 if fillalpha is None else fillalpha,
        }

        self.plots.append(kde_plot)
        self.plot_list.append("kde")

        if not self.inplace:
            return self

    def polyhist(
        self,
        color: ColorParameter = None,
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
        self._plot_funcs.append("polyhist")
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
        }
        self.plots.append(poly_hist)
        self.plot_list.append("poly_hist")

        if not self.inplace:
            return self

    def hist(
        self,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        color: ColorParameter = None,
        linecolor: ColorParameter = None,
        linewidth: int = 2,
        hatch=None,
        fillalpha: AlphaRange = 1.0,
        linealpha: float = 1.0,
        bin_limits=None,
        stat: Literal["density", "probability", "count"] = "density",
        nbins=50,
        err_func=None,
        agg_func=None,
        unique_id=None,
    ):
        self._plot_funcs.append("hist")
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
            "hatch": hatch,
            "stat": stat,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "agg_func": agg_func,
            "err_func": err_func,
            "unique_id": unique_id,
            "fillalpha": fillalpha,
            "linealpha": linealpha,
            "projection": self.plot_format["figure"]["projection"],
        }
        self.plots.append(hist)
        self.plot_list.append("hist")

        if self.plot_format["figure"]["projection"] == "polar":
            self.plot_format["grid"]["ygrid"] = True
            self.plot_format["grid"]["xgrid"] = True

        if not self.inplace:
            return self

    def ecdf(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameter | tuple[str, str] = None,
        markeredgecolor: ColorParameter | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameter = None,
        linestyle: str = "-",
        linewidth: int = 2,
        linealpha: AlphaRange = 1.0,
        fill_between: bool = True,
        fillalpha: AlphaRange = 0.5,
        unique_id: str | None = None,
        agg_func=None,
        err_func=None,
        colorall: ColorParameter = None,
        ecdf_type: Literal["spline", "bootstrap", "none"] = "none",
        ecdf_args=None,
    ):
        if ecdf_args is None and agg_func is not None:
            ecdf_args = {"size": 1000, "repititions": 1000, "seed": 42}
            ecdf_type = "bootstrap"
        self._plot_funcs.append("ecdf")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "ecdf_type": ecdf_type,
                "agg_func": agg_func,
                "err_func": err_func,
                "colorall": colorall,
                "ecdf_args": ecdf_args,
            }
        )
        if colorall is None:
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
        else:
            temp_dict = create_dict(colorall, self._plot_dict["unique_groups"])
            markeredgecolor_dict = temp_dict
            markerfacecolor_dict = temp_dict
            linecolor_dict = temp_dict

        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])

        ecdf = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "linealpha": linealpha,
            "marker": marker_dict,
            "markersize": markersize,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "unique_id": unique_id,
            "ecdf_type": ecdf_type,
            "ecdf_args": ecdf_args if ecdf_args is not None else {},
            "agg_func": agg_func,
            "err_func": err_func,
            "fillalpha": fillalpha,
            "fill_between": fill_between,
        }
        self.plots.append(ecdf)
        self.plot_list.append("ecdf")

        self.plot_format["axis"]["ylim"] = [0.0, 1.0]

        if not self.inplace:
            return self

    def scatter(
        self,
        marker: str = ".",
        markercolor: ColorParameter | tuple[str, str] = "black",
        edgecolor: ColorParameter = "black",
        markersize: float | str = 1,
        alpha: float = 1.0,
    ):
        self._plot_funcs.append("scatter")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markercolor": markercolor,
                "edgecolor": edgecolor,
                "markersize": markersize,
                "alpha": alpha,
            }
        )
        # if isinstance(marker, tuple):
        #     marker0 = marker[0]
        #     marker1 = marker[1]
        # else:
        #     marker0 = marker
        #     marker1 = None
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

        # markers = process_scatter_args(
        #     marker0,
        #     self.data,
        #     self._plot_dict["group_order"],
        #     self._plot_dict["subgroup_order"],
        #     self._plot_dict["unique_groups"],
        #     marker1,
        # )
        # markers = markers.to_list()
        colors = process_scatter_args(
            markercolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            markercolor1,
            alpha=alpha,
        )
        edgecolors = process_scatter_args(
            edgecolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            edgecolor1,
            alpha=alpha,
        )
        markersize = process_scatter_args(
            markersize,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )
        facetgroup = process_scatter_args(
            self._plot_dict["facet"],
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )
        plot_data = {
            "markers": marker,
            "markercolors": colors,
            "edgecolors": edgecolors,
            "markersizes": markersize,
            "facetgroup": facetgroup,
        }

        self.plot_list.append("scatter")
        self.plots.append(plot_data)

        if not self.inplace:
            return self

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filetype: str = "svg",
        filename: str = "",
        transparent=False,
    ):
        if (
            self.plot_format["figure"]["nrows"] is None
            and self.plot_format["figure"]["ncols"] is None
        ):
            nrows = len(self._plot_dict["group_order"])
            ncols = 1
        elif self.plot_format["figure"]["nrows"] is None:
            nrows = 1
            ncols = self.plot_format["figure"]["ncols"]
        elif self.plot_format["figure"]["ncols"] is None:
            nrows = self.plot_format["figure"]["nrows"]
            ncols = 1
        else:
            nrows = self.plot_format["figure"]["nrows"]
            ncols = self.plot_format["figure"]["ncols"]
        if self.plot_format["figure"]["figsize"] is None:
            self.plot_format["figure"]["figsize"] = (6.4 * ncols, 4.8 * nrows)
        if self._plot_dict["facet"]:
            fig, ax = plt.subplots(
                subplot_kw=dict(
                    box_aspect=self.plot_format["figure"]["aspect"],
                    projection=self.plot_format["figure"]["projection"],
                ),
                figsize=self.plot_format["figure"]["figsize"],
                gridspec_kw=self.plot_format["figure"]["gridspec_kw"],
                ncols=ncols,
                nrows=nrows,
                layout="constrained",
            )
            ax = ax.flatten()
        else:
            fig, ax = plt.subplots(
                subplot_kw=dict(
                    box_aspect=self.plot_format["figure"]["aspect"],
                    projection=self.plot_format["figure"]["projection"],
                ),
                figsize=self.plot_format["figure"]["figsize"],
                layout="constrained",
            )
            ax = [ax]
        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.data,
                y=self._plot_data["y"],
                x=self._plot_data["x"],
                unique_groups=self._plot_dict["unique_groups"],
                facet_dict=self._plot_dict["facet_dict"],
                ax=ax,
                ytransform=self._plot_transforms["ytransform"],
                xtransform=self._plot_transforms["xtransform"],
                levels=self._plot_dict["levels"],
                **j,
            )
            if i == "kde" or i == "hist":
                if self._plot_data["x"] is not None:
                    if self.plot_format["axis"]["ylim"] is None:
                        self.plot_format["axis"]["ylim"] = [0, None]
                else:
                    if self.plot_format["axis"]["xlim"] is None:
                        self.plot_format["axis"]["xlim"] = [0, None]

        if (
            self.plot_format["axis"]["ydecimals"] is None
            and "y" in self._plot_data
            and self._plot_data["y"] is not None
        ):
            ydecimals = _decimals(self.data[self._plot_data["y"]])
        else:
            ydecimals = self.plot_format["axis"]["ydecimals"]
        if (
            self.plot_format["axis"]["xdecimals"] is None
            and "x" in self._plot_data
            and self._plot_data["x"] is not None
        ):
            xdecimals = _decimals(self.data[self._plot_data["x"]])
        else:
            xdecimals = self.plot_format["axis"]["xdecimals"]
        # num_plots = len(self._plot_dict["group_order"])
        for index, sub_ax in enumerate(ax[: len(self._plot_dict["group_order"])]):
            if self.plot_format["figure"]["projection"] == "rectilinear":
                sub_ax.autoscale()
                sub_ax.spines["right"].set_visible(False)
                sub_ax.spines["top"].set_visible(False)
                sub_ax.spines["left"].set_linewidth(
                    self.plot_format["axis_format"]["linewidth"]
                )
                sub_ax.spines["bottom"].set_linewidth(
                    self.plot_format["axis_format"]["linewidth"]
                )

                self._set_lims(sub_ax, ydecimals, axis="y")
                self._set_lims(sub_ax, xdecimals, axis="x")

                if self.plot_format["axis_format"]["yminorticks"] != 0:
                    self._set_minorticks(
                        sub_ax,
                        self.plot_format["axis_format"]["yminorticks"],
                        self._plot_transforms["ytransform"],
                        ticks="y",
                    )

                if self.plot_format["axis_format"]["xminorticks"] != 0:
                    self._set_minorticks(
                        sub_ax,
                        self.plot_format["axis_format"]["xminorticks"],
                        self._plot_transforms["xtransform"],
                        ticks="x",
                    )

                sub_ax.margins(self.plot_format["figure"]["margins"])
                sub_ax.set_xlabel(
                    self._plot_data["xlabel"],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontweight=self.plot_format["labels"]["label_fontweight"],
                    fontfamily=self.plot_format["labels"]["font"],
                    rotation=self.plot_format["labels"]["xlabel_rotation"],
                )
            else:
                if (
                    self.plot_format["axis"]["xunits"] == "radian"
                    or self.plot_format["axis"]["xunits"] == "wradian"
                ):
                    xticks = sub_ax.get_xticks()
                    labels = (
                        radian_ticks(xticks, rotate=False)
                        if self.plot_format["axis"]["xunits"] == "radian"
                        else radian_ticks(xticks, rotate=True)
                    )
                    sub_ax.set_xticks(
                        xticks,
                        labels,
                        fontfamily=self.plot_format["labels"]["font"],
                        fontweight=self.plot_format["labels"]["tick_fontweight"],
                        fontsize=self.plot_format["labels"]["ticklabel_size"],
                        rotation=self.plot_format["labels"]["xtick_rotation"],
                    )
                sub_ax.spines["polar"].set_visible(False)
                sub_ax.set_xlabel(
                    self._plot_data["xlabel"],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontweight=self.plot_format["labels"]["label_fontweight"],
                    fontfamily=self.plot_format["labels"]["font"],
                    rotation=self.plot_format["labels"]["xlabel_rotation"],
                )
                sub_ax.set_rmax(sub_ax.dataLim.ymax)
                ticks = sub_ax.get_yticks()
                sub_ax.set_yticks(
                    ticks,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
            if "hline" in self._plot_dict:
                self._plot_axlines(self._plot_dict["hline"], sub_ax)

            if "vline" in self._plot_dict:
                self._plot_axlines(self._plot_dict["vline"], sub_ax)

            sub_ax.tick_params(
                axis="both",
                which="major",
                labelsize=self.plot_format["labels"]["ticklabel_size"],
                width=self.plot_format["axis_format"]["tickwidth"],
                length=self.plot_format["axis_format"]["ticklength"],
                labelfontfamily=self.plot_format["labels"]["font"],
            )

            self._set_grid(sub_ax)

            if "/" in str(self._plot_data["y"]):
                self._plot_data["y"] = self._plot_data["y"].replace("/", "_")

            sub_ax.set_ylabel(
                self._plot_data["ylabel"],
                fontsize=self.plot_format["labels"]["labelsize"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["label_fontweight"],
                rotation=self.plot_format["labels"]["ylabel_rotation"],
            )
            if self._plot_dict["facet_title"]:
                sub_ax.set_title(
                    self._plot_dict["group_order"][index],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["title_fontweight"],
                )
            else:
                sub_ax.set_title(
                    self._plot_data["title"],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["title_fontweight"],
                )

        if self._plot_data["title"] is not None:
            fig.suptitle(
                self._plot_data["title"],
                fontsize=self.plot_format["labels"]["titlesize"],
            )

        if savefig:
            path = Path(path)
            if path.suffix[1:] not in MPL_SAVE_TYPES:
                filename = self._plot_data["y"] if filename == "" else filename
                path = path / f"{filename}.{filetype}"
            else:
                filetype = path.suffix[1:]
            plt.savefig(
                path,
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
        return fig, ax


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
        group_order, subgroup_order, unique_groups, levels = self._create_groupings(
            group, subgroup, group_order, subgroup_order
        )

        if group is not None:
            loc_dict, width = _process_positions(
                subgroup=subgroup,
                group_order=group_order,
                subgroup_order=subgroup_order,
                group_spacing=group_spacing,
            )
        else:
            group_order = [("",)]
            subgroup_order = [("",)]
            loc_dict = {("",): 0.0}
            loc_dict[("",)] = 0.0
            width = 1.0

        x_ticks = [index for index, _ in enumerate(group_order)]
        self._plot_dict = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "x_ticks": x_ticks,
            "loc_dict": loc_dict,
            "width": width,
            "levels": levels,
        }

        if not self.inplace:
            return self

    def jitter(
        self,
        color: ColorParameter = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameter = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        seed: int = 42,
        markersize: float = 2.0,
        unique_id: str | None = None,
        legend: bool = False,
    ):
        self._plot_funcs.append("jitter")
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
        }
        self.plots.append(jitter_plot)
        self.plot_list.append("jitter")

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
        color: ColorParameter = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameter = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        duplicate_offset=0.0,
        markersize: float = 2.0,
        agg_func: AGGREGATE | None = None,
        legend: bool = False,
    ):
        self._plot_funcs.append("jitteru")
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
        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
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
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "markersize": markersize,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
        }
        self.plots.append(jitteru_plot)
        self.plot_list.append("jitteru")

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
        func: AGGREGATE = "mean",
        capsize: int = 0,
        capstyle: str = "round",
        barwidth: float = 1.0,
        err_func: ERROR = "sem",
        linewidth: int = 2,
        color: ColorParameter = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_funcs.append("summary")
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
        }
        self.plots.append(summary_plot)
        self.plot_list.append("summary")

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
        func: AGGREGATE = "mean",
        agg_func: AGGREGATE = None,
        agg_width: float = 1.0,
        capsize: int = 0,
        capstyle: str = "round",
        barwidth: float = 1.0,
        err_func: ERROR = "sem",
        linewidth: int = 2,
        color: ColorParameter = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_funcs.append("summaryu")
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
        }
        self.plots.append(summary_plot)
        self.plot_list.append("summaryu")

        if legend:
            d = _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def boxplot(
        self,
        facecolor: ColorParameter = None,
        edgecolor: ColorParameter = None,
        fliers="",
        width: float = 1.0,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        line_alpha: AlphaRange = 1.0,
        showmeans: bool = False,
        show_ci: bool = False,
        legend: bool = False,
    ):
        self._plot_funcs.append("boxplot")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "fliers": fliers,
                "width": width,
                "alpha": alpha,
                "linewidth": linewidth,
                "line_alpha": line_alpha,
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

        boxplot = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "fliers": fliers,
            "width": width * self._plot_dict["width"],
            "showmeans": showmeans,
            "show_ci": show_ci,
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
        }
        self.plots.append(boxplot)
        self.plot_list.append("boxplot")

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
        facecolor: ColorParameter = None,
        edgecolor: ColorParameter = None,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
        showextrema: bool = False,
        width: float = 1.0,
        showmeans: bool = False,
        showmedians: bool = False,
        legend: bool = False,
    ):
        self._plot_funcs.append("violin")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "showextrema": showextrema,
                "width": width,
                "showmeans": showmeans,
                "showmedians": showmedians,
                "legend": legend,
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
            "showextrema": showextrema,
            "width": width * self._plot_dict["width"],
            "showmeans": showmeans,
            "showmedians": showmedians,
            "linewidth": linewidth,
        }
        self.plots.append(violin)
        self.plot_list.append("violin")

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
        cutoff: None | float | int | list[float | int],
        unique_id=None,
        facecolor=None,
        edgecolor: ColorParameter = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "percent"] = "density",
        include_bins: list[bool] | None = None,
        invert: bool = False,
        legend: bool = False,
    ):
        self._plot_funcs.append("percent")
        self._plot_prefs.append(
            {
                "cutoff": cutoff,
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "linewidth": linewidth,
                "barwidth": barwidth,
                "alpha": alpha,
                "line_alpha": line_alpha,
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
            "line_alpha": line_alpha,
            "include_bins": include_bins,
            "unique_id": unique_id,
            "invert": invert,
            "axis_type": axis_type,
        }
        self.plots.append(percent_plot)
        self.plot_list.append("percent")
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
        facecolor: ColorParameter = None,
        edgecolor: ColorParameter = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "count", "percent"] = "density",
        legend: bool = False,
    ):
        self._plot_funcs.append("count")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "barwidth": barwidth,
                "linewidth": linewidth,
                "alpha": alpha,
                "line_alpha": line_alpha,
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
            "line_alpha": line_alpha,
            "axis_type": axis_type,
        }
        self.plots.append(count_plot)
        self.plot_list.append("count")

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

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filename: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_format["figure"]["aspect"]),
            figsize=self.plot_format["figure"]["figsize"],
            layout="constrained",
        )

        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.data,
                y=self._plot_data["y"],
                loc_dict=self._plot_dict["loc_dict"],
                unique_groups=self._plot_dict["unique_groups"],
                ax=ax,
                transform=self._plot_transforms["ytransform"],
                levels=self._plot_dict["levels"],
                **j,
            )

        ax.set_xticks(
            ticks=self._plot_dict["x_ticks"],
            labels=self._plot_dict["group_order"],
            rotation=self.plot_format["labels"]["xtick_rotation"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["tick_fontweight"],
            fontsize=self.plot_format["labels"]["ticklabel_size"],
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(self.plot_format["axis_format"]["linewidth"])
        ax.spines["bottom"].set_linewidth(self.plot_format["axis_format"]["linewidth"])
        if "/" in str(self._plot_data["y"]):
            self._plot_data["y"] = self._plot_data["y"].replace("/", "_")

        self._set_grid(ax)

        self._set_lims(ax, self.plot_format["axis"]["ydecimals"], axis="y")
        truncate = (
            self.plot_format["axis_format"]["xsteps"][1] != 0
            or self.plot_format["axis_format"]["xsteps"][2]
            != self.plot_format["axis_format"]["xsteps"][0]
        )
        if truncate:
            ticks = self._plot_dict["x_ticks"]
            ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])

        if self.plot_format["axis_format"]["yminorticks"] != 0:
            self._set_minorticks(
                ax,
                self.plot_format["axis_format"]["yminorticks"],
                self._plot_transforms["ytransform"],
                ticks="y",
            )

        ax.set_ylabel(
            self._plot_data["ylabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            rotation=self.plot_format["labels"]["ylabel_rotation"],
        )
        ax.set_title(
            self._plot_data["title"],
            fontsize=self.plot_format["labels"]["titlesize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["title_fontweight"],
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.plot_format["labels"]["ticklabel_size"],
            width=self.plot_format["axis_format"]["tickwidth"],
            length=self.plot_format["axis_format"]["ticklength"],
            labelfontfamily=self.plot_format["labels"]["font"],
        )
        ax.margins(x=self.plot_format["figure"]["margins"])

        if "hline" in self._plot_dict:
            self._plot_axlines(self._plot_dict["hline"], ax)

        if "vline" in self._plot_dict:
            self._plot_axlines(self._plot_dict["vline"], ax)

        if "legend_dict" in self._plot_dict:
            handles = mp._make_legend_patches(
                color_dict=self._plot_dict["legend_dict"][0],
                alpha=self._plot_dict["legend_dict"][1],
                group=self._plot_dict["group_order"],
                subgroup=self._plot_dict["subgroup_order"],
            )
            ax.legend(
                handles=handles,
                bbox_to_anchor=self._plot_dict["legend_anchor"],
                loc=self._plot_dict["legend_loc"],
                frameon=False,
            )

        if savefig:
            path = Path(path)
            if path.suffix[1:] not in MPL_SAVE_TYPES:
                filename = self._plot_data["y"] if filename == "" else filename
                path = path / f"{filename}.{filetype}"
            else:
                filetype = path.suffix[1:]
            plt.savefig(
                path,
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
        return fig, ax


class GraphPlot:

    def __init__(self, graph):
        self._plot_dict = {}
        self._plot_dict["graph"] = graph
        self.plots = []

    def graphplot(
        self,
        marker_alpha: float = 0.8,
        line_alpha: float = 0.1,
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
            "line_alpha": line_alpha,
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
