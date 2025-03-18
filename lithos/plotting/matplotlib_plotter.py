from pathlib import Path
from typing import Literal
from dataclasses import asdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.colors import to_rgba
import matplotlib as mpl

from ..utils import (
    get_backtransform,
    get_transform,
)
from .plot_utils import _decimals, radian_ticks
from .plot_utils import get_ticks
from .types import SavePath

MARKERS = [
    "o",
    "X",
    "^",
    "s",
    "d",
    "h",
    "p",
    "*",
    "<",
    "H",
    "D",
    "v",
    "P",
    ">",
    "8",
    ".",
]
HATCHES = [
    None,
    "/",
    "o",
    "-",
    "*",
    "+",
    "\\",
    "|",
    "O",
    ".",
    "x",
]


class Plotter:
    filetypes = {
        "eps",
        "jpeg",
        "jpg",
        "pdf",
        "pgf",
        "png",
        "ps",
        "raw",
        "rgba",
        "svg",
        "svgz",
        "tif",
        "tiff",
        "webp",
    }

    def __init__(
        self,
        plot_data: list,
        plot_dict: dict[str],
        metadata: dict[dict[str]],
        savefig: bool = False,
        path: SavePath = "",
        filetype: str = "svg",
        filename: str | Path = "",
        axes: mpl.axes.Axes | list[mpl.axes.Axes] = None,
        figure: mpl.figure.Figure = None,
    ):
        self.plot_data = plot_data
        self.plot_format = metadata["format"]
        self.plot_dict = plot_dict
        self.plot_transforms = metadata["transforms"]
        self.plot_labels = metadata["data"]
        self._savefig = savefig
        self.path = path
        self.filetype = filetype
        self.filename = filename

        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["svg.fonttype"] = "none"

        if axes is None:
            self.fig, self.axes = self.create_figure()
        else:
            if isinstance(axes, (list, np.ndarray, tuple)):
                self.axes = axes
            else:
                self.axes = [axes]
            self.fig = figure

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

    def _set_lims(
        self,
        ax: plt.Axes.axes,
        lim: tuple[float | int, float | int],
        ticks,
        axis: Literal["x", "y"] = "x",
    ):
        if axis == "y":
            if self.plot_format["axis"]["yscale"] not in ["log", "symlog"]:
                ax.set_ylim(bottom=lim[0], top=lim[1])
                if self.plot_format["axis_format"]["truncate_yaxis"]:
                    start = self.plot_format["axis_format"]["ysteps"][1]
                    end = self.plot_format["axis_format"]["ysteps"][2] - 1
                    ax.spines["left"].set_bounds(ticks[start], ticks[end])
                elif self.plot_format["axis"]["yaxis_lim"] is not None:
                    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
            else:
                ax.set_yscale(self.plot_format["axis"]["yscale"])
                ax.set_ylim(bottom=lim[0], top=lim[1])
        else:
            if self.plot_format["axis"]["xscale"] not in ["log", "symlog"]:
                ax.set_xlim(left=lim[0], right=lim[1])
                truncate = self.plot_format["axis_format"]["truncate_xaxis"]
                if truncate:
                    start = self.plot_format["axis_format"]["xsteps"][1]
                    end = self.plot_format["axis_format"]["xsteps"][2] - 1
                    ax.spines["bottom"].set_bounds(ticks[start], ticks[end])
                elif self.plot_format["axis"]["xaxis_lim"] is not None:
                    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
            else:
                ax.set_xscale(self.plot_format["axis"]["xscale"])
                ax.set_xlim(left=lim[0], right=lim[1])

    def _format_ticklabels(
        self,
        ax: plt.Axes.axes,
        ticks,
        decimals: int,
        axis: Literal["y", "x"] = "x",
        style: Literal["lithos", "default"] = "lithos",
    ):
        if axis == "y":
            if self.plot_format["axis"]["yscale"] not in ["log", "symlog"]:
                if (
                    "back_transform_yticks" in self.plot_transforms
                    and self.plot_transforms["back_transform_yticks"]
                ):
                    tick_labels = get_backtransform(self.plot_transforms["ytransform"])(
                        ticks
                    )
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
                if style == "lithos":
                    label_start = self.plot_format["axis_format"]["ysteps"][1]
                    label_end = self.plot_format["axis_format"]["ysteps"][2]
                else:
                    label_start = 0
                    label_end = len(ticks)
                ax.set_yticks(
                    ticks[label_start:label_end],
                    labels=tick_labels[label_start:label_end],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
            else:
                ax.set_yscale(self.plot_format["axis"]["yscale"])
                ax.set_yticks(
                    ticks,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
        else:
            if self.plot_format["axis"]["xscale"] not in ["log", "symlog"]:
                if (
                    "back_transform_xticks" in self.plot_transforms
                    and self.plot_transforms["back_transform_xticks"]
                ):
                    tick_labels = get_backtransform(self.plot_transforms["xtransform"])(
                        ticks
                    )
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
                if style == "lithos":
                    label_start = self.plot_format["axis_format"]["xsteps"][1]
                    label_end = self.plot_format["axis_format"]["xsteps"][2]
                else:
                    label_start = 0
                    label_end = len(ticks)
                ax.set_xticks(
                    ticks[label_start:label_end],
                    labels=tick_labels[label_start:label_end],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["xtick_rotation"],
                )
            else:
                ax.set_xscale(self.plot_format["axis"]["xscale"])
                ax.set_xticks(
                    ticks,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.plot_format["labels"]["ticklabel_size"],
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )

    def set_axis(
        self,
        ax: plt.Axes.axes,
        decimals: int,
        axis: Literal["x", "y"] = "x",
        style: Literal["default", "lithos"] = "lithos",
    ):

        if axis == "y":
            ticks = ax.get_yticks()
            if style == "lithos":
                lim, _, ticks = get_ticks(
                    lim=self.plot_format["axis"]["ylim"],
                    axis_lim=self.plot_format["axis"]["yaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["ysteps"],
                )
            transform = self.plot_transforms["ytransform"]
            minorticks = self.plot_format["axis_format"]["yminorticks"]
        else:
            ticks = ax.get_xticks()
            if style == "lithos":
                lim, _, ticks = get_ticks(
                    lim=self.plot_format["axis"]["xlim"],
                    axis_lim=self.plot_format["axis"]["xaxis_lim"],
                    ticks=ticks,
                    steps=self.plot_format["axis_format"]["xsteps"],
                )
            transform = self.plot_transforms["xtransform"]
            minorticks = self.plot_format["axis_format"]["xminorticks"]
        if style == "lithos":
            self._set_lims(ax=ax, lim=lim, ticks=ticks, axis=axis)
        self._format_ticklabels(
            ax=ax, ticks=ticks, decimals=decimals, axis=axis, style=style
        )
        if minorticks != 0:
            self._set_minorticks(
                ax,
                ticks,
                minorticks,
                transform,
                axis=axis,
            )

    def _set_minorticks(
        self,
        ax,
        ticks: np.ndarray[float | int],
        nticks: int,
        transform: str,
        axis: Literal["y", "x"],
    ):
        ticks = get_backtransform(transform)(ticks)
        mticks = np.zeros((len(ticks) - 1) * nticks)
        for index in range(ticks.size - 1):
            vals = np.linspace(
                ticks[index], ticks[index + 1], num=nticks + 2, endpoint=True
            )
            start = index * nticks
            end = index * nticks + nticks
            mticks[start:end] = vals[1:-1]
        if axis == "y":
            if self.plot_format["axis_format"]["truncate_yaxis"]:
                start = self.plot_format["axis_format"]["ysteps"][1] * nticks
                end = self.plot_format["axis_format"]["ysteps"][2] * nticks
            else:
                start = 0
                end = len(mticks)
            ax.set_yticks(
                get_transform(transform)(mticks[start:end]),
                minor=True,
            )
        else:
            if self.plot_format["axis_format"]["truncate_xaxis"]:
                start = self.plot_format["axis_format"]["xsteps"][1] * nticks
                end = self.plot_format["axis_format"]["xsteps"][2] * nticks
            else:
                start = 0
                end = len(mticks)
            ax.set_xticks(
                get_transform(transform)(mticks[start:end]),
                minor=True,
            )
        ax.tick_params(
            axis=axis,
            which="minor",
            width=self.plot_format["axis_format"]["minor_tickwidth"],
            length=self.plot_format["axis_format"]["minor_ticklength"],
            labelfontfamily=self.plot_format["labels"]["font"],
        )

    def _make_legend_patches(self, color_dict, alpha, group, subgroup):
        legend_patches = []
        # for j in group:
        #     if j in color_dict:
        #         legend_patches.append(
        #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
        #         )
        # for j in subgroup:
        #     if j in color_dict:
        #         legend_patches.append(
        #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
        #         )
        for key, value in color_dict.items():
            legend_patches.append(
                mpatches.Patch(color=to_rgba(value, alpha=alpha), label=key)
            )
        return legend_patches

    def get_plot_func(self, plot_type):
        if plot_type == "rectangle":
            return self._plot_rectangles
        elif plot_type == "line":
            return self._plot_line
        elif plot_type == "jitter":
            return self._plot_jitter
        elif plot_type == "scatter":
            return self._plot_scatter
        elif plot_type == "summary":
            return self._plot_summary
        elif plot_type == "box":
            return self._plot_box
        elif plot_type == "violin":
            return self._plot_violin
        else:
            raise ValueError(f"Unsupported plot function: {plot_type}")

    def _plot_rectangles(
        self,
        heights: list,
        bottoms: list,
        bins: list,
        binwidths: list,
        fillcolors: list[str],
        edgecolors: list[str],
        fill_alpha: float,
        edge_alpha: float,
        hatches: list[str],
        linewidth: float,
        ax: mpl.axes.Axes | list[mpl.axes.Axes] | np.ndarray[mpl.axes.Axes],
        facet_index: list[int] | None = None,
        axis: Literal["x", "y"] = "x",
    ):
        if facet_index is None:
            facet_index = [0] * len(heights)
        for t, b, loc, bw, fc, ec, ht, facet in zip(
            heights,
            bottoms,
            bins,
            binwidths,
            fillcolors,
            edgecolors,
            hatches,
            facet_index,
        ):
            if axis == "x":
                ax[facet].bar(
                    x=loc,
                    height=t,
                    bottom=b,
                    width=bw,
                    color=to_rgba(fc, alpha=fill_alpha),
                    edgecolor=to_rgba(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=ht,
                )
            else:
                ax[facet].barh(
                    y=bins,
                    width=t,
                    left=b,
                    height=bw,
                    color=to_rgba(fc, alpha=fill_alpha),
                    edgecolor=to_rgba(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=hatches,
                )
        return ax

    def _plot_jitter(
        self,
        x_data: list,
        y_data: list,
        marker: str,
        markerfacecolor: list[str],
        markeredgecolor: list[str],
        markersize: float,
        alpha: float,
        edge_alpha: float,
        ax: plt.Axes,
    ):
        for x, y, mk, mf, me, ms in zip(
            x_data, y_data, marker, markerfacecolor, markeredgecolor, markersize
        ):
            ax[0].plot(
                x,
                y,
                mk,
                markerfacecolor=to_rgba(mf, alpha=alpha) if mf != "none" else "none",
                markeredgecolor=(
                    to_rgba(me, alpha=edge_alpha) if me != "none" else "none"
                ),
                markersize=ms,
            )
        return ax

    def _plot_scatter(
        self,
        x_data: list,
        y_data: list,
        marker: str,
        markerfacecolor: list[str],
        markeredgecolor: list[str],
        markersize: float,
        alpha: float,
        edge_alpha: float,
        facet_index: list[int],
        ax: plt.Axes,
    ):
        for x, y, mk, mf, me, ms, facet in zip(
            x_data,
            y_data,
            marker,
            markerfacecolor,
            markeredgecolor,
            markersize,
            facet_index,
        ):
            ax[facet].plot(
                x,
                y,
                mk,
                markerfacecolor=to_rgba(mf, alpha=alpha) if mf != "none" else "none",
                markeredgecolor=(
                    to_rgba(me, alpha=edge_alpha) if me != "none" else "none"
                ),
                markersize=ms,
            )
        return ax

    def _plot_summary(
        self,
        x_data: list,
        y_data: list,
        error_data: list,
        widths: list,
        colors: list,
        linewidth: float,
        alpha: float,
        capstyle: str,
        capsize: float,
        ax: plt.Axes,
    ):
        for xd, yd, e, c, w in zip(x_data, y_data, error_data, colors, widths):
            _, caplines, bars = ax[0].errorbar(
                x=xd,
                y=yd,
                yerr=e,
                c=to_rgba(c, alpha=alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=capsize,
            )
            for cap in caplines:
                cap.set_solid_capstyle(capstyle)
                cap.set_markeredgewidth(linewidth)
                cap._marker._capstyle = CapStyle(capstyle)
            for b in bars:
                b.set_capstyle(capstyle)
            _, caplines, bars = ax[0].errorbar(
                y=yd,
                x=xd,
                xerr=w / 2,
                c=to_rgba(c, alpha=alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=0,
            )
            for b in bars:
                b.set_capstyle(capstyle)
        return ax

    def _plot_box(
        self,
        x_data: list,
        y_data: list,
        facecolors: list[str],
        edgecolors: list[str],
        alpha: float,
        linealpha: float,
        fliers: bool,
        linewidth: float,
        width: float,
        show_ci: bool,
        showmeans: bool,
        ax: plt.Axes,
    ):
        for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
            props = {
                "boxprops": {
                    "facecolor": to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs,
                    "edgecolor": (
                        to_rgba(ecs, alpha=linealpha) if ecs != "none" else ecs
                    ),
                },
                "medianprops": {
                    "color": to_rgba(ecs, alpha=linealpha) if ecs != "none" else ecs
                },
                "whiskerprops": {
                    "color": to_rgba(ecs, alpha=linealpha) if ecs != "none" else ecs
                },
                "capprops": {
                    "color": to_rgba(ecs, alpha=linealpha) if ecs != "none" else ecs
                },
            }
            if showmeans:
                props["meanprops"] = {
                    "color": to_rgba(ecs, alpha=linealpha) if ecs != "none" else ecs
                }
            bplot = ax[0].boxplot(
                y,
                positions=x,
                sym=fliers,
                widths=width,
                notch=show_ci,
                patch_artist=True,
                showmeans=showmeans,
                meanline=showmeans,
                **props,
            )
            for i in bplot["boxes"]:
                i.set_linewidth(linewidth)

    def _plot_violin(
        self,
        x_data: list,
        y_data: list,
        location: list[float],
        facecolors: list[str],
        edgecolors: list[str],
        alpha: float,
        edge_alpha: float,
        linewidth: float,
        ax: plt.Axes,
    ):
        for x, y, loc, fcs, ecs in zip(
            x_data, y_data, location, facecolors, edgecolors
        ):
            ax[0].fill_betweenx(
                x,
                y * -1 + loc,
                y + loc,
                facecolor=to_rgba(fcs, alpha),
                edgecolor=to_rgba(ecs, edge_alpha),
                linewidth=linewidth,
            )

    def _plot_line(
        self,
        ax: plt.Axes,
        x_data: list,
        y_data: list,
        error_data: list,
        facet_index: list[int],
        marker: list[str | None] | None = None,
        linecolor: list[str | None] | None = None,
        linewidth: list[float | None] | None = None,
        linestyle: list[str | None] | None = None,
        markerfacecolor: list[str | None] | None = None,
        markeredgecolor: list[str | None] | None = None,
        fill_between: bool = False,
        fb_direction: Literal["x", "y"] = "y",
        markersize: float | None = None,
        fillalpha: float | None = None,
        linealpha: float | None = None,
    ):
        for x, y, err, ls, lc, mf, me, mk, fc in zip(
            x_data,
            y_data,
            error_data,
            linestyle,
            linecolor,
            markerfacecolor,
            markeredgecolor,
            marker,
            facet_index,
        ):
            if not fill_between:
                if fb_direction == "x":
                    ax[fc].errorbar(
                        x,
                        y,
                        xerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=linewidth,
                        linewidth=linewidth,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                    )
                else:
                    ax[fc].errorbar(
                        x,
                        y,
                        yerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=linewidth,
                        linewidth=linewidth,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                    )
            else:
                if err is not None:
                    if fb_direction == "y":
                        ax[fc].fill_between(
                            x,
                            y - err,
                            y + err,
                            color=to_rgba(lc, alpha=fillalpha),
                            linewidth=0,
                            edgecolor="none",
                        )
                    else:
                        ax[fc].fill_betweenx(
                            y,
                            x - err,
                            x + err,
                            color=to_rgba(lc, alpha=fillalpha),
                            linewidth=0,
                            edgecolor="none",
                        )
                ax[fc].plot(
                    x,
                    y,
                    linestyle=ls,
                    linewidth=linewidth,
                    color=lc,
                    alpha=linealpha,
                )

    def plot_legend(self):
        fig, ax = plt.subplots()

        handles = self._make_legend_patches(
            color_dict=self.plot_dict["legend_dict"][0],
            alpha=self.plot_dict["legend_dict"][1],
            group=self.plot_dict["group_order"],
            subgroup=self.plot_dict["subgroup_order"],
        )
        ax.plot()
        ax.axis("off")
        ax.legend(handles=handles, frameon=False)
        return fig, ax

    def _plot(self):
        for p in self.plot_data:
            plot_func = self.get_plot_func(p.plot_type)
            p_dict = asdict(p)
            p_dict.pop("plot_type")
            plot_func(**p_dict, ax=self.axes)

    def plot(self):

        self._plot()
        self.format_plot()

        if self._savefig:
            self.savefig(
                path=self.path,
                fig=self.fig,
                filename=self.filename,
                filetype=self.filetype,
            )
        return self.fig, self.axes

    def savefig(
        self,
        path: SavePath,
        fig: mpl.figure.Figure,
        filename: str | None = None,
        filetype: str | None = None,
        transparent: bool = False,
    ):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix[1:] not in self.filetypes:
            path = path / f"{filename}.{filetype}"
        else:
            filetype = path.suffix[1:]
        fig.savefig(
            path,
            format=filetype,
            bbox_inches="tight",
            transparent=transparent,
        )


class LinePlotter(Plotter):
    def create_figure(self):
        if (
            self.plot_format["figure"]["nrows"] is None
            and self.plot_format["figure"]["ncols"] is None
        ):
            nrows = len(self.plot_dict["group_order"])
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
        if self.plot_dict["facet"]:
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
            ax = ax.flat
            for i in ax[len(self.plot_dict["group_order"]) :]:
                i.remove()
            ax = ax[: len(self.plot_dict["group_order"])]
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
        return fig, ax

    def format_rectilinear(self, ax: mpl.axes.Axes, xdecimals: int, ydecimals: int):
        ax.autoscale()
        for spine, lw in self.plot_format["axis_format"]["linewidth"].items():
            if lw == 0:
                ax.spines[spine].set_visible(False)
            else:
                ax.spines[spine].set_linewidth(lw)

        self.set_axis(
            ax, xdecimals, axis="x", style=self.plot_format["axis_format"]["style"]
        )
        self.set_axis(
            ax, ydecimals, axis="y", style=self.plot_format["axis_format"]["style"]
        )

        ax.margins(self.plot_format["figure"]["margins"])
        ax.set_xlabel(
            self.plot_labels["xlabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            fontfamily=self.plot_format["labels"]["font"],
            rotation=self.plot_format["labels"]["xlabel_rotation"],
        )

    def format_polar(self, ax: mpl.axes.Axes):
        if (
            self.plot_format["axis"]["xunits"] == "radian"
            or self.plot_format["axis"]["xunits"] == "wradian"
        ):
            xticks = ax.get_xticks()
            labels = (
                radian_ticks(xticks, rotate=False)
                if self.plot_format["axis"]["xunits"] == "radian"
                else radian_ticks(xticks, rotate=True)
            )
            ax.set_xticks(
                xticks,
                labels,
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["tick_fontweight"],
                fontsize=self.plot_format["labels"]["ticklabel_size"],
                rotation=self.plot_format["labels"]["xtick_rotation"],
            )
        ax.spines["polar"].set_visible(False)
        ax.set_xlabel(
            self.plot_labels["xlabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            fontfamily=self.plot_format["labels"]["font"],
            rotation=self.plot_format["labels"]["xlabel_rotation"],
        )
        ax.set_rmax(ax.dataLim.ymax)
        ticks = ax.get_yticks()
        ax.set_yticks(
            ticks,
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["tick_fontweight"],
            fontsize=self.plot_format["labels"]["ticklabel_size"],
            rotation=self.plot_format["labels"]["ytick_rotation"],
        )

    def format_plot(self):
        for p in self.plot_data:
            if p.plot_type == "kde" or p.plot_type == "hist":
                if self.plot_data["x"] is not None:
                    if self.plot_format["axis"]["ylim"] is None:
                        self.plot_format["axis"]["ylim"] = [0, None]
                else:
                    if self.plot_format["axis"]["xlim"] is None:
                        self.plot_format["axis"]["xlim"] = [0, None]

        if (
            self.plot_format["axis"]["ydecimals"] is None
            and "y" in self.plot_data
            and self.plot_data["y"] is not None
        ):
            ydecimals = _decimals(self.data[self.plot_data["y"]])
        else:
            ydecimals = self.plot_format["axis"]["ydecimals"]
        if (
            self.plot_format["axis"]["xdecimals"] is None
            and "x" in self.plot_data
            and self.plot_data["x"] is not None
        ):
            xdecimals = _decimals(self.data[self.plot_data["x"]])
        else:
            xdecimals = self.plot_format["axis"]["xdecimals"]
        # num_plots = len(self.plot_dict["group_order"])
        for index, sub_ax in enumerate(self.axes[: len(self.plot_dict["group_order"])]):
            if self.plot_format["figure"]["projection"] == "rectilinear":
                self.format_rectilinear(sub_ax, xdecimals, ydecimals)
            else:
                self.format_polar(sub_ax)
            if "hline" in self.plot_dict:
                self._plot_axlines(self.plot_dict["hline"], sub_ax)

            if "vline" in self.plot_dict:
                self._plot_axlines(self.plot_dict["vline"], sub_ax)

            sub_ax.tick_params(
                axis="both",
                which="major",
                labelsize=self.plot_format["labels"]["ticklabel_size"],
                width=self.plot_format["axis_format"]["tickwidth"],
                length=self.plot_format["axis_format"]["ticklength"],
                labelfontfamily=self.plot_format["labels"]["font"],
            )

            self._set_grid(sub_ax)

            sub_ax.set_ylabel(
                self.plot_labels["ylabel"],
                fontsize=self.plot_format["labels"]["labelsize"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["label_fontweight"],
                rotation=self.plot_format["labels"]["ylabel_rotation"],
            )
            if self.plot_dict["facet_title"]:
                sub_ax.set_title(
                    self.plot_dict["group_order"][index],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["title_fontweight"],
                )
            else:
                sub_ax.set_title(
                    self.plot_labels["title"],
                    fontsize=self.plot_format["labels"]["labelsize"],
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["title_fontweight"],
                )

        if self.plot_labels["title"] is not None:
            self.fig.suptitle(
                self.plot_labels["title"],
                fontsize=self.plot_format["labels"]["titlesize"],
            )


class CategoricalPlotter(Plotter):
    def create_figure(self):
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_format["figure"]["aspect"]),
            figsize=self.plot_format["figure"]["figsize"],
            layout="constrained",
        )
        return fig, [ax]

    def format_plot(self):
        ax = self.axes[0]
        ax.set_xticks(
            ticks=self.plot_dict["x_ticks"],
            labels=self.plot_dict["group_order"],
            rotation=self.plot_format["labels"]["xtick_rotation"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["tick_fontweight"],
            fontsize=self.plot_format["labels"]["ticklabel_size"],
        )
        for spine, lw in self.plot_format["axis_format"]["linewidth"].items():
            if lw == 0:
                ax.spines[spine].set_visible(False)
            else:
                ax.spines[spine].set_linewidth(lw)
        self._set_grid(ax)

        self.set_axis(
            ax,
            self.plot_format["axis"]["ydecimals"],
            axis="y",
            style=self.plot_format["axis_format"]["style"],
        )

        if self.plot_format["axis_format"]["truncate_xaxis"]:
            ticks = self.plot_dict["x_ticks"]
            ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])

        ax.set_ylabel(
            self.plot_labels["ylabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            rotation=self.plot_format["labels"]["ylabel_rotation"],
        )
        ax.set_title(
            self.plot_labels["title"],
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

        if "legend_dict" in self.plot_dict:
            handles = self._make_legend_patches(
                color_dict=self.plot_dict["legend_dict"][0],
                alpha=self.plot_dict["legend_dict"][1],
                group=self.plot_dict["group_order"],
                subgroup=self.plot_dict["subgroup_order"],
            )
            ax.legend(
                handles=handles,
                bbox_to_anchor=self.plot_dict["legend_anchor"],
                loc=self.plot_dict["legend_loc"],
                frameon=False,
            )


# def _plot_network(
#     graph,
#     marker_alpha: float = 0.8,
#     linealpha: float = 0.1,
#     markersize: int = 2,
#     marker_scale: int = 1,
#     linewidth: int = 1,
#     edge_color: str = "k",
#     marker_color: str = "red",
#     marker_attr: Optional[str] = None,
#     cmap: str = "gray",
#     seed: int = 42,
#     scale: int = 50,
#     plot_max_degree: bool = False,
#     layout: Literal["spring", "circular", "communities"] = "spring",
# ):

#     if isinstance(cmap, str):
#         cmap = plt.colormaps[cmap]
#     _, ax = plt.subplots()
#     Gcc = graph.subgraph(
#         sorted(nx.connected_components(graph), key=len, reverse=True)[0]
#     )
#     if layout == "spring":
#         pos = nx.spring_layout(Gcc, seed=seed, scale=scale)
#     elif layout == "circular":
#         pos = nx.circular_layout(Gcc, scale=scale)
#     elif layout == "random":
#         pos = nx.random_layout(Gcc, seed=seed)
#     elif layout == "communities":
#         communities = nx.community.greedy_modularity_communities(Gcc)
#         # Compute positions for the node clusters as if they were themselves nodes in a
#         # supergraph using a larger scale factor
#         _ = nx.cycle_graph(len(communities))
#         superpos = nx.spring_layout(Gcc, scale=scale, seed=seed)

#         # Use the "supernode" positions as the center of each node cluster
#         centers = list(superpos.values())
#         pos = {}
#         for center, comm in zip(centers, communities):
#             pos.update(
#                 nx.spring_layout(nx.subgraph(Gcc, comm), center=center, seed=seed)
#             )

#     nodelist = list(Gcc)
#     markersize = np.array([Gcc.degree(i) for i in nodelist])
#     markersize = markersize * marker_scale
#     xy = np.asarray([pos[v] for v in nodelist])

#     edgelist = list(Gcc.edges(data=True))
#     edge_pos = np.asarray([(pos[e0], pos[e1]) for (e0, e1, _) in edgelist])
#     _, _, data = edgelist[0]
#     if edge_color in data:
#         edge_color = [data["weight"] for (_, _, data) in edgelist]
#         edge_vmin = min(edge_color)
#         edge_vmax = max(edge_color)
#         color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
#         edge_color = [cmap(color_normal(e)) for e in edge_color]
#     edge_collection = LineCollection(
#         edge_pos,
#         colors=edge_color,
#         linewidths=linewidth,
#         antialiaseds=(1,),
#         linestyle="solid",
#         alpha=linealpha,
#     )
#     edge_collection.set_cmap(cmap)
#     edge_collection.set_clim(edge_vmin, edge_vmax)
#     edge_collection.set_zorder(0)  # edges go behind nodes
#     edge_collection.set_label("edges")
#     ax.add_collection(edge_collection)

#     if isinstance(marker_color, dict):
#         if marker_attr is not None:
#             mcolor = [
#                 marker_color[data[marker_attr]] for (_, data) in Gcc.nodes(data=True)
#             ]
#         else:
#             mcolor = "red"
#     else:
#         mcolor = marker_color

#     path_collection = ax.scatter(
#         xy[:, 0], xy[:, 1], s=markersize, alpha=marker_alpha, c=mcolor
#     )
#     path_collection.set_zorder(1)
#     ax.axis("off")
