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
    _decimals,
    radian_ticks,
    get_backtransform,
    get_transform,
)
from .plot_utils import get_ticks
from .types import SavePath


class MPLPlotter:
    filetypes = ["svg", "png", "jpeg"]

    def __init__(
        self,
        plot_data: list,
        plot_format: dict[str],
        ax: mpl.axes.Axes | list[mpl.axes.Axes] = None,
        fig: mpl.figure.Figure = None,
    ):
        self.plot_data = plot_data
        self.plot_format = plot_format

        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["svg.fonttype"] = "none"

        if isinstance(ax, list):
            self.ax = ax
        else:
            self.ax = [ax]
        self.fig = fig

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

    def _set_minorticks(self, ax, transform: str, ticks: Literal["y", "x"]):
        if ticks == "y":
            yticks = ax.get_yticks()
        else:
            yticks = ax.get_xticks()
        yticks = get_backtransform(transform)(yticks)
        mticks = np.zeros((len(yticks) - 1) * 5)
        for index in range(yticks.size - 1):
            vals = np.linspace(yticks[index], yticks[index + 1], num=5, endpoint=False)
            start = index * 5
            end = index * 5 + 5
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
        tops: list,
        bottoms: list,
        bins: list,
        binwidths: list,
        fillcolors: list[str],
        edgecolors: list[str],
        fill_alpha: float,
        edge_alpha: float,
        hatches: list[str],
        linewidth: float,
        ax: mpl.axes.Axes,
        axis: Literal["x", "y"] = "x",
    ):
        for t, b, b, bw, fc, ec, ht, ln, sub_ax in zip(
            tops,
            bottoms,
            bins,
            binwidths,
            fillcolors,
            edgecolors,
            hatches,
            linewidth,
            ax,
        ):
            if axis == "x":
                sub_ax.bar(
                    x=bins,
                    height=t,
                    bottom=b,
                    width=bw,
                    color=to_rgba(fc, alpha=fill_alpha),
                    edgecolor=to_rgba(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=ht,
                )
            else:
                sub_ax.barh(
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
        ax: plt.Axes,
    ):
        for x, y, mk, mf, me, ms in zip(
            x_data, y_data, marker, markerfacecolor, markeredgecolor, markersize
        ):
            ax.plot(
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
            _, caplines, bars = ax.errorbar(
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
            _, caplines, bars = ax.errorbar(
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
        line_alpha: float,
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
                        to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                    ),
                },
                "medianprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
                "whiskerprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
                "capprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
            }
            if showmeans:
                props["meanprops"] = {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                }
            bplot = ax.boxplot(
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
        facecolors: list[str],
        edgecolors: list[str],
        alpha: float,
        edge_alpha: float,
        linewidth: float,
        width: list[float],
        showmeans: bool,
        showmedians: bool,
        showextrema: bool,
        ax: plt.Axes,
    ):
        for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
            parts = ax.violinplot(
                y,
                positions=x,
                widths=width,
                showmeans=showmeans,
                showmedians=showmedians,
                showextrema=showextrema,
            )
            for body in parts["bodies"]:
                body.set_facecolor(to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs)
                body.set_edgecolor(
                    to_rgba(ecs, alpha=edge_alpha) if ecs != "none" else ecs
                )
                body.set_linewidth(linewidth)
            if showmeans:
                # Matplotlib seems to divide the alpha by 2 so this needs to be equivalent
                parts["cmeans"].set_color(
                    to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
                )
                parts["cmeans"].set_linewidth(linewidth)
            if showmedians:
                parts["cmedians"].set_color(
                    to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
                )
                parts["cmedians"].set_linewidth(linewidth)

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
            color_dict=self._plot_dict["legend_dict"][0],
            alpha=self._plot_dict["legend_dict"][1],
            group=self._plot_dict["group_order"],
            subgroup=self._plot_dict["subgroup_order"],
        )
        ax.plot()
        ax.axis("off")
        ax.legend(handles=handles, frameon=False)
        return fig, ax

    def _plot(self):
        if self.ax is None:
            self.fig, self.ax = self.create_figure()
        for p in self.plot_data:
            plot_func = self.get_plot_func(p.plot_type)
            p_dict = asdict(p)
            p_dict.pot("plot_type")
            plot_func(**p_dict, ax=self.ax)

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
                filename = self._plot_data["y"] if filename == "" else filename
                path = path / f"{filename}.{filetype}"
            else:
                filetype = path.suffix[1:]
        fig.savefig(
            path,
            format=filetype,
            bbox_inches="tight",
            transparent=transparent,
        )


class MPLLinePlotter(MPLPlotter):
    def create_figure(self):
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
            ax = ax.flat
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
                ax.spines[spine].set_linewidth(
                    self.plot_format["axis_format"]["linewidth"]["x"]
                )

        self._set_lims(ax, ydecimals, axis="y")
        self._set_lims(ax, xdecimals, axis="x")

        if self.plot_format["axis_format"]["yminorticks"]:
            self._set_minorticks(ax, self._plot_transforms["ytransform"], ticks="y")

        if self.plot_format["axis_format"]["xminorticks"]:
            self._set_minorticks(ax, self._plot_transforms["xtransform"], ticks="x")

        ax.margins(self.plot_format["figure"]["margins"])
        ax.set_xlabel(
            self._plot_data["xlabel"],
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
            self._plot_data["xlabel"],
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

    def plot(
        self,
        savefig: bool = False,
        path: SavePath = "",
        filetype: str = "svg",
        filename: str | Path = "",
        transparent: bool = False,
    ):

        self._plot(self.plot_data)

        for p in self.plot_data:
            if p.plot_type == "kde" or p.plot_type == "hist":
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
        for index, sub_ax in enumerate(self.ax[: len(self._plot_dict["group_order"])]):
            if self.plot_format["figure"]["projection"] == "rectilinear":
                self.format_rectilinear(sub_ax, ydecimals, xdecimals)
            else:
                self.format_polar(sub_ax)
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
            self.fig.suptitle(
                self._plot_data["title"],
                fontsize=self.plot_format["labels"]["titlesize"],
            )

        if savefig:
            self.savefig(
                path=path, filename=filename, filetype=filetype, transparent=transparent
            )
        return self.fig, self.ax


class MPLCategoricalPlotter(MPLPlotter):
    def create_figure(self):
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_format["figure"]["aspect"]),
            figsize=self.plot_format["figure"]["figsize"],
            layout="constrained",
        )
        return fig, ax

    def plot(
        self,
        savefig: bool = False,
        path: str = "",
        filename: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        self._plot(self.plot_data)

        self.ax.set_xticks(
            ticks=self._plot_dict["x_ticks"],
            labels=self._plot_dict["group_order"],
            rotation=self.plot_format["labels"]["xtick_rotation"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["tick_fontweight"],
            fontsize=self.plot_format["labels"]["ticklabel_size"],
        )
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["left"].set_linewidth(
            self.plot_format["axis_format"]["linewidth"]
        )
        self.ax.spines["bottom"].set_linewidth(
            self.plot_format["axis_format"]["linewidth"]
        )
        if "/" in str(self._plot_data["y"]):
            self._plot_data["y"] = self._plot_data["y"].replace("/", "_")

        self._set_grid(self.ax)

        self._set_lims(self.ax, self.plot_format["axis"]["ydecimals"], axis="y")
        truncate = (
            self.plot_format["axis_format"]["xsteps"][1] != 0
            or self.plot_format["axis_format"]["xsteps"][2]
            != self.plot_format["axis_format"]["xsteps"][0]
        )
        if truncate:
            ticks = self._plot_dict["x_ticks"]
            self.ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])

        if self.plot_format["axis_format"]["yminorticks"]:
            self._set_minorticks(
                self.ax, self._plot_transforms["ytransform"], ticks="y"
            )

        self.ax.set_ylabel(
            self._plot_data["ylabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            rotation=self.plot_format["labels"]["ylabel_rotation"],
        )
        self.ax.set_title(
            self._plot_data["title"],
            fontsize=self.plot_format["labels"]["titlesize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["title_fontweight"],
        )
        self.ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.plot_format["labels"]["ticklabel_size"],
            width=self.plot_format["axis_format"]["tickwidth"],
            length=self.plot_format["axis_format"]["ticklength"],
            labelfontfamily=self.plot_format["labels"]["font"],
        )
        self.ax.margins(x=self.plot_format["figure"]["margins"])

        if "legend_dict" in self._plot_dict:
            handles = self._make_legend_patches(
                color_dict=self._plot_dict["legend_dict"][0],
                alpha=self._plot_dict["legend_dict"][1],
                group=self._plot_dict["group_order"],
                subgroup=self._plot_dict["subgroup_order"],
            )
            self.ax.legend(
                handles=handles,
                bbox_to_anchor=self._plot_dict["legend_anchor"],
                loc=self._plot_dict["legend_loc"],
                frameon=False,
            )

        if savefig:
            self.savefig(
                path=path, filename=filename, filetype=filetype, transparent=transparent
            )
        return self.fig, self.ax


# def _plot_network(
#     graph,
#     marker_alpha: float = 0.8,
#     line_alpha: float = 0.1,
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
#         alpha=line_alpha,
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
