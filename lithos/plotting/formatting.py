"""Axes/tick/label formatting for any matplotlib figure.

Formatting - spines, ticks/tick labels, minor ticks, grids, limits and scale
transforms, margins, axis/title labels, reference lines, legend patches,
polar radian tick labels, the categorical two-tier tick system - is *generic
matplotlib machinery*: every value it needs comes from the serialized format
dicts plus a small layout context, never from plot data or rendering state.
A :class:`FigureFormatter` therefore takes plain ``(fig, axes)`` plus dicts
and can style any matplotlib figure (a public entry point is deliberately
deferred; see ``doc/migration_plan.md``).

The plotters (:mod:`lithos.plotting.matplotlib_plotter`,
:mod:`lithos.plotting.spec.plotter`) compose the layout's formatter via a
``formatter_cls`` ClassVar and call it from ``format_plot()``. The formatter
holds *references* to the metadata dicts, so write-through mutations (the
kde/hist default limits) land on the plot class's format dict and are
captured by metadata save/load.
"""

from fractions import Fraction
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from ..utils import get_backtransform, get_transform


def radian_ticks(ticks: list | np.ndarray, rotate=False):
    pi_symbol = "\u03c0"
    mm = [int(round(180 * i / np.pi)) for i in ticks]
    if rotate:
        mm = [deg if deg <= 180 else deg - 360 for deg in mm]
    jj = [Fraction(deg, 180) if deg != 0 else 0 for deg in mm]
    output = []
    for t in jj:
        sign = "-" if t < 0 else ""
        t = abs(t)
        if t.numerator == 0 or t == 0:
            output.append("0")
        elif t.numerator == 1 and t.denominator == 1:
            output.append(f"{sign}{pi_symbol}")
        elif abs(t.denominator) == 1:
            output.append(f"{t.numerator}{pi_symbol}")
        elif abs(t.numerator) == 1:
            output.append(f"{sign}{pi_symbol}/{t.denominator}")
        else:
            output.append(f"{sign}{t.numerator}{pi_symbol}/{t.denominator}")
    return output


def get_ticks(
    lim,
    axis_lim,
    ticks,
    steps,
):
    if lim[0] is None:
        bottom = ticks[0]
    else:
        bottom = lim[0]
    if lim[1] is None:
        top = ticks[-1]
    else:
        top = lim[1]
    lim = (bottom, top)
    if axis_lim is None:
        axis_lim = (bottom, top)
    if axis_lim[0] is None:
        axis_bottom = lim[0]
    else:
        axis_bottom = axis_lim[0]
    if axis_lim[1] is None:
        axis_top = lim[1]
    else:
        axis_top = axis_lim[1]
    axis_lim = (axis_bottom, axis_top)
    ticks = np.linspace(
        axis_lim[0],
        axis_lim[1],
        steps[0],
    )
    return lim, axis_lim, ticks


class FigureFormatter:
    """Base matplotlib formatter: shared axes/tick/label/legend styling.

    The formatter consumes only ``(fig, axes)`` plus plain dicts - the
    format/label/transform metadata and the layout context - so the same
    machinery can style any matplotlib figure. It never reads plot data:
    plot-type names (for layout-specific defaults like the kde/hist limits)
    are passed explicitly to :meth:`format`.

    Attributes:
        fig (Figure | None): The figure being formatted.
        axes (list[Axes | PolarAxes]): The axes being formatted.
        plot_format (dict): Formatting options (labels, axis, axis_format,
            figure, grid, reference lines); held by reference so mutations
            write through to the plot class's dict.
        plot_labels (dict): Axis/title label text; the ``x``/``y`` keys are
            the value column names and pick the value axis.
        plot_transforms (dict): Scale transforms and tick back-transform flags.
        plot_dict (dict): The layout context (group order, tick slots,
            legend colors, facet flags, ...).
    """

    def __init__(
        self,
        fig: Figure | None,
        axes: Axes | PolarAxes | list[Axes | PolarAxes],
        plot_format: dict,
        plot_labels: dict,
        plot_transforms: dict,
        plot_dict: dict,
    ):
        """Initialize the formatter around a figure and its axes.

        Args:
            fig (Figure | None): The figure being formatted (``None`` checks
                live in the format methods, as they always have).
            axes (Axes | PolarAxes | list): One axes or a list of axes.
            plot_format (dict): Formatting options; held by reference so
                write-through mutations land on the caller's dict.
            plot_labels (dict): Label text and value column names.
            plot_transforms (dict): Scale transforms and back-transform flags.
            plot_dict (dict): The layout context (group order, tick slots,
                legend colors, ...).
        """
        if not isinstance(axes, (list, np.ndarray, tuple)):
            axes = [axes]
        self.fig = fig
        self.axes = axes
        self.plot_format = plot_format
        self.plot_labels = plot_labels
        self.plot_transforms = plot_transforms
        self.plot_dict = plot_dict

    def format(self, plot_types: list[str] | None = None) -> None:
        """Format the axes after the artists are drawn (subclass hook)."""
        raise NotImplementedError("format() not implemented")

    def label_size(self, axis: Literal["x", "y"]) -> float:
        """Tick label size for one axis, falling back to the legacy shared ``ticklabel_size``."""
        labels = self.plot_format["labels"]
        return labels.get(f"{axis}ticklabel_size", labels.get("ticklabel_size", 12))

    def set_grid(self, sub_ax):
        """Apply the major/minor grid settings to one axes."""
        if self.plot_format["grid"]["ygrid"] > 0:
            sub_ax.yaxis.grid(
                linewidth=self.plot_format["grid"]["ygrid"],
                linestyle=self.plot_format["grid"]["linestyle"],
                zorder=1,
            )
        if self.plot_format["grid"]["xgrid"] > 0:
            sub_ax.xaxis.grid(
                linewidth=self.plot_format["grid"]["xgrid"],
                linestyle=self.plot_format["grid"]["linestyle"],
                zorder=1,
            )
        if self.plot_format["grid"]["yminor_grid"] > 0:
            sub_ax.grid(
                visible=True,
                which="minor",
                axis="y",
                linewidth=self.plot_format["grid"]["yminor_grid"],
                linestyle=self.plot_format["grid"]["minor_linestyle"],
                zorder=1,
            )
            sub_ax.minorticks_on()
        if self.plot_format["grid"]["xminor_grid"] > 0:
            sub_ax.grid(
                visible=True,
                which="minor",
                axis="x",
                linewidth=self.plot_format["grid"]["xminor_grid"],
                linestyle=self.plot_format["grid"]["minor_linestyle"],
                zorder=1,
            )
            sub_ax.minorticks_on()
        sub_ax.tick_params(axis="both", which="minor", bottom=False, left=False, zorder=1)

    def draw_axlines(self, line_dict, ax):
        """Draw the hline/vline reference lines on one axes."""
        for ll in line_dict["lines"]:
            if line_dict["linetype"] == "vline":
                ax.axvline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                    linewidth=line_dict["linewidth"],
                    zorder=line_dict["zorder"],
                )
            else:
                ax.axhline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                    linewidth=line_dict["linewidth"],
                    zorder=line_dict["zorder"],
                )

    def set_lims(
        self,
        ax: Axes | PolarAxes,
        lim: tuple[float | int, float | int],
        ticks,
        axis: Literal["x", "y"] = "x",
    ):
        """Set one axis' limits, spine truncation and scale."""
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
                if self.plot_format["axis_format"]["truncate_xaxis"]:
                    start = self.plot_format["axis_format"]["xsteps"][1]
                    end = self.plot_format["axis_format"]["xsteps"][2] - 1
                    ax.spines["bottom"].set_bounds(ticks[start], ticks[end])
                elif self.plot_format["axis"]["xaxis_lim"] is not None:
                    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
            else:
                ax.set_xscale(self.plot_format["axis"]["xscale"])
                ax.set_xlim(left=lim[0], right=lim[1])

    def format_ticklabels(
        self,
        ax: Axes | PolarAxes,
        ticks,
        decimals: int,
        axis: Literal["y", "x"] = "x",
        style: Literal["lithos", "default"] = "lithos",
    ):
        """Format tick positions and labels for one axis (decimals, back-transform, tick window)."""
        if axis == "y":
            if self.plot_format["axis"]["yscale"] not in ["log", "symlog"]:
                if self.plot_transforms.get("back_transform_yticks"):
                    tick_labels = get_backtransform(self.plot_transforms["ytransform"])(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        # This does not work with scientific format
                        tick_labels = np.round(tick_labels, decimals=decimals)
                        dformat = self.plot_format["axis"]["yformat"]
                        tick_labels = [f"{value:.{decimals}{dformat}}" for value in tick_labels]
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
                    fontsize=self.label_size("y"),
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
            else:
                ax.set_yscale(self.plot_format["axis"]["yscale"])
                ax.set_yticks(
                    ticks,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.label_size("y"),
                    rotation=self.plot_format["labels"]["ytick_rotation"],
                )
        else:
            if self.plot_format["axis"]["xscale"] not in ["log", "symlog"]:
                if self.plot_transforms.get("back_transform_xticks"):
                    tick_labels = get_backtransform(self.plot_transforms["xtransform"])(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        # This does not work with scientific format
                        tick_labels = np.round(tick_labels, decimals=decimals)
                        dformat = self.plot_format["axis"]["xformat"]
                        tick_labels = [f"{value:.{decimals}{dformat}}" for value in tick_labels]
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
                    fontsize=self.label_size("x"),
                    rotation=self.plot_format["labels"]["xtick_rotation"],
                )
            else:
                ax.set_xscale(self.plot_format["axis"]["xscale"])
                ax.set_xticks(
                    ticks,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.label_size("x"),
                    rotation=self.plot_format["labels"]["xtick_rotation"],
                )

    def set_axis(
        self,
        ax: Axes | PolarAxes,
        decimals: int,
        axis: Literal["x", "y"] = "x",
        style: Literal["default", "lithos"] = "lithos",
    ):
        """Set ticks, limits and minor ticks for one axis (lithos or default style)."""
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
            self.set_lims(ax=ax, lim=lim, ticks=ticks, axis=axis)
        self.format_ticklabels(ax=ax, ticks=ticks, decimals=decimals, axis=axis, style=style)
        if minorticks != 0:
            self.set_minorticks(
                ax,
                ticks,
                minorticks,
                transform,
                axis=axis,
            )

    def set_minorticks(
        self,
        ax,
        ticks: np.ndarray,
        nticks: int,
        transform: str,
        axis: Literal["y", "x"],
    ):
        """Set evenly spaced minor ticks between the major ticks."""
        ticks = get_backtransform(transform)(ticks)
        mticks = np.zeros((len(ticks) - 1) * nticks)
        for index in range(ticks.size - 1):
            vals = np.linspace(ticks[index], ticks[index + 1], num=nticks + 2, endpoint=True)
            start = index * nticks
            end = index * nticks + nticks
            mticks[start:end] = vals[1:-1]
        if self.plot_format["axis_format"][f"truncate_{axis}axis"]:
            start = self.plot_format["axis_format"][f"{axis}steps"][1] * nticks
            end = self.plot_format["axis_format"][f"{axis}steps"][2] * nticks
        else:
            start = 0
            end = len(mticks)
        if axis == "y":
            ax.set_yticks(
                get_transform(transform)(mticks[start:end]),
                minor=True,
            )
            kwargs = {"left": True}
        else:
            ax.set_xticks(
                get_transform(transform)(mticks[start:end]),
                minor=True,
            )
            kwargs = {"bottom": True}
        ax.tick_params(
            axis=axis,
            which="minor",
            width=self.plot_format["axis_format"]["minor_tickwidth"],
            length=self.plot_format["axis_format"]["minor_ticklength"],
            labelfontfamily=self.plot_format["labels"]["font"],
            **kwargs,
        )

    def legend_patches(self, color_dict, alpha):
        """Legend patches (one per group) from a per-group color dict."""
        legend_patches = []
        for key, value in color_dict.items():
            legend_patches.append(mpatches.Patch(color=to_rgba(value, alpha=alpha), label=key))
        return legend_patches

    def plot_legend(self):
        """Render a standalone legend figure for the plot's groups.

        Returns:
            tuple[Figure, Axes]: A figure holding only the legend patches.
        """
        fig, ax = plt.subplots()

        handles = self.legend_patches(
            color_dict=self.plot_dict["legend_dict"][0],
            alpha=self.plot_dict["legend_dict"][1],
        )
        ax.plot()
        ax.axis("off")
        ax.legend(handles=handles, frameon=False)
        return fig, ax


class ContinuousFormatter(FigureFormatter):
    """Continuous-layout formatter: rectilinear and polar axes, facet titles."""

    def format(self, plot_types: list[str] | None = None) -> None:
        """Format every axes (grids, ticks, spines, labels, reference lines)."""
        for p in plot_types or []:
            if p == "kde" or p == "hist":
                if self.plot_labels["x"] is not None:
                    if self.plot_format["axis"]["ylim"] is None:
                        self.plot_format["axis"]["ylim"] = [0, None]
                else:
                    if self.plot_format["axis"]["xlim"] is None:
                        self.plot_format["axis"]["xlim"] = [0, None]
        ydecimals = self.plot_format["axis"]["ydecimals"]
        xdecimals = self.plot_format["axis"]["xdecimals"]
        for index, sub_ax in enumerate(self.axes[: len(self.plot_dict["group_order"])]):
            self.set_grid(sub_ax)
            if self.plot_format["figure"]["projection"] == "rectilinear":
                self.format_rectilinear(sub_ax, xdecimals, ydecimals)
            else:
                if not isinstance(sub_ax, PolarAxes):
                    raise ValueError("Polar formatting requires polar axes; check the figure projection.")
                self.format_polar(sub_ax)
            if "hline" in self.plot_format:
                self.draw_axlines(self.plot_format["hline"], sub_ax)

            if "vline" in self.plot_format:
                self.draw_axlines(self.plot_format["vline"], sub_ax)

            for tick_axis in ("x", "y"):
                sub_ax.tick_params(
                    axis=tick_axis,
                    which="major",
                    labelsize=self.label_size(tick_axis),
                    width=self.plot_format["axis_format"]["tickwidth"],
                    length=self.plot_format["axis_format"]["ticklength"],
                    labelfontfamily=self.plot_format["labels"]["font"],
                )

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

        if self.plot_labels["figure_title"] != "":
            if self.fig is None:
                raise ValueError("self.fig must not be None.")
            self.fig.suptitle(
                self.plot_labels["figure_title"],
                fontsize=self.plot_format["labels"]["titlesize"],
            )

    def format_rectilinear(self, ax: Axes | PolarAxes, xdecimals: int, ydecimals: int):
        """Format a rectilinear axes: spines, ticks/limits, margins, labels."""
        ax.autoscale()
        for spine, lw in self.plot_format["axis_format"]["linewidth"].items():
            if lw == 0:
                ax.spines[spine].set_visible(False)
            else:
                ax.spines[spine].set_linewidth(lw)

        self.set_axis(
            ax=ax,
            decimals=xdecimals,
            axis="x",
            style=self.plot_format["axis_format"]["style"],
        )
        self.set_axis(
            ax,
            decimals=ydecimals,
            axis="y",
            style=self.plot_format["axis_format"]["style"],
        )

        ax.margins(self.plot_format["figure"]["margins"])
        ax.set_xlabel(
            self.plot_labels["xlabel"],
            fontsize=self.plot_format["labels"]["labelsize"],
            fontweight=self.plot_format["labels"]["label_fontweight"],
            fontfamily=self.plot_format["labels"]["font"],
            rotation=self.plot_format["labels"]["xlabel_rotation"],
        )

    def format_polar(self, ax: PolarAxes):
        """Format a polar axes (radian tick labels, spine, r-max)."""
        if self.plot_format["axis"]["xunits"] == "radian" or self.plot_format["axis"]["xunits"] == "wradian":
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
                fontsize=self.label_size("x"),
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
        decimals = self.plot_format["axis"]["ydecimals"]
        if self.plot_format["axis_format"]["style"] == "lithos":
            _, _, ticks = get_ticks(
                lim=self.plot_format["axis"]["ylim"],
                axis_lim=self.plot_format["axis"]["yaxis_lim"],
                ticks=ticks,
                steps=self.plot_format["axis_format"]["xsteps"],
            )
        if decimals is not None:
            if decimals == -1:
                tick_labels = ticks.astype(int)
            else:
                # This does not work with scientific format
                tick_labels = np.round(ticks, decimals=decimals)
                dformat = self.plot_format["axis"]["yformat"]
                tick_labels = [f"{value:.{decimals}{dformat}}" for value in tick_labels]
        else:
            tick_labels = ticks
        ax.set_yticks(
            ticks,
            tick_labels,
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["tick_fontweight"],
            fontsize=self.label_size("y"),
            rotation=self.plot_format["labels"]["ytick_rotation"],
        )
        # self.set_axis(
        #     ax,
        #     decimals=self.plot_format["axis"]["ydecimals"],
        #     axis="y",
        #     style=self.plot_format["axis_format"]["style"],
        # )


class CategoricalFormatter(FigureFormatter):
    """Categorical-layout formatter: two-tier ticks, labels, legend, margins."""

    def format(self, plot_types: list[str] | None = None) -> None:
        """Format the categorical axes (spines, grid, ticks, labels, title)."""
        ax = self.axes[0]

        direction = "vertical" if self.plot_labels["x"] is None else "horizontal"

        for spine, lw in self.plot_format["axis_format"]["linewidth"].items():
            if lw == 0:
                ax.spines[spine].set_visible(False)
            else:
                ax.spines[spine].set_linewidth(lw)
        self.set_grid(ax)

        if direction == "vertical":
            self.set_axis(
                ax,
                self.plot_format["axis"]["ydecimals"],
                axis="y",
                style=self.plot_format["axis_format"]["style"],
            )
            self.set_categorical_axis(ax)
            ax.set_ylabel(
                self.plot_labels["ylabel"],
                fontsize=self.plot_format["labels"]["labelsize"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["label_fontweight"],
                rotation=self.plot_format["labels"]["ylabel_rotation"],
            )
        else:
            self.set_axis(
                ax,
                self.plot_format["axis"]["xdecimals"],
                axis="x",
                style=self.plot_format["axis_format"]["style"],
            )
            self.set_categorical_axis(ax, axis="y")
            ax.set_xlabel(
                self.plot_labels["ylabel"],
                fontsize=self.plot_format["labels"]["labelsize"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["label_fontweight"],
                rotation=self.plot_format["labels"]["xlabel_rotation"],
            )

        if "hline" in self.plot_format:
            self.draw_axlines(self.plot_format["hline"], ax)

        if "vline" in self.plot_format:
            self.draw_axlines(self.plot_format["vline"], ax)

        ax.set_title(
            self.plot_labels["title"],
            fontsize=self.plot_format["labels"]["titlesize"],
            fontfamily=self.plot_format["labels"]["font"],
            fontweight=self.plot_format["labels"]["title_fontweight"],
        )
        for tick_axis in ("x", "y"):
            ax.tick_params(
                axis=tick_axis,
                which="major",
                labelsize=self.label_size(tick_axis),
                width=self.plot_format["axis_format"]["tickwidth"],
                length=self.plot_format["axis_format"]["ticklength"],
                labelfontfamily=self.plot_format["labels"]["font"],
            )
        ax.margins(x=self.plot_format["figure"]["margins"])

        if "legend_dict" in self.plot_dict:
            handles = self.legend_patches(
                color_dict=self.plot_dict["legend_dict"][0],
                alpha=self.plot_dict["legend_dict"][1],
            )
            ax.legend(
                handles=handles,
                bbox_to_anchor=self.plot_dict["legend_anchor"],
                loc=self.plot_dict["legend_loc"],
                frameon=False,
            )

    def set_categorical_axis(self, ax, axis="x"):
        """Set the categorical axis: two-tier group/subgroup ticks and labels.

        Args:
            ax: The axes to format.
            axis ("x" | "y"): Which axis is categorical.

        Raises:
            ValueError: If the label style is unknown.
        """
        bottom_labels = None
        bottom_ticks = None
        if self.plot_dict["labels"] == "style1":
            top_labels = self.plot_dict["group_order"]
            top_ticks = self.plot_dict["ticks"]
        elif self.plot_dict["labels"] == "style2":
            top_labels = self.plot_dict["subgroup_order"]
            top_ticks = self.plot_dict["subticks"]
            n_repeats = len(top_ticks) // len(top_labels)
            top_labels = np.tile(top_labels, n_repeats)
        elif self.plot_dict["labels"] == "style3":
            top_labels = self.plot_dict["subgroup_order"]
            bottom_labels = self.plot_dict["group_order"]
            if axis == "x":
                bottom_labels = [f"\n{i}" for i in bottom_labels]
            else:
                bottom_labels = [f"{i}" for i in bottom_labels]
            top_ticks = self.plot_dict["subticks"]
            bottom_ticks = self.plot_dict["ticks"]
            n_repeats = len(top_ticks) // len(top_labels)
            top_labels = np.tile(top_labels, n_repeats)
        else:
            raise ValueError("Labels must style1, style2, style3.")
        span = max(top_ticks) - min(top_ticks) + self.plot_dict["group_spacing"] / 2
        margin = self.plot_format["figure"]["margins"] * span
        if axis == "x":
            ax.set_xticks(
                ticks=top_ticks,
                labels=top_labels,
                rotation=self.plot_format["labels"]["xtick_rotation"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["tick_fontweight"],
                fontsize=self.label_size("x"),
            )
            ax.set_xlim(
                min(top_ticks) - self.plot_dict["group_spacing"] / 2 - margin,
                max(top_ticks) + self.plot_dict["group_spacing"] / 2 + margin,
            )
            if self.plot_format["axis_format"]["truncate_xaxis"]:
                ticks = self.plot_dict["ticks"]
                ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
            if bottom_labels is not None:
                sec = ax.secondary_xaxis(location=0)
                sec.set_xticks(
                    bottom_ticks,
                    labels=bottom_labels,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.label_size("x"),
                )
                sec.tick_params(axis="x", bottom=False)
        else:
            ax.set_yticks(
                ticks=top_ticks,
                labels=top_labels,
                rotation=self.plot_format["labels"]["xtick_rotation"],
                fontfamily=self.plot_format["labels"]["font"],
                fontweight=self.plot_format["labels"]["tick_fontweight"],
                fontsize=self.label_size("y"),
            )
            ax.set_ylim(
                min(top_ticks) - self.plot_dict["group_spacing"] / 2 - margin,
                max(top_ticks) + self.plot_dict["group_spacing"] / 2 + margin,
            )
            if self.plot_format["axis_format"]["truncate_yaxis"]:
                ticks = self.plot_dict["ticks"]
                ax.spines["left"].set_bounds(ticks[0], ticks[-1])
            if bottom_labels is not None:
                sec = ax.secondary_yaxis(location="left")
                sec.set_yticks(
                    bottom_ticks,
                    labels=bottom_labels,
                    fontfamily=self.plot_format["labels"]["font"],
                    fontweight=self.plot_format["labels"]["tick_fontweight"],
                    fontsize=self.label_size("y"),
                )
                sec.tick_params(axis="y", left=False)
