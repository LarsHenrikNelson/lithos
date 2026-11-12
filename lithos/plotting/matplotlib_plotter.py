"""Matplotlib renderers for the legacy plot methods and the spec plot API.

``Plotter`` holds the shared figure/axes machinery and the legacy plot-method
dispatch; ``LinePlotter`` and ``CategoricalPlotter`` add the continuous and
categorical layouts. The spec plotters (:mod:`lithos.plotting.spec.plotter`)
reuse these classes for figure creation while replacing the plot-method
dispatch with element rendering. All axes/tick/label/legend formatting lives
in :mod:`lithos.plotting.formatting`: each plotter delegates ``format_plot()``
(and ``plot_legend()``) to its layout's ``FigureFormatter``.

Only the ``Plotter`` classes and the formatters call matplotlib directly; the
plot classes configure artists through them.
"""

from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from ..types.basic_types import Direction, SavePath
from .formatting import CategoricalFormatter, ContinuousFormatter, FigureFormatter

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
    """Base matplotlib renderer: shared figure, axes and render machinery.

    Attributes:
        plot_data (list): The plot types driving formatting special cases
            (legacy ``PlotTypes`` records or spec layer views).
        plot_format (dict): Label/axis/grid/figure formatting options.
        plot_dict (dict): The layout context (group order, slot positions,
            ...).
        plot_transforms (dict): Scale transforms and tick back-transform
            flags.
        plot_labels (dict): Axis/title label text.
        fig (Figure): The rendered figure.
        axes (list[Axes]): The rendered axes (one per facet/group).
        filetypes (ClassVar[set[str]]): File formats accepted by
            :meth:`savefig`.
        formatter_cls (ClassVar): The ``FigureFormatter`` subclass that
            formats this layout; :meth:`format_plot` delegates to it.
    """

    filetypes: ClassVar[set[str]] = {
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
    formatter_cls: ClassVar[type[FigureFormatter]] = FigureFormatter

    def __init__(
        self,
        plot_data: list,
        plot_dict: dict,
        metadata: dict,
        savefig: bool = False,
        path: SavePath = "",
        filetype: str = "svg",
        filename: str = "",
        axes: Axes | PolarAxes | list[Axes | PolarAxes] | None = None,
        figure: Figure | None = None,
    ):
        """Initialize the renderer and create (or receive) the figure and axes.

        Args:
            plot_data (list): The plot types driving formatting - legacy
                ``PlotTypes`` records or spec layer views (duck-typed
                ``plot_type`` holders).
            plot_dict (dict): The layout context.
            metadata (dict): The plot metadata (format, transforms, labels).
            savefig (bool): Save the rendered figure to disk on ``plot()``.
            path (SavePath): Save directory (or full file path).
            filetype (str): Output file format.
            filename (str): Output file name.
            axes (Axes | PolarAxes | list | None): Existing axes to render
                onto; created via ``create_figure()`` when ``None``.
            figure (Figure | None): Figure owning the existing axes.

        Raises:
            ValueError: If existing axes are passed without a figure.
        """
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
        if self.fig is None:
            raise ValueError("self.fig cannot be None.")

    def create_figure(self) -> tuple[Figure, list[Axes]]:
        """Create and return the figure and axes list (subclass hook)."""
        raise NotImplementedError("Implement create_figure. Must return Figure and list[Axes].")

    def _process_color(self, color, alpha):
        """Convert a color (or list of colors) to RGBA with the given alpha."""
        if color is None:
            color = "none"
        if isinstance(color, list):
            return [to_rgba(c, alpha=alpha) if color != "none" else "none" for c in color]
        else:
            return to_rgba(color, alpha=alpha) if color != "none" else "none"

    def get_plot_func(self, plot_type):
        """Map a plot type name to its legacy plot method."""
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
        elif plot_type == "marker_line":
            return self._marker_line
        else:
            raise ValueError(f"Unsupported plot function: {plot_type}")

    def _make_formatter(self) -> FigureFormatter:
        """Build the layout's formatter over this plotter's state."""
        return self.formatter_cls(
            fig=self.fig,
            axes=self.axes,
            plot_format=self.plot_format,
            plot_labels=self.plot_labels,
            plot_transforms=self.plot_transforms,
            plot_dict=self.plot_dict,
        )

    def format_plot(self):
        """Format the axes through the layout's :class:`FigureFormatter`.

        The metadata dicts are passed by reference (write-through mutations
        such as the kde/hist default limits land on the plot class), and the
        plotted plot-type names drive layout-specific defaults.
        """
        self._make_formatter().format(plot_types=[p.plot_type for p in self.plot_data])

    def _marker_line(
        self,
        ax: list,
        x_data: list,
        y_data: list,
        linecolor: list,
        linewidth: list,
        linestyle: list,
        linealpha: float,
        marker: list,
        markersize: list,
        markerfacecolor: list,
        markeredgecolor: list,
        facet_index: list,
        zorder: list,
        direction: Direction = "horizontal",
        **kwargs,
    ):
        """Draw a line with markers per group (legacy ``marker_line``)."""
        for x, y, ls, lc, lw, fc, ec, m, ms, fi, z in zip(
            x_data,
            y_data,
            linestyle,
            linecolor,
            linewidth,
            markerfacecolor,
            markeredgecolor,
            marker,
            markersize,
            facet_index,
            zorder,
        ):
            if direction == "horizontal":
                y, x = x, y
            ax[0].plot(
                x,
                y,
                linestyle=ls,
                linewidth=lw,
                color=self._process_color(lc, linealpha),
                marker=m,
                markeredgecolor=ec,
                markerfacecolor=fc,
                markersize=ms,
                zorder=z,
            )

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
        ax: list[Axes | PolarAxes],
        zorder: list[int],
        facet_index: list[int] | None = None,
        direction: Direction = "vertical",
        **kwargs,
    ):
        """Draw histogram-style rectangles (bars) per group."""
        if facet_index is None:
            facet_index = [0] * len(heights)
        for t, b, loc, bw, fc, ec, ht, facet, z in zip(
            heights,
            bottoms,
            bins,
            binwidths,
            fillcolors,
            edgecolors,
            hatches,
            facet_index,
            zorder,
        ):
            if direction == "vertical":
                ax[facet].bar(
                    x=loc,
                    height=t,
                    bottom=b,
                    width=bw,
                    color=self._process_color(fc, fill_alpha),
                    edgecolor=self._process_color(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=ht,
                    zorder=z,
                )
            else:
                ax[facet].barh(
                    y=loc,
                    width=t,
                    left=b,
                    height=bw,
                    color=self._process_color(fc, fill_alpha),
                    edgecolor=self._process_color(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=ht,
                    zorder=z,
                )
        return ax

    def _plot_jitter(
        self,
        x_data: list,
        y_data: list,
        marker: str,
        markerfacecolor: list[str],
        markeredgecolor: list[str],
        markeredgewidth: list[float],
        markersize: list[float],
        alpha: float,
        edge_alpha: float,
        zorder: list[int],
        ax: np.ndarray | list,
        direction: Direction = "vertical",
        **kwargs,
    ):
        """Draw per-group jitter markers on a single axes (legacy ``jitter``)."""
        for x, y, mk, mf, me, ms, z in zip(
            x_data, y_data, marker, markerfacecolor, markeredgecolor, markersize, zorder
        ):
            if direction == "horizontal":
                y, x = x, y
            ax[0].plot(
                x,
                y,
                mk,
                markerfacecolor=(self._process_color(mf, alpha)),
                markeredgecolor=(self._process_color(me, edge_alpha) if me != "none" else "none"),
                markersize=ms,
                markeredgewidth=markeredgewidth,
                zorder=z,
            )
        return ax

    def _plot_scatter(
        self,
        x_data: list,
        y_data: list,
        marker: str,
        markerfacecolor: list[str],
        markeredgecolor: list[str],
        markersize: list,
        alpha: float,
        edge_alpha: float,
        linewidth: float,
        facet_index: list[int],
        zorder: list[int],
        ax: list | np.ndarray,
        **kwargs,
    ):
        """Draw per-group scatter markers, facet-aware (legacy ``scatter``)."""
        for x, y, mk, mf, me, ms, facet, z in zip(
            x_data,
            y_data,
            marker,
            markerfacecolor,
            markeredgecolor,
            markersize,
            facet_index,
            zorder,
        ):
            ax[facet].scatter(
                x=x,
                y=y,
                marker=mk,
                c=([self._process_color(x, alpha) for x in mf] if mf != "none" else "none"),
                edgecolor=([self._process_color(x, edge_alpha) for x in me] if me != "none" else "none"),
                s=ms,
                linewidth=linewidth,
                zorder=z,
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
        zorder: list[int],
        ax: list | np.ndarray,
        direction: Direction = "vertical",
        **kwargs,
    ):
        """Draw summary lines with error bars per group (legacy ``summary``)."""
        for xd, yd, e, c, w, z in zip(x_data, y_data, error_data, colors, widths, zorder):
            if direction == "horizontal":
                yd, xd = xd, yd
                e, w = w / 2, e
                capsize1, capsize2 = 0, capsize
            else:
                w = w / 2
                capsize1, capsize2 = capsize, 0
            _, caplines, bars = ax[0].errorbar(
                x=xd,
                y=yd,
                yerr=e,
                c=self._process_color(c, alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=capsize1,
                zorder=z,
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
                xerr=w,
                c=self._process_color(c, alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=capsize2,
                zorder=z,
            )
            for cap in caplines:
                cap.set_solid_capstyle(capstyle)
                cap.set_markeredgewidth(linewidth)
                cap._marker._capstyle = CapStyle(capstyle)
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
        edge_alpha: float,
        fliers: bool,
        linewidth: float,
        width: float,
        show_ci: bool,
        showmeans: bool,
        zorder: list[int],
        ax: list | np.ndarray,
        direction: Direction = "vertical",
        **kwargs,
    ):
        """Draw boxplots per group (legacy ``box``)."""
        for x, y, fcs, ecs, z in zip(x_data, y_data, facecolors, edgecolors, zorder):
            props = {
                "boxprops": {
                    "facecolor": (self._process_color(fcs, alpha)),
                    "edgecolor": (self._process_color(ecs, edge_alpha)),
                    "linewidth": linewidth,
                },
                "medianprops": {
                    "color": (self._process_color(ecs, edge_alpha)),
                    "linewidth": linewidth,
                },
                "whiskerprops": {
                    "color": (self._process_color(ecs, edge_alpha)),
                    "linewidth": linewidth,
                },
                "capprops": {
                    "color": (self._process_color(ecs, edge_alpha)),
                    "linewidth": linewidth,
                },
            }
            if showmeans:
                props["meanprops"] = {"color": (self._process_color(ecs, edge_alpha))}
            _ = ax[0].boxplot(
                y,
                positions=x,
                sym=fliers,
                widths=width,
                notch=show_ci,
                patch_artist=True,
                showmeans=showmeans,
                meanline=showmeans,
                orientation=direction,
                zorder=z,
                **props,
            )

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
        zorder: list[int],
        ax: list | np.ndarray,
        style: str,
        direction: Direction = "vertical",
        **kwargs,
    ):
        """Draw half/full/alternating violin densities per group (legacy ``violin``)."""
        if style in {"left", "right"}:
            zorder = zorder[::-1]
        alt = True
        for x, y, loc, fcs, ecs, z in zip(x_data, y_data, location, facecolors, edgecolors, zorder):
            if style == "left":
                m = y.max() / 2
                left = y * -1 + loc + m
                right = loc + m
            elif style == "right":
                m = y.max() / 2
                left = loc - m
                right = y + loc - m
            elif style == "alternate":
                m = y.max() / 2
                if alt:
                    left = y * -1 + loc + m
                    right = loc + m
                    alt = not alt
                else:
                    left = loc - m
                    right = y + loc - m
                    alt = not alt
            elif style == "full":
                left = y * -1 + loc
                right = y + loc
            if direction == "vertical":
                ax[0].fill_betweenx(
                    x,
                    left,
                    right,
                    facecolor=self._process_color(fcs, alpha),
                    edgecolor=(self._process_color(ecs, edge_alpha)),
                    linewidth=linewidth,
                    zorder=z,
                )
            else:
                ax[0].fill_between(
                    x,
                    left,
                    right,
                    facecolor=self._process_color(fcs, alpha),
                    edgecolor=(self._process_color(ecs, edge_alpha)),
                    linewidth=linewidth,
                    zorder=z,
                )

    def _plot_line(
        self,
        ax: list | np.ndarray,
        x_data: list,
        y_data: list,
        error_data: list,
        facet_index: list[int],
        zorder: list[int],
        marker: list[str | None],
        linecolor: list[str | None],
        fillcolor: list[str | None],
        linewidth: list[float | None],
        linestyle: list[str | None],
        markerfacecolor: list[str | None],
        markeredgecolor: list[str | None],
        fill_between: bool = False,
        fill_under: bool = False,
        direction: Literal["x", "y"] = "y",
        markersize: float | None = None,
        fillalpha: float | None = None,
        linealpha: float | None = None,
        **kwargs,
    ):
        """Draw lines, error bands and fill-under curves per group (legacy ``line`` family)."""
        for x, y, err, ls, lc, lw, fc, mf, me, mk, fi, z in zip(
            x_data,
            y_data,
            error_data,
            linestyle,
            linecolor,
            linewidth,
            fillcolor,
            markerfacecolor,
            markeredgecolor,
            marker,
            facet_index,
            zorder,
        ):
            if not fill_between and not fill_under:
                if err is None:
                    err = 0
                if direction == "horizontal":
                    ax[fi].errorbar(
                        x,
                        y,
                        xerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=lw,
                        linewidth=lw,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                        zorder=z,
                    )
                else:
                    ax[fi].errorbar(
                        x,
                        y,
                        yerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=lw,
                        linewidth=lw,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                        zorder=z,
                    )
            elif fill_between:
                if err is not None:
                    if err.shape[0] == 2:
                        err1 = err[0]
                        err2 = err[1]
                    else:
                        err1 = err
                        err2 = err
                    if direction == "vertical":
                        ax[fi].fill_between(
                            x,
                            y - err1,
                            y + err2,
                            color=self._process_color(fc, fillalpha),
                            linewidth=0,
                            edgecolor="none",
                            zorder=z,
                        )
                    else:
                        ax[fi].fill_betweenx(
                            x,
                            y - err1,
                            y + err2,
                            color=self._process_color(fc, fillalpha),
                            linewidth=0,
                            edgecolor="none",
                            zorder=z,
                        )
                if direction == "horizontal":
                    y, x = x, y
                ax[fi].plot(
                    x,
                    y,
                    linestyle=ls,
                    linewidth=lw,
                    color=lc,
                    alpha=linealpha,
                    zorder=z,
                )
            elif fill_under:
                if direction == "vertical":
                    ax[fi].fill_between(
                        x,
                        y,
                        0,
                        color=self._process_color(fc, fillalpha),
                        linewidth=0,
                        edgecolor="none",
                        zorder=z,
                    )
                else:
                    ax[fi].fill_betweenx(
                        x,
                        y,
                        0,
                        color=self._process_color(fc, fillalpha),
                        linewidth=0,
                        edgecolor="none",
                        zorder=z,
                    )
                if direction == "horizontal":
                    y, x = x, y
                ax[fi].plot(
                    x,
                    y,
                    linestyle=ls,
                    linewidth=lw,
                    color=lc,
                    alpha=linealpha,
                    zorder=z,
                )
            else:
                ax[fi].plot(
                    x,
                    y,
                    linestyle=ls,
                    linewidth=lw,
                    color=lc,
                    alpha=linealpha,
                    zorder=z,
                )

    def plot_legend(self):
        """Render a standalone legend figure for the plot's groups.

        Returns:
            tuple[Figure, Axes]: A figure holding only the legend patches.
        """
        return self._make_formatter().plot_legend()

    def _plot(self):
        """Dispatch every legacy plot type to its plot method."""
        for p in self.plot_data:
            plot_func = self.get_plot_func(p.plot_type)
            p_dict = asdict(p)
            p_dict.pop("plot_type")
            plot_func(**p_dict, ax=self.axes)

    def plot(self):
        """Render the plot data, format the axes and optionally save.

        Returns:
            tuple[Figure, list[Axes]]: The rendered figure and axes.

        Raises:
            ValueError: If no figure was created.
        """
        self._plot()
        self.format_plot()

        if self.fig is None:
            raise ValueError("self.fig must not be None.")

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
        fig: Figure,
        filename: str | None = None,
        filetype: str | None = None,
        transparent: bool = False,
    ):
        """Save a figure to disk.

        Args:
            path (SavePath): Directory (or full path) to save to; the file
                name/type are appended unless the path carries a known file
                suffix.
            fig (Figure): The figure to save.
            filename (str | None): File name used when the path is a
                directory.
            filetype (str | None): File format used when the path is a
                directory.
            transparent (bool): Save with a transparent background.
        """
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, Path):
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
    """Continuous-layout renderer: one axes (or one per group when faceting)."""

    formatter_cls: ClassVar[type[FigureFormatter]] = ContinuousFormatter

    def create_figure(self) -> tuple[Figure, list[Axes]]:
        """Create the figure/axes; faceted plots get one axes per group."""
        if self.plot_format["figure"]["nrows"] is None and self.plot_format["figure"]["ncols"] is None:
            nrows = len(self.plot_dict["group_order"]) if self.plot_dict["facet"] else 1
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
                subplot_kw={
                    "box_aspect": self.plot_format["figure"]["aspect"],
                    "projection": self.plot_format["figure"]["projection"],
                },
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
                subplot_kw={
                    "box_aspect": self.plot_format["figure"]["aspect"],
                    "projection": self.plot_format["figure"]["projection"],
                },
                figsize=self.plot_format["figure"]["figsize"],
                layout="constrained",
            )
            ax = [ax]
        return fig, ax


class CategoricalPlotter(Plotter):
    """Categorical-layout renderer: one axes with slot positions and tick labels."""

    formatter_cls: ClassVar[type[FigureFormatter]] = CategoricalFormatter

    def create_figure(self) -> tuple[Figure, list[Axes]]:
        """Create a single-axes figure."""
        fig, ax = plt.subplots(
            subplot_kw={"box_aspect": self.plot_format["figure"]["aspect"]},
            figsize=self.plot_format["figure"]["figsize"],
            layout="constrained",
        )
        return fig, [ax]


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
