"""Spec plotters: render resolved layers through the legacy matplotlib machinery.

The spec plotters reuse the legacy ``LinePlotter``/``CategoricalPlotter``
figure creation and axis/label formatting (that is where the categorical
tick/label system lives), but replace the plot-method dispatch with element
rendering: each layer's resolved geometry is drawn once per element spec.

Supported element/geometry combinations for this slice: point/curve geometry
(``Marker``, ``Line``, ``ErrorBar``, ``ErrorBand``, density ``Fill``),
histogram bars (``Bar``, ``Fill``), group-centered bars (``Bar``, ``Fill``)
and ``Annotation``. Combinations needed by the ``Summary`` transform
(box/whisker) arrive with their Phase 2 port. Significance brackets are not
elements: the specs added with ``add_significance()`` render once per bracket
after the layers, still inside ``_plot()`` so the axis limits are formatted
afterwards and expand to include them.
"""

import numpy as np
from matplotlib._enums import CapStyle

from ...types.basic_types import SavePath
from ..matplotlib_plotter import CategoricalPlotter, LinePlotter, Plotter
from ..plot_utils import _process_colors, create_dict
from .resolver import locate_key


class _LayerView:
    """Minimal ``plot_data`` wrapper so legacy ``format_plot()`` can read plot types."""

    def __init__(self, layer: dict):
        name = layer["transform"]["name"]
        # legacy format_plot() special-cases "hist"; the transform is "histogram"
        self.plot_type = "hist" if name == "histogram" else name


class SpecPlotter(Plotter):
    """Base spec renderer: resolved geometry + element specs -> matplotlib artists."""

    def __init__(
        self,
        layers: list[dict],
        plot_dict: dict,
        metadata: dict,
        significance: list[dict] | None = None,
        savefig: bool = False,
        path: SavePath = "",
        filetype: str = "svg",
        filename: str = "",
        axes=None,
        figure=None,
    ):
        self.layers = layers
        self.significance = significance or []
        plot_data = [_LayerView(layer) for layer in layers]
        super().__init__(
            plot_data,
            plot_dict,
            metadata,
            savefig=savefig,
            path=path,
            filetype=filetype,
            filename=filename,
            axes=axes,
            figure=figure,
        )
        # Spec metadata keeps label text ("labels") separate from the data
        # columns ("data"); merge both into the single plot_labels view the
        # legacy formatting reads. A "None" label means no label and renders blank.
        labels = metadata.get("labels", {})
        self.plot_labels = dict(metadata["data"])
        for key in ("ylabel", "xlabel", "title", "figure_title"):
            value = labels.get(key, "")
            self.plot_labels[key] = "" if value is None else value

    def _plot(self):
        for layer_index, layer in enumerate(self.layers):
            for element_index, spec in enumerate(layer["elements"]):
                if spec["zorder"] is not None:
                    zorder = spec["zorder"]
                else:
                    zorder = 2 + layer_index + element_index / 10
                for gkey, geometry in layer["geometry"].items():
                    self._render_element(self.axes[geometry.get("facet", 0)], spec, gkey, geometry, zorder, layer)
        # significance brackets render once per bracket, before format_plot()
        # autoscales the axes, so the axis limits expand to include them.
        for spec in self.significance:
            zorder = spec["zorder"] if spec["zorder"] is not None else 5
            self._render_significance(self.axes[spec["facet"]], spec, zorder)

    # -- element dispatch --------------------------------------------------
    def _render_element(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float, layer: dict):
        element_type = spec["type"]
        if element_type == "marker":
            self._render_marker(ax, spec, gkey, geometry, zorder)
        elif element_type == "line":
            self._render_line(ax, spec, gkey, geometry, zorder)
        elif element_type == "errorbar":
            self._render_errorbar(ax, spec, gkey, geometry, zorder)
        elif element_type == "errorband":
            self._render_errorband(ax, spec, gkey, geometry, zorder)
        elif element_type == "summaryline":
            self._render_summaryline(ax, spec, gkey, geometry, zorder, layer)
        elif element_type == "bar":
            self._render_bar(ax, spec, gkey, geometry, zorder)
        elif element_type == "fill":
            self._render_fill(ax, spec, gkey, geometry, zorder)
        elif element_type == "annotation":
            self._render_annotation(ax, spec, zorder)
        else:
            raise NotImplementedError(f"No renderer for element type {element_type!r} yet.")

    # -- style helpers -------------------------------------------------------
    def _color_dict(self, color):
        """Per-unique-group color dict from an element color parameter."""
        processed = _process_colors(color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"])
        return create_dict(processed, self.plot_dict["unique_groups"])

    def _group_color(self, color_dict: dict, gkey: tuple, alpha: float):
        return self._process_color(locate_key(gkey, color_dict), alpha)

    @staticmethod
    def _bar_width(spec: dict) -> float:
        """Bar width fraction (``Bar`` carries ``barwidth``; ``Fill`` defaults to the same 0.9)."""
        return spec.get("barwidth", 0.9)

    def _layer_width(self, spec: dict, geometry: dict | None = None) -> float:
        """Element width in axis units: ``barwidth`` x the key's resolved extent.

        Flat keys span the full slot; ``unique_id``-nested keys span their
        column (legacy percent/bar widths under unique_id nesting).
        """
        width = geometry.get("extent") if geometry is not None else None
        if width is None:
            width = self.plot_dict.get("width", 1.0)
        return self._bar_width(spec) * width

    @staticmethod
    def _marker_symbol(spec: dict, gkey: tuple, geometry: dict) -> str:
        """Marker symbol for one geometry key.

        A list cycles over the layer's ``unique_id`` values (via the resolved
        ``uid_index``, legacy ``jitteru`` subject markers); a dict is keyed by
        the unique_id value for nested layers and by the group key otherwise.
        """
        marker = spec["marker"]
        if isinstance(marker, str) or marker is None:
            return marker
        if isinstance(marker, list):
            if "uid_index" in geometry:
                return marker[int(geometry["uid_index"]) % len(marker)]
            return marker[0]
        key = geometry.get("uid", gkey)
        if key in marker:
            return marker[key]
        raise ValueError(f"Marker dict has no entry for {key!r}.")

    @staticmethod
    def _curve_values(geometry: dict) -> tuple[np.ndarray, np.ndarray]:
        """x and y-like arrays from point/curve geometry (``y`` or ``center``)."""
        if "x" not in geometry:
            raise NotImplementedError("Element requires x/y point geometry; the resolver could not supply x.")
        if "y" not in geometry and "center" not in geometry:
            raise NotImplementedError("Element requires y or center values in the transform geometry.")
        x = np.atleast_1d(np.asarray(geometry["x"], dtype=float))
        y = np.atleast_1d(np.asarray(geometry.get("y", geometry.get("center")), dtype=float))
        return x, y

    # -- element renderers ---------------------------------------------------
    def _render_marker(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        x, y = self._curve_values(geometry)
        markercolor = self._color_dict(spec["markercolor"])
        edgecolor = self._color_dict(spec["edgecolor"])
        markersize = spec["markersize"]
        size = markersize**2 if isinstance(markersize, (int, float)) else markersize
        ax.scatter(
            x=x,
            y=y,
            marker=self._marker_symbol(spec, gkey, geometry),
            color=self._group_color(markercolor, gkey, spec["alpha"]),
            edgecolors=self._group_color(edgecolor, gkey, spec["edge_alpha"]),
            s=size,
            linewidths=spec["markeredgewidth"],
            zorder=zorder,
        )

    def _render_line(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        x, y = self._curve_values(geometry)
        linecolor = self._color_dict(spec["linecolor"])
        ax.plot(
            x,
            y,
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            color=self._group_color(linecolor, gkey, spec["linealpha"]),
            zorder=zorder,
        )

    def _render_errorbar(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        if geometry.get("error_low") is None or geometry.get("error_high") is None:
            raise NotImplementedError("ErrorBar requires error geometry; give the transform an err_func.")
        x, center = self._curve_values(geometry)
        low = np.atleast_1d(np.asarray(geometry["error_low"], dtype=float))
        high = np.atleast_1d(np.asarray(geometry["error_high"], dtype=float))
        if center.size != x.size or low.size != center.size or high.size != center.size:
            raise ValueError("ErrorBar x, center and error arrays must have the same length.")
        linecolor = self._color_dict(spec["linecolor"])
        container = ax.errorbar(
            x,
            center,
            yerr=np.vstack([low, high]),
            fmt="none",
            color=self._group_color(linecolor, gkey, spec["linealpha"]),
            linewidth=spec["linewidth"],
            capsize=spec["capsize"],
            zorder=zorder,
        )
        for cap in container[1]:
            cap.set_solid_capstyle(spec["capstyle"])
            cap.set_markeredgewidth(spec["linewidth"])
            cap._marker._capstyle = CapStyle(spec["capstyle"])
        for bar in container[2]:
            bar.set_capstyle(spec["capstyle"])

    def _render_errorband(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        if geometry.get("error_low") is None or geometry.get("error_high") is None:
            raise NotImplementedError("ErrorBand requires error geometry; give the transform an err_func.")
        x, center = self._curve_values(geometry)
        low = np.atleast_1d(np.asarray(geometry["error_low"], dtype=float))
        high = np.atleast_1d(np.asarray(geometry["error_high"], dtype=float))
        fillcolor = self._color_dict(spec["fillcolor"])
        ax.fill_between(
            x,
            center - low,
            center + high,
            color=self._group_color(fillcolor, gkey, spec["fillalpha"]),
            edgecolor=self._process_color(spec["edgecolor"], spec["edgealpha"]),
            linewidth=spec["linewidth"],
            zorder=zorder,
        )

    def _render_summaryline(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float, layer: dict):
        """Line spanning the layer ``width`` across the center (legacy ``summary``/``summaryu``).

        Flat keys span ``width`` x the resolved slot extent; ``unique_id``-nested
        keys fill their subject column (legacy ``agg_width=1``). The line ends
        follow ``capstyle`` — the caps are invisible at ``capsize=0``, so the
        style is applied to the bar segments themselves.
        """
        x, center = self._curve_values(geometry)
        extent = geometry.get("extent", self.plot_dict.get("width", 1.0))
        if "uid_index" in geometry:
            half = extent / 2
        else:
            half = layer["width"] * extent / 2
        linecolor = self._color_dict(spec["linecolor"])
        container = ax.errorbar(
            x,
            center,
            xerr=half,
            fmt="none",
            color=self._group_color(linecolor, gkey, spec["linealpha"]),
            linewidth=spec["linewidth"],
            capsize=0,
            zorder=zorder,
        )
        for cap in container[1]:
            cap.set_solid_capstyle(spec["capstyle"])
            cap.set_markeredgewidth(spec["linewidth"])
            cap._marker._capstyle = CapStyle(spec["capstyle"])
        for bar in container[2]:
            bar.set_capstyle(spec["capstyle"])

    def _render_bar(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        if "edges" in geometry:
            # histogram rectangles from bin edges
            x, height, width = geometry["centers"], geometry["height"], geometry["binwidth"]
        elif "position" in geometry and ("center" in geometry or "y" in geometry):
            # group-centered bar at the resolved categorical position
            x = [geometry["position"]]
            height = np.atleast_1d(np.asarray(geometry.get("center", geometry.get("y")), dtype=float))
            width = [self._layer_width(spec, geometry)] * height.size
        else:
            raise NotImplementedError("Bar requires histogram or group-centered geometry.")
        edgecolor = self._color_dict(spec["edgecolor"])
        ax.bar(
            x,
            height,
            width=width,
            facecolor="none",
            edgecolor=self._group_color(edgecolor, gkey, spec["edge_alpha"]),
            linewidth=spec["linewidth"],
            zorder=zorder,
        )

    def _render_fill(self, ax, spec: dict, gkey: tuple, geometry: dict, zorder: float):
        fillcolor = self._color_dict(spec["fillcolor"])
        facecolor = self._group_color(fillcolor, gkey, spec["fillalpha"])
        edgecolor = self._process_color(spec["edgecolor"], spec["edgealpha"])
        edge_width = 0 if spec["edgecolor"] == "none" else 1
        if "edges" in geometry:
            ax.bar(
                geometry["centers"],
                geometry["height"],
                width=geometry["binwidth"],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=edge_width,
                hatch=spec["hatch"],
                zorder=zorder,
            )
        elif "x" in geometry and ("y" in geometry or "center" in geometry):
            # density fill-under curve
            ax.fill_between(
                geometry["x"],
                geometry.get("y", geometry.get("center")),
                0,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=0,
                hatch=spec["hatch"],
                zorder=zorder,
            )
        elif "position" in geometry and ("center" in geometry or "y" in geometry):
            height = np.atleast_1d(np.asarray(geometry.get("center", geometry.get("y")), dtype=float))
            ax.bar(
                [geometry["position"]] * height.size,
                height,
                width=self._layer_width(spec, geometry),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=edge_width,
                hatch=spec["hatch"],
                zorder=zorder,
            )
        else:
            raise NotImplementedError("Fill requires histogram, curve, or group-centered geometry.")

    def _render_annotation(self, ax, spec: dict, zorder: float):
        if spec["x"] is None or spec["y"] is None:
            raise ValueError("Annotation requires x and y coordinates.")
        ax.text(
            spec["x"],
            spec["y"],
            spec["text"],
            fontsize=spec["fontsize"],
            color=spec["color"],
            ha=spec["ha"],
            va=spec["va"],
            rotation=spec["rotation"],
            zorder=zorder,
        )

    def _render_significance(self, ax, spec: dict, zorder: float):
        """Draw one resolved significance bracket (``add_significance``).

        The ``"bracket"`` style is the GraphPad mustache: a horizontal line
        whose caps descend to the computed ``cap_bottoms`` - the top of the
        plotted data (or a nested bracket's line). The ``"line"`` style draws
        the horizontal line only. The label sits above the line, centered.
        """
        x1, x2, y = spec["x1"], spec["x2"], spec["y"]
        if spec["style"] == "bracket":
            bottom_left, bottom_right = spec["cap_bottoms"]
            ax.plot(
                [x1, x1, x2, x2],
                [bottom_left, y, y, bottom_right],
                color=spec["linecolor"],
                linewidth=spec["linewidth"],
                solid_capstyle="butt",
                zorder=zorder,
            )
        else:
            ax.plot(
                [x1, x2],
                [y, y],
                color=spec["linecolor"],
                linewidth=spec["linewidth"],
                solid_capstyle="butt",
                zorder=zorder,
            )
        ax.text(
            (x1 + x2) / 2,
            y,
            spec["text"],
            fontsize=spec["fontsize"],
            color=spec["linecolor"],
            ha="center",
            va="bottom",
            zorder=zorder,
        )


class ContinuousSpecPlotter(SpecPlotter, LinePlotter):
    """Continuous-layout spec plotter (reuses LinePlotter figure/formatting)."""


class CategoricalSpecPlotter(SpecPlotter, CategoricalPlotter):
    """Categorical-layout spec plotter (reuses CategoricalPlotter figure/formatting)."""


def get_spec_plotter(layout: str) -> type[SpecPlotter]:
    """Spec plotter class for a layout name."""
    if layout == "continuous":
        return ContinuousSpecPlotter
    if layout == "categorical":
        return CategoricalSpecPlotter
    raise ValueError(f"Unknown layout {layout!r}.")
