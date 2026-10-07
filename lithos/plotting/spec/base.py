"""Spec-based plot API: the ``Plot`` base class (metadata version 3).

The new user-facing plot API, decoupled from the legacy classes in
:mod:`lithos.plotting.plot_class` (which remain untouched and serve as the
parity reference). A ``Plot`` object is a *pure metadata holder*: it carries no
data and no columns until ``.plot()`` is called, so a configured plot can be
saved (``.save_metadata()``), shared, and replayed against any dataset — like
GraphPad's magic templates.

The API decomposes the concerns the legacy ``grouping()`` methods mixed
together:

- ``.grouping()`` — pure grouping (group/subgroup columns + ordering), shared.
- ``.labels()`` — label text (``None`` = no label, ``""`` = empty label; unset axis
  labels default to the y/x names passed to ``.plot()``).
- ``.label_format()`` — pure label/tick formatting (sizes, fonts, rotations).
- ``.add(transform, *elements)`` — the layer API. The transform and elements
  are held *as-is*; no data is processed until ``.plot()`` runs
  ``_process_data()``, so grouping/columns set after ``.add()`` are honored. All
  layers are processed in one pass, so stacked layers (e.g. subject-level
  ``Identity`` + aggregate-level ``Aggregate``) never disagree. Layers share
  the plot's y/x columns and scale transforms (``.transform()``); plot
  different columns by rendering separate ``Plot`` objects onto the same
  ``figure``/``axes``.
- ``.plot(y, x, data)`` — supplies the data and the plot-level y/x columns.
  ``data`` is any :data:`~lithos.types.basic_types.InputData`; ``y``/``x`` may
  also be bare numpy arrays, in which case a single-group ``DataHolder`` is
  built from them (matplotlib/seaborn-style quick plotting, no grouping).
- Layout-specific settings (faceting for :class:`~lithos.plotting.spec.line.LinePlot`,
  spacing/labels for :class:`~lithos.plotting.spec.categorical.CategoricalPlot`)
  live on the subclasses, not on ``grouping()``.
"""

from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import Self

from ...types.basic_types import InputData, JitterType, SavePath, Transform
from ...types.plot_input import Grouping, Subgrouping
from ...utils import DataHolder
from ..elements import Element
from ..transforms import Transform as StatTransform
from .metadata import (
    build_element,
    build_transform,
    layer_to_json,
    load_spec_metadata,
    save_spec_metadata,
)
from .significance import Significance, resolve_significance


class Unset:
    """Sentinel for a label that has not been set (resolved at plot time)."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = Unset()


class Plot:
    """Base class for the spec plot API.

    Subclasses declare a ``layout`` name, the default per-layer ``position``
    mode, and the allowed ``positions``; they implement ``_layout_context()``
    (the serializable positioning/labeling context handed to the plotter).
    """

    layout: ClassVar[str] = "base"
    default_position: ClassVar[str] = "passthrough"
    positions: ClassVar[tuple[str, ...]] = ("passthrough",)

    def __init__(self):
        self.layers: list[dict] = []
        self.significances: list[Significance] = []
        self.plotter = None

        self._grouping = {"group": None, "subgroup": None, "group_order": None, "subgroup_order": None}
        self._layout_options: dict = {}
        self._labels = {"ylabel": UNSET, "xlabel": UNSET, "title": UNSET, "figure_title": UNSET}
        self.plot_format: dict = {}
        self._plot_transforms: dict = {}

        self.label_format()
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
    ) -> Self:
        """Set the grouping columns and ordering (pure grouping — no layout settings).

        Args:
            group (str | int | None): Group (cluster) column name.
            subgroup (str | int | None): Subgroup column name (splits each
                cluster).
            group_order (Grouping): Explicit group ordering; ``None`` keeps
                data order.
            subgroup_order (Subgrouping): Explicit subgroup ordering;
                ``None`` keeps data order.

        Returns:
            Self: The plot, for chaining.
        """
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
        }

        return self

    def labels(
        self,
        ylabel: str | None | Unset = UNSET,
        xlabel: str | None | Unset = UNSET,
        title: str | None | Unset = UNSET,
        figure_title: str | None | Unset = UNSET,
    ) -> Self:
        """Set the label text, kept separate from the data columns.

        Each label distinguishes three states:

        - unset (default) — the axis labels fall back to the ``y``/``x``
          names passed to ``.plot()``; the titles fall back to none.
        - ``None`` — no label.
        - ``""`` — an explicitly empty label.

        Args:
            ylabel (str | None | Unset): Y axis label; ``UNSET`` falls back
                to the y column name, ``None`` renders no label.
            xlabel (str | None | Unset): X axis label; same states as
                ``ylabel``.
            title (str | None | Unset): Axes title; ``UNSET`` renders no
                title.
            figure_title (str | None | Unset): Figure title; ``UNSET``
                renders no title.

        Returns:
            Self: The plot, for chaining.
        """
        self._labels = {
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
            "figure_title": figure_title,
        }

        return self

    def _resolved_labels(self, y: str | None = None, x: str | None = None) -> dict:
        """Resolve the label states into concrete label text for the plotter.

        Unset axis labels fall back to the ``y``/``x`` names passed to
        ``.plot()`` (blank when the column is not set) and unset titles to no
        title. The ``None`` (no label) and ``""`` (empty label) states are
        preserved so they round-trip through the metadata; the plotter renders
        both blank. Metadata saved without a ``.plot()`` call resolves unset
        axis labels to ``""`` — share the column-name fallback by saving after
        plotting or by setting the labels explicitly.
        """
        labels = dict(self._labels)
        if isinstance(labels["ylabel"], Unset):
            labels["ylabel"] = y if y is not None else ""
        if isinstance(labels["xlabel"], Unset):
            labels["xlabel"] = x if x is not None else ""
        if isinstance(labels["title"], Unset):
            labels["title"] = ""
        if isinstance(labels["figure_title"], Unset):
            labels["figure_title"] = ""
        return labels

    def add(
        self,
        transform: StatTransform,
        *elements: Element,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a layer: one transform plus the elements that render its geometry.

        The transform and elements are held *as-is* — no data is processed.
        Geometry is computed later by ``_process_data()`` (at plot time)
        against the data, grouping and plot-level columns in effect then, so
        ``.grouping()``/``.transform()`` calls made after ``.add()`` are
        honored. Every layer shares the plot's ``y``/``x`` columns and scale
        transforms (``.transform()``); render different columns by drawing
        separate ``Plot`` objects onto the same ``figure``/``axes``.
        ``position`` selects how the resolver maps the layer onto the layout
        when the transform does not supply a coordinate itself.

        On categorical layouts, ``width`` sets the layer's footprint as a
        fraction of its slot. The default ``0.9`` fills the slot while leaving
        a small gap between groups; ``0.5`` uses half the slot and ``0``
        collapses everything onto the slot center. Flat geometry is spread
        randomly within the footprint (``jitter_type`` shapes the
        distribution, ``seed`` makes it reproducible) — except
        single-value-per-group geometry (aggregate centers), which always
        stays anchored at the slot center like the legacy ``summary`` line.
        Transforms that nest by ``unique_id`` place one even column per
        subject across the footprint instead (legacy
        ``jitteru``/``summaryu`` positions); a
        :class:`~lithos.plotting.elements.SummaryLine` spans the footprint
        (flat) or fills its subject column (nested).

        Args:
            transform (Transform): The layer's statistical transform.
            *elements (Element): The elements rendering the transform's
                geometry.
            position (str | None): How the resolver maps the layer onto the
                layout when the transform supplies no coordinate; ``None``
                uses the layout's default.
            width (float): The layer's footprint as a fraction of its slot
                (categorical layouts). Defaults to 0.9.
            jitter_type (JitterType): Shape of the random point spread within
                the footprint.
            seed (int): Seed for the reproducible random spread.

        Returns:
            Self: The plot, for chaining.

        Raises:
            TypeError: If ``transform`` is not a ``Transform`` or an element
                is not an ``Element``.
            ValueError: If no element is given, the position mode is not
                allowed for the layout, or ``width`` is negative.
        """
        if not isinstance(transform, StatTransform):
            raise TypeError(f"add() expects a Transform instance, got {type(transform).__name__!r}.")
        if not elements:
            raise ValueError("add() requires at least one element to render.")
        for element in elements:
            if not isinstance(element, Element):
                raise TypeError(f"add() expects Element instances, got {type(element).__name__!r}.")
        if position is None:
            position = self.default_position
        if position not in self.positions:
            raise ValueError(f"position must be one of {self.positions} for {type(self).__name__}, got {position!r}.")
        if width < 0:
            raise ValueError(f"width must be >= 0, got {width!r}.")

        self.layers.append(
            {
                "transform": transform,
                "elements": list(elements),
                "position": position,
                "width": width,
                "jitter_type": jitter_type,
                "seed": seed,
            }
        )

        return self

    def add_significance(
        self,
        text: str = "*",
        groups: list | None = None,
        x1: float | None = None,
        x2: float | None = None,
        y: float | None = None,
        style: Literal["bracket", "line"] = "bracket",
        gap: float = 0.02,
        step: float = 0.05,
        linecolor: str = "black",
        linewidth: float = 1.5,
        fontsize: float = 12,
        zorder: int | float | None = None,
    ) -> Self:
        """Add a significance bracket (GraphPad asterisks style).

        A bracket is a *layout decoration* like ``add_axline``, not an
        ``.add()`` element: it consumes no transform geometry. Its span is
        resolved at plot time, so ``.grouping()`` calls made after
        ``add_significance()`` are honored.

        ``groups`` names the bracketed groups: plain group values (e.g. ``0``
        or ``"ctrl"``) on grouped plots, ``(group, subgroup)`` tuples on
        subgrouped plots (a plain group value there spans the whole cluster).
        Two entries bracket the pair; three or more span from the leftmost to
        the rightmost slot (a main-effect bracket). Alternatively, ``x1``/``x2``
        give absolute axis positions - on continuous layouts ``groups`` may
        hold a single group key to pick the facet while ``x1``/``x2`` restrict
        the span to that x-window.

        With ``y=None`` the bracket sits automatically one ``gap`` (a fraction
        of the y-range) above the maximum y value under its span, and
        brackets with overlapping spans stack one ``step`` apart. Bracket
        (``"bracket"``) caps descend only to the top of what is plotted; the
        ``"line"`` style draws a plain horizontal line without caps. Brackets
        render before the axis limits are formatted, so autoscaling expands
        to include them (an explicit ``ylim`` still wins).

        Args:
            text (str): Label drawn above the bracket (e.g. ``"*"``, ``"**"``, ``"p=0.01"``).
            groups (list | None): Group keys the bracket spans (strings/ints, or
                ``(group, subgroup)`` tuples).
            x1 (float | None): Left edge of the span in absolute axis positions.
            x2 (float | None): Right edge of the span in absolute axis positions.
            y (float | None): Explicit bracket height; ``None`` computes it from the data.
            style ("bracket" | "line"): ``"bracket"`` (mustache caps) or ``"line"`` (plain line).
            gap (float): Height above the plotted data, as a fraction of the y-range.
            step (float): Spacing between stacked brackets, as a fraction of the y-range.
            linecolor (str): Bracket and text color.
            linewidth (float): Bracket line width.
            fontsize (float): Label font size.
            zorder (int | float | None): Explicit z-order (drawn above the
                layers by default).

        Returns:
            Self: The plot, for chaining.

        Raises:
            ValueError: If neither ``groups`` nor both ``x1``/``x2`` is
                given, only one of ``x1``/``x2`` is given, categorical
                brackets combine ``groups`` with ``x1``/``x2``, ``groups``
                is empty, ``style`` is unknown, or ``gap``/``step`` is
                negative.
        """
        if groups is None and (x1 is None or x2 is None):
            raise ValueError("add_significance() needs groups or both x1 and x2.")
        if (x1 is None) != (x2 is None):
            raise ValueError("add_significance() spans need both x1 and x2.")
        if groups is not None:
            if not groups:
                raise ValueError("groups must be a non-empty list of group keys.")
            if x1 is not None and self.layout == "categorical":
                raise ValueError(
                    "Categorical brackets use groups or x1/x2 positions, not both; "
                    "pass groups (positions come from the layout) or absolute x1/x2 only."
                )
        if style not in ("bracket", "line"):
            raise ValueError(f"style must be 'bracket' or 'line', got {style!r}.")
        if gap < 0 or step < 0:
            raise ValueError("gap and step must be >= 0.")

        self.significances.append(
            Significance(
                text=text,
                groups=groups,
                x1=x1,
                x2=x2,
                y=y,
                style=style,
                gap=gap,
                step=step,
                linecolor=linecolor,
                linewidth=linewidth,
                fontsize=fontsize,
                zorder=zorder,
            )
        )

        return self

    def _process_data(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
    ) -> list[dict]:
        """Compute every layer's geometry against the given data and columns.

        This is the only place transforms are invoked: each held transform is
        called with the plot-level columns and the scale transforms
        (``.transform()``) in effect right now, so calls made after ``.add()``
        are honored. The returned layers are plain serializable dicts (asdict
        transform spec, element specs, fresh geometry) — the backend-agnostic
        artifact that can be handed to any renderer.
        """
        processed = []
        levels = self._levels()
        for layer in self.layers:
            unique_id = getattr(layer["transform"], "unique_id", None)
            if unique_id is not None and unique_id in levels:
                raise ValueError(
                    f"unique_id {unique_id!r} must not be one of the grouping columns {levels!r}; "
                    "nesting keys would collide with the group slots."
                )
            geometry = layer["transform"](
                data,
                y=y,
                x=x,
                levels=levels,
                ytransform=self._plot_transforms["ytransform"],
                xtransform=self._plot_transforms["xtransform"],
            )
            processed.append(
                {
                    "transform": asdict(layer["transform"]),
                    "elements": [element.to_spec() for element in layer["elements"]],
                    "y": y,
                    "x": x,
                    "ytransform": self._plot_transforms["ytransform"],
                    "xtransform": self._plot_transforms["xtransform"],
                    "position": layer["position"],
                    "width": layer["width"],
                    "jitter_type": layer["jitter_type"],
                    "seed": layer["seed"],
                    "geometry": geometry,
                }
            )
        return processed

    def process_data(
        self,
        y: str | np.ndarray | None = None,
        x: str | np.ndarray | None = None,
        data: InputData | None = None,
    ):
        """Compute every layer's geometry without rendering (public ``_process_data``).

        Args:
            y (str | np.ndarray | None): Value column name, or a bare numpy
                array.
            x (str | np.ndarray | None): Independent column name, or a bare
                numpy array.
            data (InputData | None): The dataset holding the columns.

        Returns:
            list[dict]: The processed, backend-agnostic layer list (see
            ``_process_data``).
        """

        y_name, x_name, holder = self._resolve_plot_data(y, x, data)
        return self._process_data(holder, y_name, x_name)

    def _levels(self) -> tuple:
        """Grouping columns (group, subgroup) with ``None`` entries dropped."""
        return tuple(level for level in (self._grouping["group"], self._grouping["subgroup"]) if level is not None)

    def label_format(
        self,
        labelsize: float = 20,
        titlesize: float = 22,
        xticklabel_size: int = 12,
        yticklabel_size: int = 12,
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
        """Set the label/tick formatting: sizes, fonts, weights and rotations.

        Args:
            labelsize (float): Axis label font size. Defaults to 20.
            titlesize (float): Title font size. Defaults to 22.
            xticklabel_size (int): X tick label font size. Defaults to 12.
            yticklabel_size (int): Y tick label font size. Defaults to 12.
            font (str): Font family for all text.
            fontweight (None | str | float): Weight applied to titles, labels
                and ticks when given; ``None`` keeps the individual weights.
            title_fontweight (str | float): Title weight.
            label_fontweight (str | float): Axis label weight.
            tick_fontweight (str | float): Tick label weight.
            xlabel_rotation: X label rotation (``"horizontal"``/
                ``"vertical"`` or degrees).
            ylabel_rotation: Y label rotation.
            xtick_rotation: X tick label rotation.
            ytick_rotation: Y tick label rotation.

        Returns:
            Self: The plot, for chaining.
        """
        if fontweight is not None:
            title_fontweight = fontweight
            label_fontweight = fontweight
            tick_fontweight = fontweight

        label_props = {
            "labelsize": labelsize,
            "titlesize": titlesize,
            "font": font,
            "xticklabel_size": xticklabel_size,
            "yticklabel_size": yticklabel_size,
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
        """Set the axis scales, limits and tick label formatting.

        Args:
            ylim (list | tuple | None): Y axis limits ``(low, high)``.
            xlim (list | tuple | None): X axis limits ``(low, high)``.
            yaxis_lim (list | tuple | None): Y limits used to truncate the y
                spine to the tick range.
            xaxis_lim (list | tuple | None): X limits used to truncate the x
                spine to the tick range.
            yscale ("linear" | "log" | "symlog"): Y axis scale.
            xscale ("linear" | "log" | "symlog"): X axis scale.
            ydecimals (int | None): Y tick label decimals; ``-1`` renders
                integers, ``None`` keeps matplotlib defaults.
            xdecimals (int | None): X tick label decimals.
            xformat ("f" | "e"): X tick label number format.
            yformat ("f" | "e"): Y tick label number format.
            yunits ("degree" | "radian" | "wradian" | None): Y tick angle
                units.
            xunits ("degree" | "radian" | "wradian" | None): X tick angle
                units.

        Returns:
            Self: The plot, for chaining.
        """
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
        """Set the axis line, tick and truncation formatting.

        Args:
            linewidth (float | dict[str, float]): Spine width; a number sets
                left/bottom, a dict keys by spine (``"left"``/``"bottom"``/
                ``"top"``/``"right"``).
            tickwidth (float): Major tick width.
            ticklength (float): Major tick length.
            minor_tickwidth (float): Minor tick width.
            minor_ticklength (float): Minor tick length.
            yminorticks (int): Minor tick subdivisions between major y ticks
                (0 disables).
            xminorticks (int): Minor tick subdivisions between major x ticks.
            ysteps (int | tuple[int, int, int]): Y major tick step, or
                ``(step, start, end)`` indices into the tick sequence for a
                truncated axis.
            xsteps (int | tuple[int, int, int]): X major tick step, or
                ``(step, start, end)``.
            truncate_xaxis (bool): Truncate the x spine to the tick range.
            truncate_yaxis (bool): Truncate the y spine to the tick range.
            style ("default" | "lithos"): Axis formatting style.

        Returns:
            Self: The plot, for chaining.
        """
        if isinstance(ysteps, int):
            ysteps = (ysteps, 0, ysteps)
        if isinstance(xsteps, int):
            xsteps = (xsteps, 0, xsteps)
        if isinstance(linewidth, (int, float)):
            linewidth = {"left": linewidth, "bottom": linewidth, "top": 0, "right": 0}
        elif isinstance(linewidth, dict):
            temp_lw = {"left": 0, "bottom": 0, "top": 0, "right": 0}
            for key, value in linewidth.items():
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
        """Set the figure/axes grid options.

        Args:
            margins (float): Axis margins as a fraction.
            aspect (int | float | None): Axes box aspect (rectilinear
                projection only).
            figsize (tuple[int, int] | None): Figure size; ``None`` scales
                with the axes grid.
            gridspec_kw (dict | None): Matplotlib gridspec keywords.
            nrows (int | None): Axes grid rows (faceted plots).
            ncols (int | None): Axes grid columns.
            projection ("rectilinear" | "polar"): Axes projection.

        Returns:
            Self: The plot, for chaining.
        """
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
        """Set the grid lines.

        Args:
            ygrid (int | float): Y major grid line width (0 hides the grid).
            xgrid (int | float): X major grid line width (0 hides the grid).
            yminor_grid (int | float): Y minor grid line width (0 hides it).
            xminor_grid (int | float): X minor grid line width (0 hides it).
            linestyle (str | tuple): Major grid line style.
            minor_linestyle (str | tuple): Minor grid line style.

        Returns:
            Self: The plot, for chaining.
        """
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

    def transform(
        self,
        ytransform: Transform | None = None,
        back_transform_yticks: bool = False,
        xtransform: Transform | None = None,
        back_transform_xticks: bool = False,
    ) -> Self:
        """Set the scale transforms applied to the plotted values.

        Args:
            ytransform (Transform | None): Transform for y values (e.g.
                ``"log10"``).
            back_transform_yticks (bool): Draw y tick labels back on the
                original scale (named transforms only).
            xtransform (Transform | None): Transform for x values.
            back_transform_xticks (bool): Draw x tick labels back on the
                original scale (named transforms only).

        Returns:
            Self: The plot, for chaining.
        """
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
        """Add horizontal/vertical reference lines across the axes.

        Args:
            linetype ("hline" | "vline"): Line orientation.
            lines (list): Axis positions of the reference lines (a single
                number is wrapped in a list).
            linestyle: Reference line style. Defaults to ``"solid"``.
            linealpha: Reference line alpha. Defaults to 1.
            linecolor: Reference line color. Defaults to ``"black"``.
            linewidth: Reference line width. Defaults to 1.5.
            zorder: Draw order. Defaults to 1.

        Returns:
            Self: The plot, for chaining.

        Raises:
            AttributeError: If ``linetype`` is not ``"hline"`` or
                ``"vline"``.
        """
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

    def get_format(self) -> dict:
        """Return the accumulated formatting settings.

        Returns:
            dict: The ``plot_format`` dictionary (labels, axis, axis_format,
            figure, grid and any reference lines).
        """
        return self.plot_format

    # -- layout hooks ------------------------------------------------------
    def _layout_context(self, data: DataHolder) -> dict:
        """Serializable positioning/labeling context consumed by the plotter."""
        raise NotImplementedError("Subclasses must implement _layout_context().")

    def layout_options(self) -> dict:
        """Return a copy of the layout-specific options.

        Returns:
            dict: The options set by the subclass layout setters (e.g.
            ``pitch``/``labels`` for ``CategoricalPlot``, ``facet`` for
            ``LinePlot``).
        """
        return dict(self._layout_options)

    def _set_layout_options(self, options: dict):
        self._layout_options = dict(options)

    # -- metadata (version 3, JSON) ------------------------------------------
    def metadata(self, y: str | None = None, x: str | None = None) -> dict:
        """Version-3 metadata: grouping, layout, format, labels and layer specs.

        The output is plain JSON. Layers are serialized lazily: each held
        transform becomes an ``asdict`` spec and each element its ``to_spec()``
        dict — no geometry is stored (``load_metadata`` recomputes it by
        replaying ``.add()`` against the data passed to ``.plot()``).

        ``y``/``x`` (the columns passed to ``.plot()``) are recorded in
        ``"data"`` and used to resolve unset axis labels; without them the
        metadata is a pure template — unset labels are stored blank and the
        columns are chosen at the next ``.plot()`` call.

        Args:
            y (str | None): The y column name passed to ``.plot()``.
            x (str | None): The x column name passed to ``.plot()``.

        Returns:
            dict: Version-3 metadata: grouping, layout, format, labels and
            layer specs (no geometry, no data).
        """
        return {
            "version": 3,
            "layout": self.layout,
            "grouping": dict(self._grouping),
            "layout_options": self._layout_options,
            "data": {"y": y, "x": x},
            "labels": self._resolved_labels(y, x),
            "format": self.plot_format,
            "transforms": self._plot_transforms,
            "layers": [layer_to_json(layer) for layer in self.layers],
            "significances": [sig.to_spec() for sig in self.significances],
        }

    def save_metadata(self, file_path: str | Path):
        """Save the plot as a shareable template (no data, no geometry).

        Args:
            file_path (str | Path): Output path; a bare name is stored in
                the configured metadata directory as ``<name>.json``.
        """
        save_spec_metadata(self.metadata(), file_path)

    def load_metadata(self, metadata_path: str | dict | Path) -> Self:
        """Load version-3 JSON metadata onto this plot, replaying the ``.add()`` layers.

        Args:
            metadata_path (str | dict | Path): Saved metadata path (or an
                already-loaded metadata dict).

        Returns:
            Self: The plot, for chaining.

        Raises:
            ValueError: If the metadata is not version 3 or its layout does
                not match this plot's layout.
        """
        metadata = load_spec_metadata(metadata_path)
        if metadata.get("version") != 3:
            raise ValueError("Not a spec (version 3) metadata file.")
        if metadata.get("layout") != self.layout:
            raise ValueError(
                f"Metadata layout {metadata.get('layout')!r} does not match {type(self).__name__} layout {self.layout!r}."
            )

        self._grouping = dict(metadata["grouping"])
        self._set_layout_options(metadata.get("layout_options", {}))
        labels = metadata.get("labels", {})
        self._labels = {key: labels.get(key, "") for key in ("ylabel", "xlabel", "title", "figure_title")}
        for key, value in metadata["format"].items():
            self.plot_format[key] = value
        self._plot_transforms = dict(metadata["transforms"])

        self.layers = []
        for layer in metadata["layers"]:
            self.add(
                build_transform(layer["transform"]),
                *[build_element(spec) for spec in layer["elements"]],
                position=layer["position"],
                width=layer.get("width", 0.0),
                jitter_type=layer.get("jitter_type", "fill"),
                seed=layer.get("seed", 42),
            )
        self.significances = [Significance(**spec) for spec in metadata.get("significances", [])]

        return self

    # -- rendering -----------------------------------------------------------
    def plot(
        self,
        y: str | np.ndarray | None = None,
        x: str | np.ndarray | None = None,
        data: InputData | None = None,
        savefig: bool = False,
        path: SavePath = "",
        filename: str = "",
        filetype: str = "svg",
        save_metadata: bool = False,
        **kwargs,
    ) -> Self:
        """Resolve every layer against the layout and render through the spec plotter.

        This is the only place data enters a plot — the plot object itself is
        a pure metadata holder. Pass ``data`` plus the ``y``/``x`` column
        names, or bare numpy arrays as ``y``/``x`` (an implicit single-group
        ``DataHolder`` is built from the arrays, so grouping is not available
        for array input).

        Args:
            y (str | np.ndarray | None): Value column name (with ``data``)
                or a bare numpy array.
            x (str | np.ndarray | None): Independent column name or a bare
                numpy array.
            data (InputData | None): The dataset holding the columns.
            savefig (bool): Save the rendered figure to disk.
            path (SavePath): Save directory (or full file path); defaults to
                the working directory.
            filename (str): Output file name; defaults to the y column name.
            filetype (str): Output file format (e.g. ``"svg"``).
            save_metadata (bool): Also save the plot template next to the
                figure.
            **kwargs: Extra arguments forwarded to the plotter.

        Returns:
            Self: The plot, for chaining.
        """
        y_name, x_name, holder = self._resolve_plot_data(y, x, data)
        if path == "" or path is None:
            path = Path().cwd()
        elif isinstance(path, str):
            path = Path(path)
        filename_output = filename if filename != "" else (y_name or "")

        context = self._layout_context(holder)

        from .plotter import get_spec_plotter
        from .resolver import resolve_layers

        resolved = resolve_layers(self._process_data(holder, y_name, x_name), context)
        significance = resolve_significance(
            self.significances, holder, y_name, x_name, context, self._plot_transforms["ytransform"]
        )
        self.plotter = get_spec_plotter(context["layout"])(
            layers=resolved,
            significance=significance,
            plot_dict=context,
            metadata=self.metadata(y_name, x_name),
            savefig=savefig,
            path=path,
            filetype=filetype,
            filename=filename_output,
            **kwargs,
        )
        self.plotter.plot()

        if save_metadata and isinstance(path, Path):
            self.save_metadata(path / f"{filename_output}.txt")

        return self

    def _resolve_plot_data(
        self,
        y: str | np.ndarray | None,
        x: str | np.ndarray | None,
        data: InputData | None,
    ) -> tuple[str | None, str | None, DataHolder]:
        """Normalize ``plot()`` inputs into column names and a ``DataHolder``.

        Bare numpy arrays become an implicit single-group ``DataHolder`` with
        columns named ``"y"``/``"x"``; column names require ``data``.
        """
        if y is None and x is None:
            raise ValueError("plot() requires y and/or x (column names together with data, or numpy arrays).")
        if data is not None:
            if isinstance(y, np.ndarray) or isinstance(x, np.ndarray):
                raise ValueError(
                    "y/x must be column names when data is provided; numpy arrays are only valid without data."
                )
            return y, x, DataHolder(data)
        if isinstance(y, str) or isinstance(x, str):
            raise ValueError("y/x column names require data; pass data= or use numpy arrays for y/x.")
        holder_data = {}
        y_name = x_name = None
        if isinstance(y, np.ndarray):
            holder_data["y"] = y
            y_name = "y"
        if isinstance(x, np.ndarray):
            holder_data["x"] = x
            x_name = "x"
        return y_name, x_name, DataHolder(holder_data)
