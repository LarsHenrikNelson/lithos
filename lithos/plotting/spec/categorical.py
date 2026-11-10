"""Categorical-layout spec plot (new API).

``CategoricalPlot`` renders layers on categorical axes: group slot positions
plus the categorical tick-label system. Layout-specific settings are *pitch*
(the distance between group centers; clusters span a fixed 1 unit) and the
*label style*, kept separate from grouping (``.spacing()`` /
``.categorical_labels()``).

Per-layer point spread within a slot (jitter, even unique_id columns) is set
with the ``width``/``jitter_type``/``seed`` arguments of ``.add()``, resolved
by the position resolver.

Transforms stay layout-agnostic: the same ``Histogram``/``KDE`` geometry is
interpreted by this layout's resolver/plotter, so density transforms work on
both ``LinePlot`` and ``CategoricalPlot``.

The Phase 2.5 sugar verbs for categorical aggregates live here: ``summary()``
(the GraphPad summary line), ``bar()`` and ``box()``; points (``scatter()``)
and densities (``kde()`` = violins) come mixed in from
:mod:`lithos.plotting.spec.methods`.
"""

from typing import ClassVar, Literal

import numpy as np
from typing_extensions import Self

from ...types.basic_types import CategoricalLabels, JitterType
from ...types.plot_input import Agg, AlphaRange, ColorParameters, Error
from ...utils import DataHolder
from ..elements import Bar, ErrorBar, Fill, Line, Marker, SummaryLine
from ..plot_utils import _create_groupings
from ..transforms import Aggregate, Summary
from .base import Plot
from .methods import _resolve_role, _with_intent

CLUSTER_WIDTH = 1.0
"""Width (axis units) a group cluster occupies: subgroup slots split it evenly."""


def _process_spec_positions(pitch, group_order, subgroup_order=None):
    """Pitch-based positions for the spec categorical layout.

    Group centers sit ``i * pitch`` apart, while every cluster spans a fixed
    :data:`CLUSTER_WIDTH` (1 unit) — unlike the legacy ``group_spacing``
    parameter, which doubled as the cluster width. Keeping the cluster width
    fixed means group layers can never overlap (``CLUSTER_WIDTH <= pitch``)
    and per-layer ``width`` fractions in ``.add()`` are always fractions of
    one familiar unit slot.
    """
    group_loc = {key: float(index) * pitch for index, key in enumerate(group_order)}
    if subgroup_order is not None:
        width = CLUSTER_WIDTH / len(subgroup_order)
        start = (CLUSTER_WIDTH / 2) - (width / 2)
        sub_loc = np.linspace(-start, start, len(subgroup_order))
        subgroup_loc = {key: value for key, value in zip(subgroup_order, sub_loc)}
        loc_dict = {}
        for i, i_value in group_loc.items():
            for j, j_value in subgroup_loc.items():
                loc_dict[(i, j)] = float(i_value + j_value)

    else:
        loc_dict = {(key,): value for key, value in group_loc.items()}
        width = 1.0
    return loc_dict, width


class CategoricalPlot(Plot):
    """Categorical positions (slot centers + per-layer spread) and tick labels (new spec API)."""

    layout: ClassVar[str] = "categorical"
    default_position: ClassVar[str] = "dodge"
    positions: ClassVar[tuple[str, ...]] = ("passthrough", "dodge")

    def __init__(self):
        super().__init__()
        self._layout_options = {"pitch": 1.0, "labels": "style1"}

    def spacing(self, pitch: float = 1.0) -> Self:
        """Distance between group centers on the categorical axis.

        Clusters always span a fixed 1 unit (:data:`CLUSTER_WIDTH`), so values
        above 1 add whitespace between groups; values below 1 squeeze them
        together. The spread of points *within* a slot is set per layer with
        the ``width`` argument of ``.add()``, not here.

        Args:
            pitch (float): Distance between group centers in axis units.
                Defaults to 1.0.

        Returns:
            Self: The plot, for chaining.
        """
        self._layout_options["pitch"] = pitch
        return self

    def categorical_labels(self, labels: CategoricalLabels = "style1") -> Self:
        """Categorical tick label style (style1: groups, style2: subgroups, style3: both).

        Args:
            labels (CategoricalLabels): Tick label style - ``"style1"``
                (group labels), ``"style2"`` (subgroup labels) or
                ``"style3"`` (both).

        Returns:
            Self: The plot, for chaining.
        """
        self._layout_options["labels"] = labels
        return self

    # -- Phase 2.5 sugar verbs (categorical aggregates) -----------------------

    def summary(
        self,
        *,
        unique_id: str | None = None,
        func: Agg = "mean",
        err_func: Error | None = None,
        agg_func: Agg | None = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        line: SummaryLine | bool | None = None,
        error: ErrorBar | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a GraphPad-style summary layer: a short line per group (legacy ``summary``/``summaryu``).

        Every group draws a :class:`~lithos.plotting.elements.SummaryLine`
        across its slot (the layer ``width`` spans it); with ``err_func`` an
        :class:`~lithos.plotting.elements.ErrorBar` joins from the same
        geometry. With ``unique_id`` each subject claims its own even column
        (legacy ``summaryu``); ``agg_func`` re-aggregates the per-subject
        centers back into one line per group - the transform applies
        ``agg_func`` at the second level, where the legacy ``summaryu``
        re-applied ``func`` (they agree when ``func == agg_func``).

        Deliberate divergences from the legacy methods: the verb keeps the
        element defaults (``glasbey_category10`` lines - legacy ``summary``
        was black) and draws no error without ``err_func`` (legacy defaulted
        ``err_func="sem"``).

        Args:
            unique_id (str | None): Subject column; one summary line per
                subject, one even column each.
            func (Agg): Center aggregate applied within each group (or each
                subject first, when ``unique_id`` is given).
            err_func (Error | None): Error around the center from the same
                values; adds the ``ErrorBar`` role.
            agg_func (Agg | None): Second-level aggregation across the
                per-subject centers (requires ``unique_id``).
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            line (SummaryLine | bool | None): Center-line role - an explicit
                ``SummaryLine`` wins completely, ``False`` drops it,
                ``None`` builds the default.
            error (ErrorBar | bool | None): Error role - an explicit
                ``ErrorBar`` wins completely, ``False`` drops it, ``None``
                builds the default (on only when ``err_func`` is given).
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): The line's footprint as a fraction of its slot.
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.

        Returns:
            Self: The plot, for chaining.
        """
        transform = Aggregate(func=func, err_func=err_func, agg_func=agg_func, unique_id=unique_id)
        line_element = _resolve_role(line, lambda: _with_intent(SummaryLine(), color, alpha))
        error_element = _resolve_role(
            error, lambda: _with_intent(ErrorBar(), color, alpha), default_on=err_func is not None
        )
        elements = [element for element in (line_element, error_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
        )

    def bar(
        self,
        *,
        func: Agg = "mean",
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        fill: Fill | bool | None = None,
        bar: Bar | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a bar layer: one rectangle per group from the group center (legacy ``bar``).

        Faces render through :class:`~lithos.plotting.elements.Fill` and
        outlines through :class:`~lithos.plotting.elements.Bar`, both from
        the single ``Aggregate`` center. With ``unique_id`` each subject
        claims its own even column (one thinner bar per subject);
        ``agg_func`` re-aggregates the per-subject ``func`` centers back
        into one bar per group. Bars start at the axis origin; the layer
        ``width`` is the slot footprint fraction (legacy ``barwidth``).

        The verb omits ``err_func`` on purpose (the legacy ``bar`` had no
        errors): errors over bars are a separate ``.add(Aggregate(err_func=...),
        ErrorBar())`` layer.

        Args:
            func (Agg): Height aggregate applied within each group (or each
                subject first, when ``unique_id`` is given).
            unique_id (str | None): Subject column; one bar per subject.
            agg_func (Agg | None): Second-level aggregation across the
                per-subject heights (requires ``unique_id``).
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields (faces and outlines; never
                edges); ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            fill (Fill | bool | None): Bar-face role - an explicit ``Fill``
                wins completely, ``False`` drops it, ``None`` builds the default.
            bar (Bar | bool | None): Bar-outline role - an explicit ``Bar``
                wins completely, ``False`` drops it, ``None`` builds the default.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): The bar's footprint as a fraction of its slot.
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.

        Returns:
            Self: The plot, for chaining.
        """
        transform = Aggregate(func=func, agg_func=agg_func, unique_id=unique_id)
        fill_element = _resolve_role(fill, lambda: _with_intent(Fill(), color, alpha))
        bar_element = _resolve_role(bar, lambda: _with_intent(Bar(), color, alpha))
        elements = [element for element in (fill_element, bar_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
        )

    def box(
        self,
        *,
        whisker: Literal["quantiles", "minmax", "none"] = "quantiles",
        whisker_quantiles: tuple[float, float] = (5, 95),
        notch: bool = False,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        fill: Fill | bool | None = None,
        bar: Bar | bool | None = None,
        line: Line | bool | None = None,
        error: ErrorBar | bool | None = None,
        marker: Marker | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a box-plot layer (legacy ``CategoricalPlot.box``): one box per group.

        The roles draw from one ``Summary`` geometry: whiskers
        (:class:`~lithos.plotting.elements.ErrorBar`) from
        ``whisker_low``/``whisker_high``, the body face (:class:`Fill`) and
        outline (:class:`Bar`) for q1..q3 - both notched when ``notch`` is
        set (matplotlib's notch polygon, McGill 1.58 notch bounds), the
        median line (:class:`Line`) across the span, and the group mean
        (:class:`Marker`, opt-in - legacy ``showmeans``). The layer
        ``width`` is the box footprint fraction (legacy ``width``).

        Deliberate divergences from the legacy method: whiskers are
        quantiles (5/95 by default) or min/max rather than matplotlib's
        1.5*IQR extents, and fliers do not exist in box geometry - draw the
        raw points with a separate ``.scatter()`` layer. Legacy
        ``show_ci``/``showmeans``/``fliers`` become ``notch``/``marker``
        role / a separate layer.

        Args:
            whisker ("quantiles" | "minmax" | "none"): What the whisker
                extents represent; ``"none"`` turns the whisker role off.
            whisker_quantiles (tuple[float, float]): (low, high) percentiles
                used by ``whisker="quantiles"``.
            notch (bool): Draw the notch bounds around the median (legacy
                ``show_ci``); the face, outline and median line adopt the
                notched form.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            fill (Fill | bool | None): Box-face role - an explicit ``Fill``
                wins completely, ``False`` drops it, ``None`` builds the default.
            bar (Bar | bool | None): Box-outline role - an explicit ``Bar``
                wins completely, ``False`` drops it, ``None`` builds the default.
            line (Line | bool | None): Median-line role - an explicit
                ``Line`` wins completely, ``False`` drops it, ``None`` builds
                the default.
            error (ErrorBar | bool | None): Whisker role - an explicit
                ``ErrorBar`` wins completely, ``False`` drops it, ``None``
                builds the default (off when ``whisker="none"``).
            marker (Marker | bool | None): Mean-point role - absent by
                default; ``True`` accepts the intent-styled default, an
                explicit ``Marker`` wins completely.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): The box footprint as a fraction of its slot.
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.

        Returns:
            Self: The plot, for chaining.
        """
        transform = Summary(whisker=whisker, whisker_quantiles=whisker_quantiles, notch=notch)
        error_element = _resolve_role(
            error, lambda: _with_intent(ErrorBar(), color, alpha), default_on=whisker != "none"
        )
        fill_element = _resolve_role(fill, lambda: _with_intent(Fill(), color, alpha))
        bar_element = _resolve_role(bar, lambda: _with_intent(Bar(), color, alpha))
        line_element = _resolve_role(line, lambda: _with_intent(Line(), color, alpha))
        marker_element = _resolve_role(marker, lambda: _with_intent(Marker(), color, alpha), default_on=False)
        elements = [
            element
            for element in (error_element, fill_element, bar_element, line_element, marker_element)
            if element is not None
        ]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
        )

    def _layout_context(self, data: DataHolder) -> dict:
        """Build the categorical layout context (slot positions, ticks, label style)."""
        group = self._grouping["group"]
        pitch = self._layout_options["pitch"]
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            data,
            group,
            self._grouping["subgroup"],
            self._grouping["group_order"],
            self._grouping["subgroup_order"],
        )
        if group is not None:
            loc_dict, width = _process_spec_positions(
                pitch=pitch,
                group_order=group_order,
                subgroup_order=subgroup_order,
            )
        else:
            group_order = [""]
            subgroup_order = [""]
            unique_groups = [("",)]
            loc_dict = {("",): 0.0}
            width = 1.0

        return {
            "layout": self.layout,
            "group_order": list(group_order),
            "subgroup_order": list(subgroup_order) if subgroup_order is not None else None,
            "unique_groups": [tuple(group) for group in unique_groups],
            "levels": tuple(levels),
            "loc_dict": loc_dict,
            "width": width,
            "ticks": [index * pitch for index in range(len(group_order))],
            "subticks": [float(value) for value in loc_dict.values()],
            "pitch": pitch,
            # the legacy axis formatting treats "group_spacing" as the cluster
            # width (used for axis margins); clusters are fixed at 1 unit here
            "group_spacing": CLUSTER_WIDTH,
            "labels": self._layout_options["labels"],
        }
