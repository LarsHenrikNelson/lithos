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
"""

from typing import ClassVar

import numpy as np
from typing_extensions import Self

from ...types.basic_types import CategoricalLabels
from ...utils import DataHolder
from ..plot_utils import _create_groupings
from .base import Plot

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
        """
        self._layout_options["pitch"] = pitch
        return self

    def categorical_labels(self, labels: CategoricalLabels = "style1") -> Self:
        """Categorical tick label style (style1: groups, style2: subgroups, style3: both)."""
        self._layout_options["labels"] = labels
        return self

    def _layout_context(self, data: DataHolder) -> dict:
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
