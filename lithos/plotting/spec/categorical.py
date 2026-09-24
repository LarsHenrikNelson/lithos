"""Categorical-layout spec plot (new API).

``CategoricalPlot`` renders layers on categorical axes: group positions
(dodge/jitter) plus the categorical tick-label system. Layout-specific
settings are *spacing* and the *label style*, kept separate from grouping
(``.spacing()`` / ``.categorical_labels()``).

Transforms stay layout-agnostic: the same ``Histogram``/``KDE`` geometry is
interpreted by this layout's resolver/plotter, so density transforms work on
both ``LinePlot`` and ``CategoricalPlot``.
"""

from typing import ClassVar

from typing_extensions import Self

from ...types.basic_types import CategoricalLabels, InputData
from ..plot_utils import _create_groupings, _process_positions
from .base import Plot


class CategoricalPlot(Plot):
    """Categorical positions (dodge/jitter) and categorical tick labels (new spec API)."""

    layout: ClassVar[str] = "categorical"
    default_position: ClassVar[str] = "dodge"
    positions: ClassVar[tuple[str, ...]] = ("passthrough", "dodge", "jitter")

    def __init__(self, data: InputData):
        super().__init__(data)
        self._layout_options = {"group_spacing": 1.0, "labels": "style1"}

    def spacing(self, group_spacing: float = 1.0) -> Self:
        """Spacing between groups on the categorical axis."""
        self._layout_options["group_spacing"] = group_spacing
        return self

    def categorical_labels(self, labels: CategoricalLabels = "style1") -> Self:
        """Categorical tick label style (style1: groups, style2: subgroups, style3: both)."""
        self._layout_options["labels"] = labels
        return self

    def _layout_context(self) -> dict:
        group = self._grouping["group"]
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            self.data,
            group,
            self._grouping["subgroup"],
            self._grouping["group_order"],
            self._grouping["subgroup_order"],
        )
        if group is not None:
            loc_dict, width = _process_positions(
                group_spacing=self._layout_options["group_spacing"],
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
            "ticks": [index for index, _ in enumerate(group_order)],
            "subticks": [float(value) for value in loc_dict.values()],
            "group_spacing": self._layout_options["group_spacing"],
            "labels": self._layout_options["labels"],
        }
