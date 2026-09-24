"""Continuous-layout spec plot (new API).

``LinePlot`` renders layers on continuous numeric axes; layout-specific
settings are *faceting*, kept separate from grouping (``.facet()``).
"""

from typing import ClassVar

from typing_extensions import Self

from ...types.basic_types import InputData
from ..plot_utils import _create_groupings, create_dict
from .base import Plot


class LinePlot(Plot):
    """Continuous x-axis layout with optional faceting (new spec API)."""

    layout: ClassVar[str] = "continuous"
    default_position: ClassVar[str] = "passthrough"
    positions: ClassVar[tuple[str, ...]] = ("passthrough",)

    def __init__(self, data: InputData):
        super().__init__(data)
        self._layout_options = {"facet": False, "facet_title": False}

    def facet(self, facet: bool = True, facet_title: bool = False) -> Self:
        """Facet by group (one axes per group), optionally titled by group name."""
        self._layout_options["facet"] = facet
        self._layout_options["facet_title"] = facet_title
        return self

    def _layout_context(self) -> dict:
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            self.data,
            self._grouping["group"],
            self._grouping["subgroup"],
            self._grouping["group_order"],
            self._grouping["subgroup_order"],
        )
        if self._layout_options["facet"]:
            # one axes index per group
            loc_dict = create_dict(group_order, unique_groups)
        else:
            loc_dict = create_dict(0, unique_groups)

        return {
            "layout": self.layout,
            "group_order": list(group_order),
            "subgroup_order": list(subgroup_order) if subgroup_order is not None else None,
            "unique_groups": [tuple(group) for group in unique_groups],
            "levels": tuple(levels),
            "loc_dict": loc_dict,
            "facet": self._layout_options["facet"],
            "facet_title": self._layout_options["facet_title"],
        }
