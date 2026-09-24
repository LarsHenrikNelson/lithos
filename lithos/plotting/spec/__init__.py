"""Spec-based plot API (new API, Phase 1).

The new element/transform plot classes, independent of the legacy API in
:mod:`lithos.plotting.plot_class` (which remains untouched):

- :class:`Plot` — shared base: grouping, ``.add(transform, *elements)``,
  version-2 metadata, rendering.
- :class:`LinePlot` — continuous layout (faceting).
- :class:`CategoricalPlot` — categorical positions (dodge/jitter) and
  categorical tick labels.

Example:
    >>> plot = (
    ...     CategoricalPlot(data)
    ...     .grouping(group="grouping_1")
    ...     .plot_data(y="y", ylabel="value")
    ... )
    >>> plot.add(Identity(), position="jitter", Marker())
    >>> plot.add(Aggregate(err_func="sem"), Marker(), ErrorBar())
    >>> plot.plot()
"""

from .base import Plot
from .categorical import CategoricalPlot
from .line import LinePlot

__all__ = ["CategoricalPlot", "LinePlot", "Plot"]
