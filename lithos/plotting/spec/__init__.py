"""Spec-based plot API (new API, Phase 1).

The new element/transform plot classes, independent of the legacy API in
:mod:`lithos.plotting.plot_class` (which remains untouched):

- :class:`Plot` — shared base: grouping, ``.add(transform, *elements)``,
  version-3 metadata, rendering. A ``Plot`` object holds *no data* — it is a
  pure metadata holder, so configured plots can be saved, shared, and
  replayed against any dataset.
- :class:`LinePlot` — continuous layout (faceting).
- :class:`CategoricalPlot` — categorical positions (slot centers plus
  per-layer jitter/unique_id-column spread) and categorical tick labels.

Example:
    >>> plot = (
    ...     CategoricalPlot()
    ...     .grouping(group="grouping_1")
    ...     .labels(ylabel="value")
    ... )
    >>> plot.add(Identity(), Marker(), width=0.5, seed=30)  # jitter, half slot
    >>> plot.add(Aggregate(err_func="sem"), SummaryLine(), ErrorBar())
    >>> plot.plot(y="y", data=df)

``data`` plus column names, or bare numpy arrays (``plot(y=np.array(...))``),
enter only through ``.plot()``.
"""

from .base import Plot
from .categorical import CategoricalPlot
from .line import LinePlot
from .significance import Significance

__all__ = ["CategoricalPlot", "LinePlot", "Plot", "Significance"]
