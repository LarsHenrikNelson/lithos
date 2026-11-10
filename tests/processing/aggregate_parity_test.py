"""Parity: ``Aggregate``/``Identity`` geometry vs the legacy family-2 processors.

The legacy categorical methods (``summary``/``summaryu``/``bar``) and line
methods (``line``/``aggline``) share the stats layer with the ``Aggregate``
transform (and the raw-series branch with ``Identity``), so their processed
center/error values must match for the same data. Positioning does not take
part in parity - the resolver pins the slot and even-column placement
(``tests/spec_plot_test.py``); the legacy outputs hold ``loc_dict`` positions.
"""

import numpy as np
import pytest

from lithos import Aggregate, CategoricalPlot, Identity, LinePlot
from lithos.plotting.processing import CategoricalProcessor, LineProcessor
from lithos.utils import DataHolder

_MARKERS = ("o", "X", "^", "s", "d")
_HATCHES = ("/", "o", "-", "*", "+")

FIXTURES = [
    ("one_grouping", None),
    ("two_grouping", "grouping_2"),
]
UID_FIXTURES = [
    ("one_grouping_with_unique_ids", None),
    ("two_grouping_with_unique_ids", "grouping_2"),
]


def _repeated_x_data():
    """Two groups with several y values per (group, x) bin (every bin n >= 2).

    The session fixtures hold one row per unique x per group; the per-x
    aggregation would degenerate to n=1 bins there (sem of one value).
    """
    return {
        "grouping_1": [0] * 7 + [1] * 6,
        "x": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0] + [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        "y": [1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0] + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }


def _levels(subgroup):
    return ("grouping_1",) if subgroup is None else ("grouping_1", "grouping_2")


def _as_group_key(label):
    return tuple(label) if isinstance(label, tuple) else (label,)


class TestSummaryParity:
    """Parity: ``Aggregate(func, err_func)`` group centers vs the legacy ``_summary``."""

    def _legacy(self, data, subgroup, **kwargs):
        plot = CategoricalPlot(data).grouping(group="grouping_1", subgroup=subgroup).summary(**kwargs).plot_data(y="y")
        processor = CategoricalProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize("fixture, subgroup", FIXTURES)
    def test_group_summary_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)  # legacy defaults func="mean", err_func="sem"
        geometry = Aggregate(func="mean", err_func="sem")(DataHolder(data), y="y", levels=_levels(subgroup))

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.y_data[i], g["center"][0])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"][0])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"][0])

    def test_median_summary_matches_legacy(self, one_grouping):
        data, _ = one_grouping
        legacy = self._legacy(data, None, func="median")
        geometry = Aggregate(func="median", err_func="sem")(DataHolder(data), y="y", levels=("grouping_1",))

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            np.testing.assert_allclose(legacy.y_data[i], geometry[_as_group_key(label)]["center"][0])


class TestSummaryUParity:
    """Parity: nested ``Aggregate(unique_id)`` vs the legacy ``_summaryu``."""

    def _legacy(self, data, subgroup, **kwargs):
        plot = (
            CategoricalPlot(data)
            .grouping(group="grouping_1", subgroup=subgroup)
            .summaryu(unique_id="unique_grouping", **kwargs)
            .plot_data(y="y")
        )
        processor = CategoricalProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_per_subject_summaryu_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = _levels(subgroup)
        legacy = self._legacy(data, subgroup)  # agg_func=None: one summary line per subject
        geometry = Aggregate(func="mean", err_func="sem", unique_id="unique_grouping")(
            DataHolder(data), y="y", levels=levels
        )

        # Legacy appends one entry per subject (group loop outer, uids ascending);
        # the transform nests the subject into the key. Fixture rows are
        # contiguous per (group, uid), so both iterate the same order.
        items = list(geometry.items())
        assert len(legacy.y_data) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[: len(levels)] == _as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.y_data[i], g["center"][0])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"][0])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"][0])

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_aggregated_summaryu_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        # Legacy summaryu re-applies func at the second level (it ignores
        # agg_func for the value); parity pins func == agg_func. The transform
        # honors agg_func as a deliberate fix for that legacy quirk.
        legacy = self._legacy(data, subgroup, agg_func="mean")
        geometry = Aggregate(func="mean", agg_func="mean", err_func="sem", unique_id="unique_grouping")(
            DataHolder(data), y="y", levels=_levels(subgroup)
        )

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.y_data[i], g["center"][0])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"][0])


class TestBarParity:
    """Parity: ``Aggregate`` nesting branches vs the legacy ``_bar`` heights."""

    def _legacy(self, data, subgroup, **kwargs):
        plot = CategoricalPlot(data).grouping(group="grouping_1", subgroup=subgroup).bar(**kwargs).plot_data(y="y")
        processor = CategoricalProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize("fixture, subgroup", FIXTURES)
    def test_group_bar_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)  # legacy defaults func="mean"
        geometry = Aggregate(func="mean")(DataHolder(data), y="y", levels=_levels(subgroup))

        assert len(legacy.heights) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            np.testing.assert_allclose(legacy.heights[i], geometry[_as_group_key(label)]["center"][0])
            np.testing.assert_allclose(legacy.bottoms[i], 0)

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_per_subject_bar_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = _levels(subgroup)
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping")  # agg_func=None: one bar per subject
        geometry = Aggregate(func="mean", unique_id="unique_grouping")(DataHolder(data), y="y", levels=levels)

        items = list(geometry.items())
        assert len(legacy.heights) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[: len(levels)] == _as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.heights[i], g["center"][0])
            np.testing.assert_allclose(legacy.bottoms[i], 0)

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_aggregated_bar_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        # Unlike summaryu, the legacy bar honors agg_func at the second level.
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping", agg_func="mean")
        geometry = Aggregate(func="mean", unique_id="unique_grouping", agg_func="mean")(
            DataHolder(data), y="y", levels=_levels(subgroup)
        )

        assert len(legacy.heights) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            np.testing.assert_allclose(legacy.heights[i], geometry[_as_group_key(label)]["center"][0])


class TestAgglineParity:
    """Parity: ``Aggregate`` per-x branches vs the legacy ``_aggline``."""

    def _legacy(self, data, subgroup=None, **kwargs):
        plot = LinePlot(data).grouping(group="grouping_1", subgroup=subgroup).aggline(**kwargs).plot_data(y="y", x="x")
        processor = LineProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    def test_group_aggline_matches_legacy(self):
        data = _repeated_x_data()
        legacy = self._legacy(data)  # legacy defaults func="mean", err_func="sem", agg_func=None
        geometry = Aggregate(func="mean", err_func="sem", how="groupby")(
            DataHolder(data), y="y", x="x", levels=("grouping_1",)
        )

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["center"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"])

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_aggregated_aggline_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = _levels(subgroup)
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping", agg_func="mean")
        shared = dict(func="mean", agg_func="mean", err_func="sem", unique_id="unique_grouping")
        group_by = Aggregate(how="groupby", **shared)(DataHolder(data), y="y", x="x", levels=levels)
        matrix = Aggregate(how="matrix", **shared)(DataHolder(data), y="y", x="x", levels=levels)

        # The fixture grid is aligned, so the groupby and matrix strategies agree;
        # both must match the legacy groupby-based aggregation.
        assert set(group_by) == set(matrix)
        assert len(legacy.y_data) == len(group_by)
        for i, label in enumerate(legacy.group_labels):
            g = group_by[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["center"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            m = matrix[_as_group_key(label)]
            np.testing.assert_allclose(m["center"], g["center"])
            np.testing.assert_allclose(m["error_low"], g["error_low"])


class TestLineParity:
    """Parity: ``Identity`` raw series and the ``Aggregate`` matrix pivot vs the legacy ``_line``."""

    def _legacy(self, data, subgroup, **kwargs):
        plot = LinePlot(data).grouping(group="grouping_1", subgroup=subgroup).line(**kwargs).plot_data(y="y", x="x")
        processor = LineProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize("fixture, subgroup", FIXTURES)
    def test_raw_line_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)  # func=None: raw per-group series
        geometry = Identity()(DataHolder(data), y="y", x="x", levels=_levels(subgroup))

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            assert legacy.error_data[i] is None

    @pytest.mark.parametrize("fixture, subgroup", UID_FIXTURES)
    def test_line_matrix_aggregation_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping", func="mean", err_func="sem")
        geometry = Aggregate(func="mean", err_func="sem", unique_id="unique_grouping", how="matrix")(
            DataHolder(data), y="y", x="x", levels=_levels(subgroup)
        )

        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["center"])
            # legacy error_data holds the symmetric per-x errors
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"])
