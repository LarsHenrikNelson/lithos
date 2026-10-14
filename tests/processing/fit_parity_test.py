import numpy as np
import pytest

from lithos import Fit, LinePlot
from lithos.plotting.processing import LineProcessor
from lithos.utils import DataHolder


class TestFitParity:
    """Parity: Fit transform geometry vs the legacy ``_fit`` processor output.

    The legacy ``LinePlot.fit()`` and the new ``Fit`` transform share the stats
    layer, so their processed x/y/error values must match for the same data:
    the per-group fit, the per-subject fits (``unique_id``, ``agg_func=None``)
    and the aggregated per-subject fits (``unique_id`` + ``agg_func`` /
    ``err_func`` on the shared ``linspace(min, max, 100)`` grid).
    """

    def _legacy(self, data, subgroup, **fit_kwargs):
        plot = LinePlot(data).grouping(group="grouping_1", subgroup=subgroup).fit(**fit_kwargs).plot_data(y="y", x="x")
        processor = LineProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    def _levels(self, subgroup):
        return ("grouping_1",) if subgroup is None else ("grouping_1", "grouping_2")

    def _as_group_key(self, label):
        return tuple(label) if isinstance(label, tuple) else (label,)

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping", None),
            ("two_grouping", "grouping_2"),
        ],
    )
    def test_group_fit_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)
        geometry = Fit()(DataHolder(data), y="y", x="x", levels=self._levels(subgroup))

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[self._as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            # legacy error_data holds the symmetric ci half-widths
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"])

    def test_polynomial_group_fit_matches_legacy(self, one_grouping):
        data, _ = one_grouping
        legacy = self._legacy(data, None, fit_func="polynomial", fit_args={"degree": 3})
        geometry = Fit(fit_func="polynomial", fit_args={"degree": 3})(
            DataHolder(data), y="y", x="x", levels=("grouping_1",)
        )

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[self._as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            assert g["error_low"] is not None

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_per_subject_fits_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = self._levels(subgroup)
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping", agg_func=None)
        geometry = Fit(unique_id="unique_grouping")(DataHolder(data), y="y", x="x", levels=levels)

        # legacy appends one fit per subject (labeled by the group key); the
        # transform nests the subject into the key. Both order subjects by
        # np.unique(uid) within groups in data order, so entries line up.
        items = list(geometry.items())
        assert len(legacy.x_data) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[: len(levels)] == self._as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            # legacy draws no per-subject band; the transform computes the
            # per-subject interval instead (rendering stays opt-in via elements)
            assert g["error_low"] is not None

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_aggregated_fits_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup, unique_id="unique_grouping")  # agg_func="mean", err_func="sem"
        geometry = Fit(unique_id="unique_grouping", agg_func="mean", err_func="sem")(
            DataHolder(data), y="y", x="x", levels=self._levels(subgroup)
        )

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[self._as_group_key(label)]
            # shared grid, aggregated fit and per-grid-point errors
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"])
