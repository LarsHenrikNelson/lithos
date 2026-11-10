"""Parity: ``Summary`` box geometry vs the stats the legacy ``_box`` renders.

The legacy processor passes raw group values straight to matplotlib's
``boxplot``, which computes medians/quartiles/means there - so parity
compares the ``Summary`` transform against ``boxplot_stats`` on the same
values: the exact numbers legacy draws.

Whiskers and fliers are the documented, deliberate divergence: matplotlib
draws 1.5*IQR whiskers with flier points, while ``Summary`` whiskers are
quantiles (default 5/95) or min/max, and the box pipeline holds no flier
geometry - raw points come from a separate ``Identity`` layer. The notch
bounds use the McGill 1.58 constant (matplotlib uses 1.57).
"""

import numpy as np
import pytest
from matplotlib.cbook import boxplot_stats

from lithos import CategoricalPlot, Summary
from lithos.plotting.processing import CategoricalProcessor
from lithos.utils import DataHolder

_MARKERS = ("o", "X", "^", "s", "d")
_HATCHES = ("/", "o", "-", "*", "+")

FIXTURES = [
    ("one_grouping", None),
    ("two_grouping", "grouping_2"),
]


def _levels(subgroup):
    return ("grouping_1",) if subgroup is None else ("grouping_1", "grouping_2")


def _as_group_key(label):
    return tuple(label) if isinstance(label, tuple) else (label,)


class TestBoxParity:
    """Parity: ``Summary`` box statistics vs the numbers the legacy ``_box`` renders."""

    def _legacy(self, data, subgroup, **kwargs):
        plot = CategoricalPlot(data).grouping(group="grouping_1", subgroup=subgroup).box(**kwargs).plot_data(y="y")
        processor = CategoricalProcessor(markers=_MARKERS, hatches=_HATCHES)
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize("fixture, subgroup", FIXTURES)
    def test_box_stats_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)
        geometry = Summary()(DataHolder(data), y="y", levels=_levels(subgroup))

        # legacy holds the raw group values; matplotlib's boxplot_stats are
        # exactly the statistics it draws
        assert len(legacy.y_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            stats = boxplot_stats(np.asarray(legacy.y_data[i], dtype=float))[0]
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(stats["med"], g["median"])
            np.testing.assert_allclose(stats["q1"], g["q1"])
            np.testing.assert_allclose(stats["q3"], g["q3"])
            np.testing.assert_allclose(stats["mean"], g["mean"])

    @pytest.mark.parametrize("fixture, subgroup", FIXTURES)
    def test_minmax_whiskers_match_legacy_range(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup)
        geometry = Summary(whisker="minmax")(DataHolder(data), y="y", levels=_levels(subgroup))

        # matplotlib clamps its 1.5*IQR whiskers inside the data range; the
        # minmax whiskers cover the full range by definition
        for i, label in enumerate(legacy.group_labels):
            vals = np.asarray(legacy.y_data[i], dtype=float)
            stats = boxplot_stats(vals)[0]
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(g["whisker_low"], vals.min())
            np.testing.assert_allclose(g["whisker_high"], vals.max())
            assert stats["whislo"] >= g["whisker_low"]
            assert stats["whishi"] <= g["whisker_high"]

    def test_quantile_whiskers_and_notch(self, one_grouping):
        data, _ = one_grouping
        geometry = Summary(notch=True)(DataHolder(data), y="y", levels=("grouping_1",))

        for key, g in geometry.items():
            vals = np.asarray(
                [v for g1, v in zip(data["grouping_1"], data["y"], strict=True) if g1 == key[0]], dtype=float
            )
            # 5/95 quantile whiskers by default; notch = median +/- 1.58 IQR/sqrt(n)
            np.testing.assert_allclose(g["whisker_low"], np.percentile(vals, 5))
            np.testing.assert_allclose(g["whisker_high"], np.percentile(vals, 95))
            notch = 1.58 * (np.percentile(vals, 75) - np.percentile(vals, 25)) / np.sqrt(vals.size)
            np.testing.assert_allclose(g["notch_low"], g["median"] - notch)
            np.testing.assert_allclose(g["notch_high"], g["median"] + notch)
