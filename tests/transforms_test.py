import numpy as np
import pytest

from lithos import (
    Aggregate,
    Density,
    Fit,
    Identity,
    Summary,
)
from lithos.plotting.transforms import as_error_pair
from lithos.utils import DataHolder


def _holder(data):
    return DataHolder(data)


def _group_vals(data, key):
    mask = np.asarray(data["grouping_1"]) == key[0]
    return np.asarray(data["y"], dtype=float)[mask]


class TestAsErrorPair:
    def test_none(self):
        assert as_error_pair(None) == (None, None)

    def test_scalar_symmetric(self):
        assert as_error_pair(0.5) == (0.5, 0.5)

    def test_pair(self):
        assert as_error_pair((0.4, 0.8)) == (0.4, 0.8)

    def test_array_size_two(self):
        assert as_error_pair(np.array([0.3, 0.9])) == (0.3, 0.9)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            as_error_pair(np.zeros(3))


class TestIdentity:
    @pytest.mark.parametrize(
        "fixture,levels,expected_groups",
        [
            ("one_grouping", ("grouping_1",), 3),
            ("two_grouping", ("grouping_1",), 2),
            ("two_grouping", ("grouping_1", "grouping_2"), 4),
        ],
    )
    def test_group_count(self, request, fixture, levels, expected_groups):
        data, _ = request.getfixturevalue(fixture)
        geometry = Identity()(_holder(data), y="y", x="x", levels=levels)
        assert len(geometry) == expected_groups

    def test_two_grouping_values(self, two_grouping):
        data, _ = two_grouping
        geometry = Identity()(
            _holder(data), y="y", x="x", levels=("grouping_1", "grouping_2")
        )
        # subgroups are [2, 3] (offset above the group values [0, 1])
        assert set(geometry.keys()) == {(0, 2), (0, 3), (1, 2), (1, 3)}
        for gkey, g in geometry.items():
            assert g["x"].shape == (30,)
            assert g["y"].shape == (30,)
            assert g["n"] == 30

    def test_requires_x_and_y(self, one_grouping):
        data, _ = one_grouping
        with pytest.raises(ValueError):
            Identity()(_holder(data), y="y", x=None)


class TestAggregate:
    def test_mean_sem_per_group(self, one_grouping):
        data, _ = one_grouping
        geometry = Aggregate(func="mean", err_func="sem")(
            _holder(data), y="y", levels=("grouping_1",)
        )
        assert len(geometry) == 3
        for key, g in geometry.items():
            vals = _group_vals(data, key)
            n = vals.size
            assert g["center"] == pytest.approx(float(np.mean(vals)))
            assert g["error_low"] == pytest.approx(
                float(np.std(vals) / np.sqrt(n - 1))
            )
            assert g["error_high"] == g["error_low"]
            assert g["n"] == 30

    def test_no_error(self, one_grouping):
        data, _ = one_grouping
        geometry = Aggregate(func="mean")(
            _holder(data), y="y", levels=("grouping_1",)
        )
        for key, g in geometry.items():
            assert g["error_low"] is None
            assert g["error_high"] is None

    def test_nested_unique_id(self, two_grouping_with_unique_ids):
        data, _ = two_grouping_with_unique_ids
        geometry = Aggregate(
            func="mean", agg_func="mean", err_func="sem", unique_id="unique_grouping"
        )(_holder(data), y="y", levels=("grouping_1", "grouping_2"))
        # fixture is create_synthetic_data(2, 3, 3, 30): 3 subgroups [2, 3, 4]
        assert set(geometry.keys()) == {
            (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)
        }
        for gkey, g in geometry.items():
            assert g["n"] == 3  # three unique ids per group/subgroup
            assert g["error_low"] is not None


class TestDensity:
    def test_kde(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="kde")(
            _holder(data), y="y", levels=("grouping_1",)
        )
        assert len(geometry) == 3
        for gkey, g in geometry.items():
            assert g["x"].shape == g["y"].shape
            assert g["x"].size > 1

    def test_hist(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="hist", bins=10)(
            _holder(data), y="y", levels=("grouping_1",)
        )
        for gkey, g in geometry.items():
            assert g["edges"].size == g["height"].size + 1
            assert g["binwidth"].size == g["height"].size
            assert g["stat"] == "density"
            np.testing.assert_allclose(
                g["centers"], g["edges"][:-1] + g["binwidth"] / 2
            )

    def test_ecdf(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="ecdf")(
            _holder(data), y="y", levels=("grouping_1",)
        )
        for gkey, g in geometry.items():
            assert np.all((g["y"] >= 0) & (g["y"] <= 1))
            assert np.all(np.diff(g["x"]) >= 0)

    def test_invalid_kind_raises(self, one_grouping):
        data, _ = one_grouping
        with pytest.raises(ValueError):
            Density(kind="nope")(_holder(data), y="y", levels=("grouping_1",))


class TestSummary:
    def test_default_quantiles(self, one_grouping):
        data, _ = one_grouping
        geometry = Summary()(_holder(data), y="y", levels=("grouping_1",))
        for gkey, g in geometry.items():
            assert g["whisker_low"] <= g["q1"] <= g["median"] <= g["q3"]
            assert g["q3"] <= g["whisker_high"]
            assert g["notch_low"] is None
            assert g["n"] == 30

    def test_mean_summary(self, one_grouping):
        data, _ = one_grouping
        geometry = Summary(func="mean", err_func="sem", notch=True)(
            _holder(data), y="y", levels=("grouping_1",)
        )
        for key, g in geometry.items():
            vals = _group_vals(data, key)
            assert g["center"] == pytest.approx(float(np.mean(vals)))
            assert g["error_low"] is not None
            assert g["notch_low"] is not None

    def test_minmax_whiskers(self, one_grouping):
        data, _ = one_grouping
        geometry = Summary(whisker="minmax")(
            _holder(data), y="y", levels=("grouping_1",)
        )
        for key, g in geometry.items():
            vals = _group_vals(data, key)
            assert g["whisker_low"] == pytest.approx(float(vals.min()))
            assert g["whisker_high"] == pytest.approx(float(vals.max()))


class TestFit:
    def test_linear_fit(self, two_grouping):
        data, _ = two_grouping
        geometry = Fit(fit_func="linear", ci_func="ci")(
            _holder(data), y="y", x="x", levels=("grouping_1", "grouping_2")
        )
        # subgroups are [2, 3] (offset above the group values [0, 1])
        assert set(geometry.keys()) == {(0, 2), (0, 3), (1, 2), (1, 3)}
        for gkey, g in geometry.items():
            assert g["x"].shape == g["y"].shape
            assert g["ci"] is not None
            assert g["ci"].shape == g["x"].shape

    def test_requires_x(self, one_grouping):
        data, _ = one_grouping
        with pytest.raises(ValueError):
            Fit()(_holder(data), y="y", x=None)


