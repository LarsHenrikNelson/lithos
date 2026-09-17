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
        geometry = Identity()(_holder(data), y="y", x="x", levels=("grouping_1", "grouping_2"))
        # subgroups are [2, 3] (offset above the group values [0, 1])
        assert set(geometry.keys()) == {(0, 2), (0, 3), (1, 2), (1, 3)}
        for gkey, g in geometry.items():
            assert g["x"].shape == (30,)
            assert g["y"].shape == (30,)
            assert g["n"] == 30

    def test_requires_at_least_one_column(self, one_grouping):
        data, _ = one_grouping
        with pytest.raises(ValueError):
            Identity()(_holder(data), y=None, x=None)

    def test_y_only_geometry(self, one_grouping):
        data, _ = one_grouping
        geometry = Identity()(_holder(data), y="y", levels=("grouping_1",))
        assert len(geometry) == 3
        for gkey, g in geometry.items():
            assert g["y"].shape == (30,)
            assert g["n"] == 30
            assert "x" not in g

    def test_x_only_geometry(self, one_grouping):
        data, _ = one_grouping
        geometry = Identity()(_holder(data), x="x", levels=("grouping_1",))
        assert len(geometry) == 3
        for gkey, g in geometry.items():
            assert g["x"].shape == (30,)
            assert g["n"] == 30
            assert "y" not in g


# paired data: every unique_id shares the same complete x grid (shuffled rows).
PAIRED_ALIGNED = {
    "grouping_1": [0, 0, 0, 0, 1, 1, 1, 1],
    "x": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    "y": [3.0, 1.0, 4.0, 2.0, 10.0, 40.0, 20.0, 30.0],
    "unique_grouping": ["a", "a", "b", "b", "a", "a", "b", "b"],
}


class TestIdentityPaired:
    def test_per_subject_series_sorted_by_x(self):
        geometry = Identity(unique_id="unique_grouping")(_holder(PAIRED_ALIGNED), y="y", x="x", levels=("grouping_1",))
        # one geometry dict per (group, subject)
        assert set(geometry.keys()) == {(0, "a"), (0, "b"), (1, "a"), (1, "b")}
        np.testing.assert_allclose(geometry[(0, "a")]["x"], [0.0, 1.0])
        np.testing.assert_allclose(geometry[(0, "a")]["y"], [1.0, 3.0])
        np.testing.assert_allclose(geometry[(0, "b")]["y"], [2.0, 4.0])
        np.testing.assert_allclose(geometry[(1, "a")]["y"], [10.0, 40.0])
        np.testing.assert_allclose(geometry[(1, "b")]["y"], [20.0, 30.0])
        assert all(g["n"] == 2 for g in geometry.values())

    def test_fixture_per_subject_series(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        geometry = Identity(unique_id="unique_grouping")(_holder(data), y="y", x="x", levels=("grouping_1",))
        # fixture is create_synthetic_data(2, 0, 3, 30): 2 groups * 3 subjects
        assert len(geometry) == 6
        for gkey, g in geometry.items():
            assert g["n"] == 30
            assert g["x"].shape == (30,)
            assert g["y"].shape == (30,)
            # paired series are ordered by the order column
            assert np.all(np.diff(g["x"]) >= 0)

    def test_y_only_series(self):
        geometry = Identity(unique_id="unique_grouping")(_holder(PAIRED_ALIGNED), y="y", levels=("grouping_1",))
        assert set(geometry.keys()) == {(0, "a"), (0, "b"), (1, "a"), (1, "b")}
        for gkey, g in geometry.items():
            assert "x" not in g
            assert g["n"] == 2
            assert g["y"].shape == (2,)

    def test_duplicate_order_value_raises(self):
        bad = {
            "grouping_1": [0, 0, 0],
            "x": [0.0, 0.0, 1.0],
            "y": [1.0, 2.0, 3.0],
            "unique_grouping": ["a", "a", "a"],
        }
        with pytest.raises(AttributeError):
            Identity(unique_id="unique_grouping")(_holder(bad), y="y", x="x", levels=("grouping_1",))

    def test_ragged_subject_size_raises(self):
        bad = {
            "grouping_1": [0, 0, 0],
            "x": [0.0, 1.0, 0.0],
            "y": [1.0, 2.0, 3.0],
            "unique_grouping": ["a", "a", "b"],
        }
        with pytest.raises(AttributeError):
            Identity(unique_id="unique_grouping")(_holder(bad), y="y", x="x", levels=("grouping_1",))

    def test_missing_order_value_raises(self):
        # subjects have equal sizes but different order sets ('a' has 1.0, 'b' has 2.0)
        bad = {
            "grouping_1": [0, 0, 0, 0],
            "x": [0.0, 1.0, 0.0, 2.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "unique_grouping": ["a", "a", "b", "b"],
        }
        with pytest.raises(ValueError):
            Identity(unique_id="unique_grouping")(_holder(bad), y="y", x="x", levels=("grouping_1",))

    def test_default_is_unchanged(self, one_grouping):
        # unique_id=None keeps the legacy Identity behavior exactly
        data, _ = one_grouping
        geometry = Identity()(_holder(data), y="y", x="x", levels=("grouping_1",))
        assert len(geometry) == 3
        for gkey, g in geometry.items():
            assert g["x"].shape == (30,)
            assert g["y"].shape == (30,)


class TestAggregate:
    def test_mean_sem_per_group(self, one_grouping):
        data, _ = one_grouping
        geometry = Aggregate(func="mean", err_func="sem")(_holder(data), y="y", levels=("grouping_1",))
        assert len(geometry) == 3
        for key, g in geometry.items():
            vals = _group_vals(data, key)
            n = vals.size
            assert g["center"].shape == (1,)
            assert g["center"][0] == pytest.approx(float(np.mean(vals)))
            assert g["error_low"].shape == (1,)
            assert g["error_low"][0] == pytest.approx(float(np.std(vals) / np.sqrt(n - 1)))
            assert g["error_high"][0] == g["error_low"][0]
            assert g["n"].shape == (1,)
            assert g["n"][0] == 30

    def test_no_error(self, one_grouping):
        data, _ = one_grouping
        geometry = Aggregate(func="mean")(_holder(data), y="y", levels=("grouping_1",))
        for key, g in geometry.items():
            assert g["error_low"] is None
            assert g["error_high"] is None

    def test_nested_unique_id(self, two_grouping_with_unique_ids):
        data, _ = two_grouping_with_unique_ids
        geometry = Aggregate(func="mean", agg_func="mean", err_func="sem", unique_id="unique_grouping")(
            _holder(data), y="y", levels=("grouping_1", "grouping_2")
        )
        # fixture is create_synthetic_data(2, 3, 3, 30): 3 subgroups [2, 3, 4]
        assert set(geometry.keys()) == {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)}
        for gkey, g in geometry.items():
            assert g["n"].shape == (1,)  # three unique ids per group/subgroup
            assert g["n"][0] == 3
            assert g["error_low"].shape == (1,)
            assert g["error_low"][0] is not None


# per-x aggregation (legacy aggline/line parity): ragged counts at each x.
RAGGED_PER_X = {
    "grouping_1": [0, 0, 0, 0, 0, 1, 1, 1],
    "x": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    "y": [1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0],
    "unique_grouping": ["a", "a", "b", "a", "b", "a", "b", "a"],
}

# aligned data: every unique_id shares the same complete x grid.
ALIGNED_PER_X = {
    "grouping_1": [0, 0, 0, 0, 1, 1, 1, 1],
    "x": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    "unique_grouping": ["a", "b", "a", "b", "a", "b", "a", "b"],
}


class TestAggregatePerX:
    def test_groupby_ragged(self):
        geometry = Aggregate(func="mean", err_func="sem", how="groupby")(
            _holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        g0 = geometry[(0,)]
        np.testing.assert_allclose(g0["x"], [0.0, 1.0])
        np.testing.assert_allclose(g0["center"], [2.0, 5.0])
        np.testing.assert_allclose(g0["error_low"], [np.std([1.0, 2.0, 3.0]) / np.sqrt(2), np.std([4.0, 6.0]) / 1.0])
        assert g0["n"].tolist() == [3, 2]
        g1 = geometry[(1,)]
        np.testing.assert_allclose(g1["center"], [15.0, 30.0])
        assert g1["n"].tolist() == [2, 1]

    def test_groupby_ragged_no_error(self):
        geometry = Aggregate(func="mean", how="groupby")(_holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",))
        for gkey, g in geometry.items():
            assert g["error_low"] is None
            assert g["error_high"] is None

    def test_groupby_unique_id_two_level(self):
        geometry = Aggregate(func="mean", agg_func="mean", unique_id="unique_grouping", how="groupby")(
            _holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        g0 = geometry[(0,)]
        # x=0: a -> mean(1, 2) = 1.5, b -> 3.0; second-level mean = 2.25
        # x=1: a -> 4.0, b -> 6.0; second-level mean = 5.0
        np.testing.assert_allclose(g0["center"], [2.25, 5.0])
        assert g0["n"].tolist() == [2, 2]
        g1 = geometry[(1,)]
        np.testing.assert_allclose(g1["center"], [15.0, 30.0])
        assert g1["n"].tolist() == [2, 1]

    def test_matrix_aligned(self):
        geometry = Aggregate(func="mean", err_func="sem", unique_id="unique_grouping", how="matrix")(
            _holder(ALIGNED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        g0 = geometry[(0,)]
        np.testing.assert_allclose(g0["x"], [0.0, 1.0])
        np.testing.assert_allclose(g0["center"], [1.5, 3.5])
        np.testing.assert_allclose(g0["error_low"], [0.5, 0.5])
        assert g0["n"].tolist() == [2, 2]
        g1 = geometry[(1,)]
        np.testing.assert_allclose(g1["center"], [15.0, 35.0])
        assert g1["n"].tolist() == [2, 2]

    def test_auto_matches_groupby(self):
        # ragged data -> auto falls back to groupby with identical results
        by_group = Aggregate(func="mean", unique_id="unique_grouping", how="groupby")(
            _holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        by_auto = Aggregate(func="mean", unique_id="unique_grouping", how="auto")(
            _holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        assert set(by_auto.keys()) == set(by_group.keys())
        for key in by_group:
            np.testing.assert_allclose(by_auto[key]["center"], by_group[key]["center"])
            np.testing.assert_allclose(by_auto[key]["x"], by_group[key]["x"])

    def test_auto_matches_matrix(self):
        # aligned data -> auto picks matrix, same numbers as groupby
        by_group = Aggregate(func="mean", err_func="sem", unique_id="unique_grouping", how="groupby")(
            _holder(ALIGNED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        by_auto = Aggregate(func="mean", err_func="sem", unique_id="unique_grouping", how="auto")(
            _holder(ALIGNED_PER_X), y="y", x="x", levels=("grouping_1",)
        )
        for key in by_group:
            np.testing.assert_allclose(by_auto[key]["center"], by_group[key]["center"])
            np.testing.assert_allclose(by_auto[key]["error_low"], by_group[key]["error_low"])
            assert by_auto[key]["n"].tolist() == by_group[key]["n"].tolist()

    def test_matrix_requires_unique_id(self):
        with pytest.raises(ValueError):
            Aggregate(func="mean", how="matrix")(_holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",))

    def test_matrix_ragged_raises(self):
        with pytest.raises(ValueError):
            Aggregate(func="mean", unique_id="unique_grouping", how="matrix")(
                _holder(RAGGED_PER_X), y="y", x="x", levels=("grouping_1",)
            )


class TestDensity:
    def test_kde(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="kde")(_holder(data), y="y", levels=("grouping_1",))
        assert len(geometry) == 3
        for gkey, g in geometry.items():
            assert g["x"].shape == g["y"].shape
            assert g["x"].size > 1

    def test_hist(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="hist", bins=10)(_holder(data), y="y", levels=("grouping_1",))
        for gkey, g in geometry.items():
            assert g["edges"].size == g["height"].size + 1
            assert g["binwidth"].size == g["height"].size
            assert g["stat"] == "density"
            np.testing.assert_allclose(g["centers"], g["edges"][:-1] + g["binwidth"] / 2)

    def test_ecdf(self, one_grouping):
        data, _ = one_grouping
        geometry = Density(kind="ecdf")(_holder(data), y="y", levels=("grouping_1",))
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
        geometry = Summary(func="mean", err_func="sem", notch=True)(_holder(data), y="y", levels=("grouping_1",))
        for key, g in geometry.items():
            vals = _group_vals(data, key)
            assert g["center"] == pytest.approx(float(np.mean(vals)))
            assert g["error_low"] is not None
            assert g["notch_low"] is not None

    def test_minmax_whiskers(self, one_grouping):
        data, _ = one_grouping
        geometry = Summary(whisker="minmax")(_holder(data), y="y", levels=("grouping_1",))
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
