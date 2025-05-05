import pytest

from lithos.plotting.processing import LineProcessor
from lithos import LinePlot


class TestLineProcessor:
    @pytest.fixture
    def _fixt(self, test_groupings, request) -> tuple[dict, tuple[int, int, int, int]]:
        data, x = request.getfixturevalue(test_groupings)
        return data, x

    def get_n_groups(self, vals, uid=None, agg_func=None):
        output = 1
        if uid is not None and agg_func is None:
            end = 3
        elif uid is not None and agg_func is None:
            end = 3
        else:
            end = 2
        for i in vals[:end]:
            if i != 0:
                output *= i
        return output

    @pytest.mark.parametrize(
        "data, subgroup, agg_func, uid",
        [
            ("one_grouping", None, None, None),
            ("one_grouping_with_unique_ids", None, "mean", "unique_grouping"),
            ("one_grouping_with_unique_ids", None, None, "unique_grouping"),
            ("two_grouping", "grouping_2", None, None),
            ("two_grouping_with_unique_ids", "grouping_2", "mean", "unique_grouping"),
            ("two_grouping_with_unique_ids", "grouping_2", None, "unique_grouping"),
        ],
    )
    def test_line(self, data, subgroup, agg_func, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = LineProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            LinePlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .line(unique_id=uid, func=agg_func)
            .plot_data(y="y", x="x")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(
            _fixt[1], uid=uid, agg_func=agg_func
        )

    @pytest.mark.parametrize(
        "data, subgroup, agg_func, uid",
        [
            ("one_grouping", None, None, None),
            ("one_grouping_with_unique_ids", None, "mean", "unique_grouping"),
            ("one_grouping_with_unique_ids", None, None, "unique_grouping"),
            ("two_grouping", "grouping_2", None, None),
            ("two_grouping_with_unique_ids", "grouping_2", "mean", "unique_grouping"),
            ("two_grouping_with_unique_ids", "grouping_2", None, "unique_grouping"),
        ],
    )
    def test_kde(self, data, subgroup, agg_func, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = LineProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            LinePlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .kde(unique_id=uid, agg_func=agg_func)
            .plot_data(y="y", x="x")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(
            _fixt[1], uid=uid, agg_func=agg_func
        )

    @pytest.mark.parametrize(
        "data, subgroup, agg_func, uid",
        [
            ("one_grouping", None, None, None),
            ("one_grouping_with_unique_ids", None, "mean", "unique_grouping"),
            ("one_grouping_with_unique_ids", None, None, "unique_grouping"),
            ("two_grouping", "grouping_2", None, None),
            ("two_grouping_with_unique_ids", "grouping_2", "mean", "unique_grouping"),
            ("two_grouping_with_unique_ids", "grouping_2", None, "unique_grouping"),
        ],
    )
    def test_aggline(self, data, subgroup, agg_func, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = LineProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            LinePlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .aggline(unique_id=uid, func=agg_func, agg_func="mean")
            .plot_data(y="y", x="x")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(
            _fixt[1], uid=uid, agg_func="mean"
        )

    @pytest.mark.parametrize(
        "data, subgroup, agg_func, uid",
        [
            ("one_grouping", None, None, None),
            ("one_grouping_with_unique_ids", None, "mean", "unique_grouping"),
            ("one_grouping_with_unique_ids", None, None, "unique_grouping"),
            ("two_grouping", "grouping_2", None, None),
            # ("two_grouping_with_unique_ids", "grouping_2", "mean", "unique_grouping"),
            # ("two_grouping_with_unique_ids", "grouping_2", None, "unique_grouping"),
        ],
    )
    def test_hist(self, data, subgroup, agg_func, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = LineProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            LinePlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .hist(unique_id=uid, agg_func=agg_func)
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].group_labels) == self.get_n_groups(
            _fixt[1], uid=uid, agg_func=agg_func
        )
