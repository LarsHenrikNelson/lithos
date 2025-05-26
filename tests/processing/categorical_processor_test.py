import pytest

from lithos.plotting.processing import CategoricalProcessor
from lithos import CategoricalPlot


class TestCategoricalProcessor:
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
        "data, subgroup, agg_func",
        [
            ("one_grouping_with_unique_ids", None, None),
            ("one_grouping_with_unique_ids", None, "mean"),
            ("two_grouping_with_unique_ids", "grouping_2", None),
            ("two_grouping_with_unique_ids", "grouping_2", "mean"),
        ],
    )
    def test_jitteru(self, data, subgroup, agg_func, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )
        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .jitteru(unique_id="unique_grouping", agg_func=agg_func)
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(
            _fixt[1], uid="unique_grouping", agg_func=None
        )

    @pytest.mark.parametrize(
        "data, subgroup, uid",
        [
            ("one_grouping", None, None),
            ("one_grouping_with_unique_ids", None, "unique_grouping"),
            ("two_grouping", "grouping_2", None),
            ("two_grouping_with_unique_ids", "grouping_2", "unique_grouping"),
        ],
    )
    def test_jitter(self, data, subgroup, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .jitter(unique_id=uid)
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(_fixt[1], uid)

    @pytest.mark.parametrize(
        "data, subgroup, uid",
        [
            ("one_grouping", None, None),
            ("one_grouping_with_unique_ids", None, "unique_grouping"),
            ("two_grouping", "grouping_2", None),
            ("two_grouping_with_unique_ids", "grouping_2", "unique_grouping"),
        ],
    )
    def test_violin(self, data, subgroup, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .violin(unique_id=uid)
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(_fixt[1], uid)

    @pytest.mark.parametrize(
        "data, subgroup, agg_func",
        [
            ("one_grouping_with_unique_ids", None, None),
            ("one_grouping_with_unique_ids", None, "mean"),
            ("two_grouping_with_unique_ids", "grouping_2", None),
            ("two_grouping_with_unique_ids", "grouping_2", "mean"),
        ],
    )
    def test_summaryu(self, data, subgroup, agg_func, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .summaryu(unique_id="unique_grouping", agg_func=agg_func)
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(
            _fixt[1], uid="unique_grouping", agg_func=agg_func
        )

    @pytest.mark.parametrize(
        "data, subgroup, uid",
        [
            ("one_grouping", None, None),
            ("two_grouping", "grouping_2", None),
        ],
    )
    def test_summary(self, data, subgroup, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .summary()
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].x_data) == self.get_n_groups(_fixt[1], uid)

    @pytest.mark.parametrize(
        "data, subgroup, uid",
        [
            ("one_grouping", None, None),
            ("two_grouping", "grouping_2", None),
        ],
    )
    def test_box(self, data, subgroup, uid, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .box()
            .plot_data(y="y")
        )
        metadata = plot.metadata()
        output, _ = processor(plot.data, metadata)
        assert len(output[0].x_data) == self.get_n_groups(_fixt[1], uid)

    @pytest.mark.parametrize(
        "data, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_percent(self, data, subgroup, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .percent(unique_id="unique_grouping")
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].heights) == self.get_n_groups(
            _fixt[1], uid="unique_grouping"
        )

    @pytest.mark.parametrize(
        "data, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_bar(self, data, subgroup, request) -> None:
        _fixt = request.getfixturevalue(data)
        processor = CategoricalProcessor(
            markers=("o", "X", "^", "s", "d"),
            hatches=("/", "o", "-", "*", "+"),
        )

        plot = (
            CategoricalPlot(_fixt[0])
            .grouping(group="grouping_1", subgroup=subgroup)
            .bar(unique_id="unique_grouping")
            .plot_data(y="y")
        )
        output, _ = processor(plot.data, plot.metadata())
        assert len(output[0].heights) == self.get_n_groups(
            _fixt[1], uid="unique_grouping"
        )
