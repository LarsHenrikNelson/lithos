import pytest

from lithos.plotting.processing import (
    BaseProcessor,
)


class TestBaseProcessorClass:
    # def get_column_names(self, x):
    #     cols = sum(True for i in x[:3] if i > 0)
    #     if cols == 1:
    #         return "grouping_1", None
    #     elif cols == 2:
    #         return "grouping_1", "grouping_2"

    @pytest.mark.parametrize(
        "data, ",
        [{(0,): 0, (1,): 1}, {(1, 2): 0, (0, 2): 0, (0, 3): 1, (1, 3): 1}],
    )
    def test_zorder(self, data):
        processor = BaseProcessor(
            markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+")
        )
        processor._plot_dict["zorder_dict"] = data
        old = data
        for i in range(10):
            new = processor._set_zorder()
            for key, value in new.items():
                assert value > old[key]
            old = new

    @pytest.mark.parametrize(
        "args, group_order, subgroup_order, unique_groups, correct_output",
        [
            (
                {"marker": {0: "o", 1: "-"}, "color": "glasbey_category10"},
                [0, 1],
                None,
                [(0,), (1,)],
                {
                    "marker": {(0,): "o", (1,): "-"},
                    "color": {(0,): "#1f77b3", (1,): "#ff7e0e"},
                },
            ),
            (
                {"linestyle": "-"},
                [0, 1],
                None,
                [(0,), (1,)],
                {"linestyle": {(0,): "-", (1,): "-"}},
            ),
            (
                {"marker": "o"},
                [0, 1],
                [2, 3],
                [(1, 2), (0, 2), (0, 3), (1, 3)],
                {"marker": {(1, 2): "o", (0, 2): "o", (0, 3): "o", (1, 3): "o"}},
            ),
            ({"hatch": True}, [""], None, [("",)], {"hatch": {("",): True}}),
            (
                {"marker": {0: "black", 1: "red"}},
                [0, 1],
                [2, 3],
                [(1, 2), (0, 2), (0, 3), (1, 3)],
                {
                    "marker": {
                        (1, 2): "red",
                        (0, 2): "black",
                        (0, 3): "black",
                        (1, 3): "red",
                    }
                },
            ),
            (
                {"marker": {2: "black", 3: "red"}, "color": "blues-100:255"},
                [0, 1],
                [2, 3],
                [(1, 2), (0, 2), (0, 3), (1, 3)],
                {
                    "marker": {
                        (1, 2): "black",
                        (0, 2): "black",
                        (0, 3): "red",
                        (1, 3): "red",
                    },
                    "color": {
                        (1, 2): "#a4c1e5",
                        (0, 2): "#a4c1e5",
                        (0, 3): "#3a7bb1",
                        (1, 3): "#3a7bb1",
                    },
                },
            ),
        ],
    )
    def test_preprocess_args(
        self, args, group_order, subgroup_order, unique_groups, correct_output
    ):
        processor = BaseProcessor(
            markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+")
        )
        processor._plot_dict["unique_groups"] = unique_groups
        processor._plot_dict["group_order"] = group_order
        processor._plot_dict["subgroup_order"] = subgroup_order
        output = processor.preprocess_args(args)
        for key1, value in output.items():
            for key2, value in value.items():
                assert value == correct_output[key1][key2]
