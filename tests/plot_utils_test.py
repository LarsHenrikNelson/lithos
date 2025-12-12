import pytest
import numpy as np

from lithos.plotting.plot_utils import (
    create_dict,
    radian_ticks,
    _create_groupings,
    _process_positions,
)
from lithos.utils import DataHolder


@pytest.mark.parametrize(
    "color, unique_groups, correct_output",
    [
        ({0: "black", 1: "red"}, [(0,), (1,)], {(0,): "black", (1,): "red"}),
        (3, [(0,), (1,)], {(0,): 3, (1,): 3}),
        (
            2,
            [(1, 2), (0, 2), (0, 3), (1, 3)],
            {(1, 2): 2, (0, 2): 2, (0, 3): 2, (1, 3): 2},
        ),
        (
            "o",
            [(1, 2), (0, 2), (0, 3), (1, 3)],
            {(1, 2): "o", (0, 2): "o", (0, 3): "o", (1, 3): "o"},
        ),
        (3, [("",)], {("",): 3}),
        (
            {0: "black", 1: "red"},
            [(1, 2), (0, 2), (0, 3), (1, 3)],
            {(1, 2): "red", (0, 2): "black", (0, 3): "black", (1, 3): "red"},
        ),
        (
            {2: "black", 3: "red"},
            [(1, 2), (0, 2), (0, 3), (1, 3)],
            {(1, 2): "black", (0, 2): "black", (0, 3): "red", (1, 3): "red"},
        ),
    ],
)
def test_create_dict(color, unique_groups, correct_output):
    output = create_dict(color, unique_groups)
    for key, value in output.items():
        assert value == correct_output[key]


pi_symbol = "\u03c0"


@pytest.mark.parametrize(
    "values, rotate, correct_values",
    [
        (
            [0, 45, 90, 135, 180, 225, 270, 315],
            False,
            [
                "0",
                f"{pi_symbol}/4",
                f"{pi_symbol}/2",
                f"3{pi_symbol}/4",
                f"{pi_symbol}",
                f"5{pi_symbol}/4",
                f"3{pi_symbol}/2",
                f"7{pi_symbol}/4",
            ],
        ),
        (
            [0, 45, 90, 135, 180, 225, 270, 315],
            True,
            [
                "0",
                f"{pi_symbol}/4",
                f"{pi_symbol}/2",
                f"3{pi_symbol}/4",
                f"{pi_symbol}",
                f"-3{pi_symbol}/4",
                f"-{pi_symbol}/2",
                f"-{pi_symbol}/4",
            ],
        ),
    ],
)
def test_radian_ticks(values, rotate, correct_values):
    values = [np.pi * deg / 180 for deg in values]
    output = radian_ticks(values, rotate)
    assert output == correct_values


@pytest.mark.parametrize(
    "group, subgroup, group_order, subgroup_order",
    [
        ("grouping_1", None, None, None),
        ("grouping_1", None, [0, 1], None),
    ],
)
def test_create_groupings_1_group(
    two_grouping_with_unique_ids, group, subgroup, group_order, subgroup_order
):
    data, groups = two_grouping_with_unique_ids
    data = DataHolder(two_grouping_with_unique_ids[0])
    gorder, sorder, unique_groups, levels = _create_groupings(
        data, group, subgroup, group_order, subgroup_order
    )
    assert len(gorder) == groups[0]
    if group_order is not None:
        assert len(gorder) == len(group_order)
        assert gorder == group_order
    assert len(levels) == 1
    assert sorder == subgroup_order
    assert len(unique_groups) == groups[0]


@pytest.mark.parametrize(
    "group, subgroup, group_order, subgroup_order",
    [
        ("grouping_1", "grouping_2", [0, 1], [0, 1, 2]),
        ("grouping_1", "grouping_2", [0, 1], None),
    ],
)
def test_create_groupings_2_groups(
    two_grouping_with_unique_ids, group, subgroup, group_order, subgroup_order
):
    data, groups = two_grouping_with_unique_ids
    data = DataHolder(two_grouping_with_unique_ids[0])
    gorder, sorder, unique_groups, levels = _create_groupings(
        data, group, subgroup, group_order, subgroup_order
    )

    if subgroup_order is not None:
        assert len(sorder) == len(subgroup_order)
        assert sorder == subgroup_order

    assert len(sorder) == groups[1]
    assert len(levels) == 2
    assert len(unique_groups) == groups[0] * groups[1]


@pytest.mark.parametrize(
    "group_spacing, group_order, subgroup_order, output",
    [
        (0.9, [0, 1], None, ({(0,): 0.0, (1,): 1.0}, 1.0)),
        (
            0.9,
            [0, 1],
            [0, 1, 2],
            (
                {
                    (0, 0): -0.30000000000000004,
                    (0, 1): 0.0,
                    (0, 2): 0.30000000000000004,
                    (1, 0): 0.7,
                    (1, 1): 1.0,
                    (1, 2): 1.3,
                },
                0.3,
            ),
        ),
        (
            0.7,
            [0, 1],
            [0, 1, 2],
            (
                {
                    (0, 0): -0.23333333333333334,
                    (0, 1): 0.0,
                    (0, 2): 0.23333333333333334,
                    (1, 0): 0.7666666666666666,
                    (1, 1): 1.0,
                    (1, 2): 1.2333333333333334,
                },
                0.2333333333333333,
            ),
        ),
    ],
)
def test_process_positions(group_spacing, group_order, subgroup_order, output):
    loc_dict, width = _process_positions(group_spacing, group_order, subgroup_order)

    assert width == output[1]
    for key, value in output[0].items():
        assert np.isclose(loc_dict[key], value)
