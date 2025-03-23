import pytest
import numpy as np

from lithos.plotting.plot_utils import create_dict, radian_ticks


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
