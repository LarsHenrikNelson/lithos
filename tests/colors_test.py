import pytest

from lithos.plotting.plot_utils import _process_colors, _continuous_cycler


@pytest.mark.parametrize("color", ["black", "red", "blue"])
def test_process_colors_string_input(color):
    output = _process_colors(color)
    assert output == color


@pytest.mark.parametrize(
    "input, group, subgroup, output",
    [
        (
            {0: "black", 1: "red"},
            [0, 1],
            None,
            {0: "black", 1: "red"},
        ),
        (
            {2: "black", 3: "red"},
            [0, 1],
            [2, 3],
            {2: "black", 3: "red"},
        ),
        (None, [0, 1], None, {0: "#1f77b3", 1: "#ff7e0e"}),
        (None, [0, 1], [2, 3], {2: "#1f77b3", 3: "#ff7e0e"}),
    ],
)
def test_process_colors_dict_input(input, group, subgroup, output):
    output = _process_colors(input, group, subgroup)
    assert output == output


@pytest.mark.parametrize(
    "color, group, subgroup, correct_color",
    [
        ("glasbey_category10", [0, 1], None, {0: "#1f77b3", 1: "#ff7e0e"}),
        ("glasbey_category10", [0, 1], [2, 3], {2: "#1f77b3", 3: "#ff7e0e"}),
        (
            "blues-100:255",
            [0, 1],
            [2, 3, 4],
            {2: "#a4c1e5", 3: "#789dc6", 4: "#3a7bb1"},
        ),
        (
            "blues-100:256",
            [0, 1],
            [2, 3, 4],
            {2: "#a4c1e5", 3: "#789dc6", 4: "#3a7bb1"},
        ),
    ],
)
def test_process_colors_colormap_input(color, group, subgroup, correct_color):
    output = _process_colors(color, group, subgroup)
    assert isinstance(output, dict)
    for key, value in output.items():
        assert value == correct_color[key]


@pytest.mark.parametrize(
    "data, arg_cycle, start, stop",
    [
        ({"group": [1, 1, 1, 2, 2, 2, 3, 3, 3]}, "glasbey_category10", 0, 255),
        ({"group": [1, 1, 1, 2, 2, 2, 3, 3, 3]}, "kb", 0, 255),
    ],
)
def test_continuous_cycler_discrete_groups(data, arg_cycle, start, stop):
    color = _continuous_cycler("group", data, arg_cycle, start=start, stop=stop)
    assert len(color) == len(data["group"])
    assert len(set(color)) == len(set(data["group"]))


@pytest.mark.parametrize(
    "n_samples, arg_cycle, start, stop",
    [(255, "kb", 0, 255)],
)
def test_continuous_cycler_continuous_groups(n_samples, arg_cycle, start, stop):
    data = {"group": list(range(n_samples))}
    color = _continuous_cycler("group", data, arg_cycle, start=start, stop=stop)
    assert len(color) == 255
