"""Tests for lithos.plotting.formatting (formatting extracted from the plotters).

The formatters consume only ``(fig, axes)`` plus plain dicts - never a plot
object or plot data - so most tests style a bare matplotlib axes directly.
That isolation is the acceptance criterion for the formatting/rendering
split: the same machinery can format any matplotlib figure. The only lithos
object used is a spec ``LinePlot()`` as a convenient source of the default
(plain-data) format dicts.
"""

import re

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lithos.plotting.formatting import (
    CategoricalFormatter,
    ContinuousFormatter,
    FigureFormatter,
)
from lithos.plotting.spec import LinePlot


def default_format() -> dict:
    """The default format dict the spec plot classes build."""
    return LinePlot().get_format()


def default_transforms() -> dict:
    """The default scale-transform dict the spec plot classes build."""
    return LinePlot().metadata()["transforms"]


def make_labels(x=None) -> dict:
    """Legacy-shaped plot_labels: label text plus value column names."""
    return {
        "y": "value",
        "x": x,
        "ylabel": "Response",
        "xlabel": "Time",
        "title": "Experiment",
        "figure_title": "",
    }


def make_context(**overrides) -> dict:
    """Continuous-layout context (group order, facet flags)."""
    context = {"group_order": [""], "facet": False, "facet_title": False}
    context.update(overrides)
    return context


def categorical_context(**overrides) -> dict:
    """Categorical-layout context (group/subgroup order, tick slots, labels)."""
    context = {
        "group_order": ["g1", "g2"],
        "subgroup_order": None,
        "group_spacing": 0.8,
        "labels": "style1",
        "ticks": [0.0, 1.0],
        "subticks": [0.0, 0.25, 0.5, 0.75],
        "facet": False,
        "facet_title": False,
    }
    context.update(overrides)
    return context


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestFigureFormatter:
    def test_single_axes_become_a_list(self):
        fig, ax = plt.subplots()
        formatter = FigureFormatter(fig, ax, default_format(), make_labels(), default_transforms(), make_context())
        assert formatter.axes == [ax]

    def test_dicts_are_held_by_reference(self):
        fig, ax = plt.subplots()
        plot_format = default_format()
        formatter = FigureFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        assert formatter.plot_format is plot_format
        assert formatter.plot_dict["group_order"] == [""]

    def test_format_hook_raises(self):
        fig, ax = plt.subplots()
        formatter = FigureFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), make_context())
        with pytest.raises(NotImplementedError):
            formatter.format()

    def test_label_size_axis_fallbacks(self):
        plot_format = default_format()
        # legacy-shaped labels dict: only the shared ticklabel_size exists
        plot_format["labels"].pop("xticklabel_size")
        plot_format["labels"].pop("yticklabel_size")
        plot_format["labels"]["ticklabel_size"] = 11
        fig, ax = plt.subplots()
        formatter = FigureFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        assert formatter.label_size("x") == 11
        assert formatter.label_size("y") == 11
        # a per-axis size always wins over the shared fallback
        plot_format["labels"]["xticklabel_size"] = 7
        assert formatter.label_size("x") == 7
        assert formatter.label_size("y") == 11


class TestContinuousLimits:
    def test_kde_pins_density_axis_zero_when_vertical(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0.2, 0.9])
        plot_format = default_format()
        # the density-floor pin applies only when the user limit is None
        plot_format["axis"]["xlim"] = None
        formatter = ContinuousFormatter(
            fig, [ax], plot_format, make_labels(x=None), default_transforms(), make_context()
        )
        formatter.format(plot_types=["kde"])
        assert plot_format["axis"]["xlim"] == [0, None]
        assert plot_format["axis"]["ylim"] == (None, None)

    def test_hist_pins_density_axis_zero_when_horizontal(self):
        fig, ax = plt.subplots()
        ax.plot([0.1, 0.2, 0.3], [1, 2, 3])
        plot_format = default_format()
        # the density-floor pin applies only when the user limit is None
        plot_format["axis"]["ylim"] = None
        formatter = ContinuousFormatter(
            fig, [ax], plot_format, make_labels(x="value"), default_transforms(), make_context()
        )
        formatter.format(plot_types=["hist"])
        assert plot_format["axis"]["ylim"] == [0, None]
        assert plot_format["axis"]["xlim"] == (None, None)

    def test_explicit_limits_are_not_overwritten(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0.2, 0.9])
        plot_format = default_format()
        plot_format["axis"]["xlim"] = [0.1, 0.9]
        formatter = ContinuousFormatter(
            fig, [ax], plot_format, make_labels(x=None), default_transforms(), make_context()
        )
        formatter.format(plot_types=["kde"])
        assert plot_format["axis"]["xlim"] == [0.1, 0.9]

    def test_default_tuple_limits_do_not_pin(self):
        # spec axis() normalizes None to (None, None); the pin only fires on a
        # literal None limit (documented behavior, preserved verbatim).
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0.2, 0.9])
        formatter = ContinuousFormatter(
            fig, [ax], default_format(), make_labels(x=None), default_transforms(), make_context()
        )
        formatter.format(plot_types=["kde"])
        assert formatter.plot_format["axis"]["xlim"] == (None, None)


class TestContinuousFormatting:
    def test_rectilinear_format_end_to_end(self):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        ax.scatter(rng.random(20), rng.random(20))
        plot_format = default_format()
        plot_format["axis_format"]["linewidth"] = {"top": 0, "right": 0, "bottom": 1.5, "left": 1.5}
        plot_format["figure"]["margins"] = 0.07
        plot_format["labels"]["xticklabel_size"] = 9
        formatter = ContinuousFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        formatter.format(plot_types=[])
        fig.canvas.draw()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["bottom"].get_linewidth() == 1.5
        assert ax.get_ylabel() == "Response"
        assert ax.get_xlabel() == "Time"
        assert ax.get_title() == "Experiment"
        assert ax.get_xmargin() == 0.07
        assert ax.get_xticklabels()[0].get_fontsize() == 9

    def test_reference_lines(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plot_format = default_format()
        plot_format["hline"] = {
            "linetype": "hline",
            "lines": [0.5],
            "linestyle": "--",
            "linealpha": 1,
            "linecolor": "red",
            "linewidth": 1,
            "zorder": 1,
        }
        plot_format["vline"] = {
            "linetype": "vline",
            "lines": [0.3],
            "linestyle": ":",
            "linealpha": 1,
            "linecolor": "blue",
            "linewidth": 1,
            "zorder": 1,
        }
        formatter = ContinuousFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        formatter.format(plot_types=[])
        hlines = [line for line in ax.lines if set(np.atleast_1d(line.get_ydata())) == {0.5}]
        vlines = [line for line in ax.lines if set(np.atleast_1d(line.get_xdata())) == {0.3}]
        assert len(hlines) == 1
        assert hlines[0].get_color() == "red"
        assert len(vlines) == 1
        assert vlines[0].get_color() == "blue"

    def test_facet_titles_use_group_order(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [0, 1])
        context = make_context(group_order=["a", "b"], facet=True, facet_title=True)
        formatter = ContinuousFormatter(fig, [ax1, ax2], default_format(), make_labels(), default_transforms(), context)
        formatter.format(plot_types=[])
        assert ax1.get_title() == "a"
        assert ax2.get_title() == "b"

    def test_figure_title_lands_on_figure(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        labels = make_labels()
        labels["figure_title"] = "Overview"
        formatter = ContinuousFormatter(fig, [ax], default_format(), labels, default_transforms(), make_context())
        formatter.format(plot_types=[])
        assert any(text.get_text() == "Overview" for text in fig.texts)

    def test_grid_linewidths_apply(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plot_format = default_format()
        plot_format["grid"]["ygrid"] = 2.0
        plot_format["grid"]["xgrid"] = 0
        formatter = ContinuousFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        formatter.format(plot_types=[])
        fig.canvas.draw()
        assert any(line.get_linewidth() == 2.0 for line in ax.get_ygridlines())
        assert not any(line.get_linewidth() == 2.0 for line in ax.get_xgridlines())

    def test_polar_radian_tick_labels(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        theta = np.linspace(0, 2 * np.pi, 20)
        ax.plot(theta, np.linspace(0, 1, 20))
        plot_format = default_format()
        plot_format["figure"]["projection"] = "polar"
        plot_format["axis"]["xunits"] = "radian"
        formatter = ContinuousFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        formatter.format(plot_types=[])
        labels = "".join(label.get_text() for label in ax.get_xticklabels())
        assert "\u03c0" in labels

    def test_polar_needs_polar_axes(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plot_format = default_format()
        plot_format["figure"]["projection"] = "polar"
        formatter = ContinuousFormatter(fig, [ax], plot_format, make_labels(), default_transforms(), make_context())
        with pytest.raises(ValueError, match="Polar formatting requires polar axes"):
            formatter.format(plot_types=[])


class TestCategoricalFormatting:
    def test_vertical_layout_end_to_end(self):
        fig, ax = plt.subplots()
        ax.scatter([0, 1], [1, 2])
        plot_format = default_format()
        plot_format["axis_format"]["linewidth"] = {"top": 0, "right": 0, "left": 1.5, "bottom": 1.5}
        formatter = CategoricalFormatter(
            fig, [ax], plot_format, make_labels(), default_transforms(), categorical_context()
        )
        formatter.format(plot_types=["scatter"])
        fig.canvas.draw()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.get_ylabel() == "Response"
        assert ax.get_title() == "Experiment"
        assert [label.get_text() for label in ax.get_xticklabels()] == ["g1", "g2"]

    def test_style1_group_labels_at_slot_ticks(self):
        fig, ax = plt.subplots()
        formatter = CategoricalFormatter(
            fig, [ax], default_format(), make_labels(), default_transforms(), categorical_context()
        )
        formatter.set_categorical_axis(ax)
        assert list(ax.get_xticks()) == [0.0, 1.0]
        assert [label.get_text() for label in ax.get_xticklabels()] == ["g1", "g2"]

    def test_style2_tiled_subgroup_labels(self):
        fig, ax = plt.subplots()
        context = categorical_context(labels="style2", subgroup_order=["s1", "s2"])
        formatter = CategoricalFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), context)
        formatter.set_categorical_axis(ax)
        assert [label.get_text() for label in ax.get_xticklabels()] == ["s1", "s2", "s1", "s2"]

    def test_style3_two_tier_labels(self):
        fig, ax = plt.subplots()
        context = categorical_context(labels="style3", subgroup_order=["s1", "s2"])
        formatter = CategoricalFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), context)
        formatter.set_categorical_axis(ax)
        assert [label.get_text() for label in ax.get_xticklabels()] == ["s1", "s2", "s1", "s2"]
        assert len(ax.child_axes) == 1
        assert [label.get_text() for label in ax.child_axes[0].get_xticklabels()] == ["\ng1", "\ng2"]

    def test_unknown_label_style_raises(self):
        fig, ax = plt.subplots()
        context = categorical_context(labels="style4")
        formatter = CategoricalFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), context)
        with pytest.raises(ValueError, match="Labels must style1"):
            formatter.set_categorical_axis(ax)

    def test_categorical_axis_on_y(self):
        fig, ax = plt.subplots()
        formatter = CategoricalFormatter(
            fig, [ax], default_format(), make_labels(), default_transforms(), categorical_context()
        )
        formatter.set_categorical_axis(ax, axis="y")
        assert list(ax.get_yticks()) == [0.0, 1.0]
        assert [label.get_text() for label in ax.get_yticklabels()] == ["g1", "g2"]

    def test_horizontal_value_axis_uses_xdecimals(self):
        fig, ax = plt.subplots()
        ax.scatter([0.2, 0.8], [0.25, 0.75])
        plot_format = default_format()
        plot_format["axis"]["xdecimals"] = 1
        plot_format["axis"]["ydecimals"] = 3
        formatter = CategoricalFormatter(
            fig, [ax], plot_format, make_labels(x="value"), default_transforms(), categorical_context()
        )
        formatter.format(plot_types=[])
        text_labels = [label.get_text() for label in ax.get_xticklabels() if label.get_text()]
        assert text_labels
        # horizontal layouts put the values on x, so the value ticks must
        # render with xdecimals (1), not the categorical ydecimals (3)
        assert all(re.fullmatch(r"-?\d+\.\d", label) for label in text_labels)

    def test_horizontal_truncation_bounds_left_spine(self):
        # truncation creates the discontinuous axis: the spine carrying the
        # categorical slots stops short of the axis intersection
        fig, ax = plt.subplots()
        plot_format = default_format()
        plot_format["axis_format"]["truncate_yaxis"] = True
        formatter = CategoricalFormatter(
            fig, [ax], plot_format, make_labels(x="value"), default_transforms(), categorical_context()
        )
        formatter.set_categorical_axis(ax, axis="y")
        assert ax.spines["left"].get_bounds() == (0.0, 1.0)
        assert ax.spines["bottom"].get_bounds() is None

    def test_vertical_truncation_bounds_bottom_spine(self):
        # the vertical counterpart truncates the bottom spine, not the left
        fig, ax = plt.subplots()
        plot_format = default_format()
        plot_format["axis_format"]["truncate_xaxis"] = True
        formatter = CategoricalFormatter(
            fig, [ax], plot_format, make_labels(), default_transforms(), categorical_context()
        )
        formatter.set_categorical_axis(ax)
        assert ax.spines["bottom"].get_bounds() == (0.0, 1.0)
        assert ax.spines["left"].get_bounds() is None


class TestLegends:
    def test_legend_patches_from_color_dict(self):
        fig, ax = plt.subplots()
        formatter = FigureFormatter(
            fig, [ax], default_format(), make_labels(), default_transforms(), categorical_context()
        )
        patches = formatter.legend_patches(color_dict={"g1": "red", "g2": "blue"}, alpha=0.5)
        assert [patch.get_label() for patch in patches] == ["g1", "g2"]

    def test_format_turns_context_into_a_legend(self):
        fig, ax = plt.subplots()
        ax.scatter([0, 1], [1, 2])
        context = categorical_context(
            legend_dict=[{"g1": "red", "g2": "blue"}, 1],
            legend_anchor=(1.02, 1),
            legend_loc="upper left",
        )
        formatter = CategoricalFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), context)
        formatter.format(plot_types=[])
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["g1", "g2"]

    def test_plot_legend_standalone_figure(self):
        fig, ax = plt.subplots()
        context = categorical_context(
            legend_dict=[{"g1": "red", "g2": "blue"}, 1],
            legend_anchor=(1.02, 1),
            legend_loc="upper left",
        )
        formatter = CategoricalFormatter(fig, [ax], default_format(), make_labels(), default_transforms(), context)
        legend_fig, legend_ax = formatter.plot_legend()
        legend = legend_ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["g1", "g2"]
