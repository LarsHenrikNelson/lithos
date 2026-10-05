"""Significance bracket tests: add_significance(), resolution, metadata, rendering."""

import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lithos.plotting.elements import Marker
from lithos.plotting.spec import CategoricalPlot, LinePlot
from lithos.plotting.spec.significance import Significance, resolve_significance
from lithos.plotting.transforms import Aggregate, Identity
from lithos.utils import DataHolder


def controlled_data() -> dict:
    """Deterministic data: group tops 2, 4, 6; y-range 5."""
    return {"y": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "grouping_1": [0, 0, 1, 1, 2, 2]}


def resolve(plot, data, y="y", x=None):
    holder = DataHolder(data)
    context = plot._layout_context(holder)
    return resolve_significance(plot.significances, holder, y, x, context, plot._plot_transforms["ytransform"])


class TestAddValidation:
    def test_requires_groups_or_positions(self):
        with pytest.raises(ValueError, match="groups or both x1 and x2"):
            CategoricalPlot().add_significance()

    def test_requires_both_x_bounds(self):
        with pytest.raises(ValueError, match="both x1 and x2"):
            CategoricalPlot().add_significance(x1=1.0)

    def test_rejects_empty_groups(self):
        with pytest.raises(ValueError, match="non-empty"):
            CategoricalPlot().add_significance(groups=[])

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="bracket.*or.*line"):
            CategoricalPlot().add_significance(groups=[0, 1], style="mustache")

    def test_rejects_negative_gap_and_step(self):
        with pytest.raises(ValueError, match="gap and step"):
            CategoricalPlot().add_significance(groups=[0, 1], gap=-0.1)
        with pytest.raises(ValueError, match="gap and step"):
            CategoricalPlot().add_significance(groups=[0, 1], step=-0.1)

    def test_categorical_rejects_groups_and_positions(self):
        with pytest.raises(ValueError, match="not both"):
            CategoricalPlot().add_significance(groups=[0, 1], x1=0.0, x2=1.0)

    def test_continuous_allows_groups_with_window(self):
        plot = LinePlot().add_significance(groups=[0], x1=1.0, x2=2.0)
        assert len(plot.significances) == 1

    def test_returns_self_and_stores_spec(self):
        plot = CategoricalPlot().add_significance(text="**", groups=[0, 1], style="line")
        assert plot.significances == [Significance(text="**", groups=[0, 1], style="line")]

    def test_spec_defaults_and_json_round_trip(self):
        spec = Significance().to_spec()
        assert spec["text"] == "*"
        assert spec["groups"] is None
        assert spec["y"] is None
        assert spec["style"] == "bracket"
        assert spec["gap"] == 0.02
        assert spec["step"] == 0.05
        assert json.loads(json.dumps(spec)) == spec

    def test_public_export(self):
        import lithos

        assert lithos.Significance is Significance


class TestCategoricalSpans:
    def test_two_groups_bracket_the_pair(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1])
        brackets = resolve(plot, controlled_data())
        assert len(brackets) == 1
        assert brackets[0]["x1"] == 0.0
        assert brackets[0]["x2"] == 1.0
        assert brackets[0]["facet"] == 0

    def test_three_groups_span_leftmost_to_rightmost(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[2, 0, 1])
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["x1"] == 0.0
        assert brackets[0]["x2"] == 2.0

    def test_pitch_scales_positions(self):
        plot = CategoricalPlot().grouping(group="grouping_1").spacing(pitch=2.0).add_significance(groups=[0, 1])
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["x1"] == 0.0
        assert brackets[0]["x2"] == 2.0

    def test_absolute_positions(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(x1=-0.5, x2=1.5)
        brackets = resolve(plot, controlled_data())
        assert (brackets[0]["x1"], brackets[0]["x2"]) == (-0.5, 1.5)
        # auto-y covers the groups whose slots fall inside the window (0.0, 1.0)
        assert brackets[0]["y"] == pytest.approx(4.0 + 0.02 * 5.0)

    def test_unknown_group_raises(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=["missing"])
        with pytest.raises(ValueError, match="No slot found.*missing"):
            resolve(plot, controlled_data())


class TestSubgroupSpans:
    def test_subgroup_tuple_keys_resolve_to_slots(self, two_grouping):
        data, _ = two_grouping
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1", subgroup="grouping_2")
            .add_significance(groups=[(0, 2), (0, 3)])
        )
        brackets = resolve(plot, data)
        assert brackets[0]["x1"] == pytest.approx(-0.25)
        assert brackets[0]["x2"] == pytest.approx(0.25)

    def test_group_only_key_spans_the_cluster(self, two_grouping):
        data, _ = two_grouping
        plot = CategoricalPlot().grouping(group="grouping_1", subgroup="grouping_2").add_significance(groups=[0, 1])
        brackets = resolve(plot, data)
        assert brackets[0]["x1"] == pytest.approx(-0.25)
        assert brackets[0]["x2"] == pytest.approx(1.25)

    def test_list_keys_from_json_normalize(self, two_grouping):
        data, _ = two_grouping
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1", subgroup="grouping_2")
            .add_significance(groups=[[0, 2], [0, 3]])
        )
        brackets = resolve(plot, data)
        assert brackets[0]["x1"] == pytest.approx(-0.25)
        assert brackets[0]["x2"] == pytest.approx(0.25)


class TestAutoY:
    def test_auto_y_sits_one_gap_above_span_data(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1])
        brackets = resolve(plot, controlled_data())
        # group tops are 2, 4, 6; the y-range is 5 -> 4 + 0.02 * 5
        assert brackets[0]["y"] == pytest.approx(4.0 + 0.02 * 5.0)
        assert brackets[0]["level"] == 0
        assert brackets[0]["auto"] is True

    def test_auto_y_uses_transformed_values(self):
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .transform(ytransform=np.log10)
            .add_significance(groups=[0, 1])
        )
        brackets = resolve(plot, controlled_data())
        y_all = np.log10(np.asarray(controlled_data()["y"], dtype=float))
        span = float(y_all.max() - y_all.min())
        top = float(np.log10(4.0))  # the tallest y value in groups 0 and 1
        assert brackets[0]["y"] == pytest.approx(top + 0.02 * span)

    def test_explicit_y_is_used_verbatim(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1], y=10.0)
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["y"] == 10.0
        assert brackets[0]["level"] is None
        assert brackets[0]["auto"] is False

    def test_overlapping_brackets_stack(self):
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .add_significance(text="*", groups=[0, 1])
            .add_significance(text="**", groups=[0, 2])
        )
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["level"] == 0
        assert brackets[0]["y"] == pytest.approx(4.0 + 0.02 * 5.0)
        assert brackets[1]["level"] == 1
        assert brackets[1]["y"] == pytest.approx(6.0 + 0.02 * 5.0 + 0.05 * 5.0)

    def test_disjoint_brackets_share_a_level(self):
        data = {
            "y": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "grouping_1": [0, 0, 1, 1, 2, 2, 3, 3],
        }
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .add_significance(groups=[0, 1])
            .add_significance(groups=[2, 3])
        )
        brackets = resolve(plot, data)
        assert [bracket["level"] for bracket in brackets] == [0, 0]
        assert brackets[0]["y"] == pytest.approx(4.0 + 0.02 * 7.0)
        assert brackets[1]["y"] == pytest.approx(8.0 + 0.02 * 7.0)

    def test_bracket_caps_stop_at_data_top(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1])
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["cap_bottoms"] == pytest.approx([4.0, 4.0])

    def test_caps_stop_at_nested_bracket_line(self):
        # outer bracket first, inner explicit bracket above the span data:
        # the inner caps stop at the outer bracket's line, not at the data
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .add_significance(groups=[0, 2])
            .add_significance(groups=[0, 1], y=6.5)
        )
        brackets = resolve(plot, controlled_data())
        outer_y = 6.0 + 0.02 * 5.0
        assert brackets[1]["cap_bottoms"] == pytest.approx([outer_y, outer_y])

    def test_line_style_has_no_caps(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1], style="line")
        brackets = resolve(plot, controlled_data())
        assert brackets[0]["cap_bottoms"] is None

    def test_auto_y_without_y_column_raises(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1])
        with pytest.raises(ValueError, match="y column"):
            resolve(plot, controlled_data(), y=None)

    def test_bracket_style_without_y_column_raises(self):
        plot = CategoricalPlot().grouping(group="grouping_1").add_significance(groups=[0, 1], y=5.0)
        with pytest.raises(ValueError, match="y column"):
            resolve(plot, controlled_data(), y=None)


class TestContinuousSpans:
    def test_x_window_filters_rows(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().add_significance(x1=5.0, x2=10.0)
        brackets = resolve(plot, data, y="y", x="x")
        assert (brackets[0]["x1"], brackets[0]["x2"]) == (5.0, 10.0)
        assert brackets[0]["facet"] == 0
        y_all = np.asarray(data["y"], dtype=float)
        x_all = np.asarray(data["x"], dtype=float)
        window = (x_all >= 5.0) & (x_all <= 10.0)
        span = float(y_all.max() - y_all.min())
        assert brackets[0]["y"] == pytest.approx(float(y_all[window].max()) + 0.02 * span)

    def test_group_key_selects_facet_and_full_span(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").facet(True).add_significance(groups=[1])
        brackets = resolve(plot, data, y="y", x="x")
        assert brackets[0]["facet"] == 1
        assert (brackets[0]["x1"], brackets[0]["x2"]) == (0.0, 29.0)
        mask = np.asarray(data["grouping_1"]) == 1
        y_all = np.asarray(data["y"], dtype=float)
        span = float(y_all.max() - y_all.min())
        assert brackets[0]["y"] == pytest.approx(float(y_all[mask].max()) + 0.02 * span)

    def test_group_key_with_window_filters_rows(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").facet(True)
        plot.add_significance(groups=[1], x1=5.0, x2=10.0)
        brackets = resolve(plot, data, y="y", x="x")
        assert brackets[0]["facet"] == 1
        mask = (np.asarray(data["grouping_1"]) == 1) & (np.asarray(data["x"]) >= 5.0) & (np.asarray(data["x"]) <= 10.0)
        y_all = np.asarray(data["y"], dtype=float)
        span = float(y_all.max() - y_all.min())
        assert brackets[0]["y"] == pytest.approx(float(y_all[mask].max()) + 0.02 * span)

    def test_multiple_group_keys_on_continuous_raise(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").add_significance(groups=[0, 1])
        with pytest.raises(ValueError, match="single group key"):
            resolve(plot, data, y="y", x="x")

    def test_full_span_without_x_column_raises(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").add_significance(groups=[0])
        with pytest.raises(ValueError, match="full span"):
            resolve(plot, data, y="y", x=None)


class TestMetadata:
    def test_significances_in_metadata(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add_significance(text="**", groups=[0, 1], style="line")
        metadata = plot.metadata(y="y", x="x")
        assert len(metadata["significances"]) == 1
        assert metadata["significances"][0]["text"] == "**"
        assert metadata["significances"][0]["groups"] == [0, 1]
        assert metadata["significances"][0]["style"] == "line"
        assert json.loads(json.dumps(metadata))

    def test_roundtrip_rebuilds_brackets(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add_significance(text="*", groups=[0, 1], gap=0.1, step=0.2, linecolor="red")
        rebuilt = LinePlot().load_metadata(plot.metadata())
        assert rebuilt.significances == plot.significances

    def test_roundtrip_via_file(self, one_grouping, tmp_path):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add_significance(text="*", groups=[0, 2])
        plot.save_metadata(tmp_path / "sig_plot")
        rebuilt = CategoricalPlot().load_metadata(tmp_path / "sig_plot")
        assert rebuilt.significances == plot.significances
        rebuilt.plot(y="y", data=data)
        assert rebuilt.plotter is not None


class TestRendering:
    def teardown_method(self):
        plt.close("all")

    def test_bracket_renders_once_not_per_group(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").add(Identity(), Marker())
        plot.add_significance(text="*", groups=[0, 1])
        plot.plot(y="y", data=data)

        ax = plot.plotter.axes[0]
        # the bracket is one Line2D with the mustache path, not one per group
        brackets = [line for line in ax.lines if len(line.get_xdata()) == 4]
        assert len(brackets) == 1
        np.testing.assert_allclose(brackets[0].get_xdata(), [0.0, 0.0, 1.0, 1.0])
        labels = [text for text in ax.texts if text.get_text() == "*"]
        assert len(labels) == 1

    def test_line_style_renders_two_point_line(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").add(Identity(), Marker())
        plot.add_significance(text="*", groups=[0, 1], style="line")
        plot.plot(y="y", data=data)

        ax = plot.plotter.axes[0]
        brackets = [line for line in ax.lines if len(line.get_xdata()) == 2]
        assert len(brackets) == 1
        ydata = brackets[0].get_ydata()
        np.testing.assert_allclose(ydata, [ydata[0], ydata[0]])

    def test_bracket_caps_stop_at_data_top_when_rendered(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").add(Identity(), Marker())
        plot.add_significance(text="*", groups=[0, 1])
        plot.plot(y="y", data=data)

        ax = plot.plotter.axes[0]
        bracket = next(line for line in ax.lines if len(line.get_xdata()) == 4)
        ydata = bracket.get_ydata()
        spec = plot.plotter.significance[0]
        assert spec["y"] == max(ydata[1], ydata[2])
        # the caps descend from the bracket line to the data top, never past it
        assert max(ydata[0], ydata[3]) <= spec["y"]
        assert min(ydata[0], ydata[3]) == spec["cap_bottoms"][0]

    def test_axis_limits_expand_to_include_bracket(self):
        data = {"y": np.array([0.0, 1.0, 8.0, 9.0, 2.0, 3.0]), "grouping_1": [0, 0, 1, 1, 2, 2]}
        plot = CategoricalPlot().grouping(group="grouping_1").add(Aggregate(), Marker())
        # gap=0.5 puts the bracket well above the data + default margins
        plot.add_significance(text="*", groups=[0, 1, 2], gap=0.5)
        plot.plot(y="y", data=data)
        bracket_y = plot.plotter.significance[0]["y"]
        assert bracket_y > 9.0  # above the tallest data
        assert plot.plotter.axes[0].get_ylim()[1] >= bracket_y

    def test_explicit_ylim_still_wins(self):
        data = {"y": np.array([0.0, 1.0, 8.0, 9.0, 2.0, 3.0]), "grouping_1": [0, 0, 1, 1, 2, 2]}
        plot = CategoricalPlot().grouping(group="grouping_1").add(Aggregate(), Marker())
        plot.add_significance(text="*", groups=[0, 1, 2], gap=0.5)
        plot.axis(ylim=[0, 5])
        plot.plot(y="y", data=data)
        assert plot.plotter.axes[0].get_ylim() == (0.0, 5.0)

    def test_line_plot_renders_x_window_bracket(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().add(Aggregate(), Marker())
        plot.add_significance(text="*", x1=5.0, x2=10.0)
        plot.plot(y="y", x="x", data=data)

        ax = plot.plotter.axes[0]
        # the mustache bracket path over the x-window
        brackets = [line for line in ax.lines if len(line.get_xdata()) == 4]
        assert len(brackets) == 1
        np.testing.assert_allclose(brackets[0].get_xdata(), [5.0, 5.0, 10.0, 10.0])
        labels = [text for text in ax.texts if text.get_text() == "*"]
        assert len(labels) == 1
