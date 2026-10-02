"""Tests for the spec-based plot API (lithos.plotting.spec).

A ``Plot`` object is a pure metadata holder: no data and no columns exist on
it until ``.plot(y, x, data)`` (or numpy arrays for ``y``/``x``) is called.
"""

import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lithos.plotting.elements import Bar, ErrorBand, ErrorBar, Fill, Line, Marker, SummaryLine
from lithos.plotting.spec import CategoricalPlot, LinePlot
from lithos.plotting.spec.metadata import (
    build_element,
    build_transform,
    layer_to_json,
    to_jsonable,
)
from lithos.plotting.spec.plotter import SpecPlotter
from lithos.plotting.spec.resolver import locate_key, resolve_layers
from lithos.plotting.transforms import Aggregate, Identity
from lithos.utils import DataHolder


class TestAddValidation:
    def test_rejects_non_transform(self):
        plot = LinePlot()
        with pytest.raises(TypeError):
            plot.add("mean", Line())

    def test_rejects_non_element(self):
        plot = LinePlot()
        with pytest.raises(TypeError):
            plot.add(Identity(), "marker")

    def test_requires_at_least_one_element(self):
        plot = LinePlot()
        with pytest.raises(ValueError):
            plot.add(Identity())

    def test_rejects_unsupported_position(self):
        plot = CategoricalPlot()
        with pytest.raises(ValueError):
            plot.add(Identity(), Marker(), position="stack")

    def test_rejects_jitter_position(self):
        # "jitter" is no longer a position mode: width in .add() controls the spread
        plot = CategoricalPlot()
        with pytest.raises(ValueError):
            plot.add(Identity(), Marker(), position="jitter")

    def test_rejects_negative_width(self):
        plot = CategoricalPlot()
        with pytest.raises(ValueError):
            plot.add(Identity(), Marker(), width=-0.5)


class TestAddLayers:
    def test_layers_hold_raw_objects(self):
        plot = LinePlot().grouping(group="grouping_1")
        identity = Identity()
        plot.add(identity, Marker())

        assert len(plot.layers) == 1
        layer = plot.layers[0]
        assert layer["transform"] is identity
        assert layer["transform"].name == "identity"
        assert all(isinstance(element, Marker) for element in layer["elements"])
        # no data is processed at .add() time
        assert "geometry" not in layer

    def test_geometry_honors_grouping_set_after_add(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot()
        plot.add(Identity(), Marker())
        plot.grouping(group="grouping_1")

        processed = plot._process_data(DataHolder(data), y="y", x="x")
        assert len(processed) == 1
        layer = processed[0]
        assert layer["transform"]["name"] == "identity"
        assert [spec["type"] for spec in layer["elements"]] == ["marker"]
        assert len(layer["geometry"]) == 3
        for geometry in layer["geometry"].values():
            assert geometry["n"] == 30
            assert geometry["x"].shape == (30,)
            assert geometry["y"].shape == (30,)

    def test_process_data_without_grouping(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot()
        plot.add(Identity(), Marker())

        geometry = plot._process_data(DataHolder(data), y="y", x="x")[0]["geometry"]
        assert len(geometry) == 1
        single = geometry[("",)]
        assert single["n"] == 90
        assert single["x"].shape == (90,)
        assert single["y"].shape == (90,)

    def test_chained_layers_are_independent(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker())
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())

        assert len(plot.layers) == 2
        assert plot.layers[0]["transform"].name == "identity"
        assert plot.layers[1]["transform"].name == "aggregate"
        processed = plot._process_data(DataHolder(data), y="y", x="x")
        assert processed[0]["geometry"] is not processed[1]["geometry"]

    def test_transform_set_after_add_is_honored(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())
        plot.transform(ytransform=np.square)

        layer = plot._process_data(DataHolder(data), y="y", x="x")[0]
        assert layer["ytransform"] is np.square

    def test_processed_layers_carry_plot_columns(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())

        layer = plot._process_data(DataHolder(data), y="y", x="x")[0]
        assert layer["y"] == "y"
        assert layer["x"] == "x"

    def test_position_defaults_per_layout(self):
        line_layer = LinePlot().add(Identity(), Marker()).layers[0]
        cat_layer = CategoricalPlot().add(Identity(), Marker()).layers[0]
        assert line_layer["position"] == "passthrough"
        assert cat_layer["position"] == "dodge"

    def test_layer_holds_width_and_jitter_kwargs(self):
        plot = CategoricalPlot()
        plot.add(Identity(), Marker(), width=0.5, jitter_type="dist", seed=30)
        layer = plot.layers[0]
        assert layer["width"] == 0.5
        assert layer["jitter_type"] == "dist"
        assert layer["seed"] == 30


class TestPlotDataResolution:
    def test_requires_y_or_x(self):
        plot = LinePlot().add(Identity(), Marker())
        with pytest.raises(ValueError):
            plot.plot(data={"y": np.arange(10)})

    def test_column_names_require_data(self):
        plot = LinePlot().add(Identity(), Marker())
        with pytest.raises(ValueError):
            plot.plot(y="y")

    def test_arrays_with_data_are_rejected(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().add(Identity(), Marker())
        with pytest.raises(ValueError):
            plot.plot(y=np.arange(10), data=data)

    def test_array_columns_resolve_to_names(self):
        plot = LinePlot().add(Identity(), Marker())
        y_name, x_name, holder = plot._resolve_plot_data(np.arange(10.0), np.arange(10.0), None)

        assert y_name == "y"
        assert x_name == "x"
        assert holder.shape == (10, 2)


class TestLayoutContext:
    def test_line_context_facet(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").facet(facet=True)
        context = plot._layout_context(DataHolder(data))
        assert context["layout"] == "continuous"
        assert context["facet"] is True
        assert sorted(context["loc_dict"].values()) == [0, 1, 2]

    def test_categorical_context_positions(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").spacing(pitch=1.2)
        context = plot._layout_context(DataHolder(data))
        assert context["layout"] == "categorical"
        # cluster width is fixed at 1 unit regardless of pitch
        assert context["width"] == 1.0
        assert context["ticks"] == [0, 1.2, 2.4]
        assert sorted(context["loc_dict"].values()) == [0.0, 1.2, 2.4]
        assert context["group_spacing"] == 1.0

    def test_categorical_default_pitch(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        context = plot._layout_context(DataHolder(data))
        assert context["pitch"] == 1.0
        assert context["ticks"] == [0, 1, 2]
        assert sorted(context["loc_dict"].values()) == [0.0, 1.0, 2.0]

    def test_categorical_subgroup_context(self, two_grouping):
        data, _ = two_grouping
        plot = CategoricalPlot().grouping(group="grouping_1", subgroup="grouping_2")
        context = plot._layout_context(DataHolder(data))
        # the 1-unit cluster splits evenly between the subgroups
        assert context["width"] == 0.5
        assert len(context["subticks"]) == len(context["unique_groups"])
        assert sorted(context["loc_dict"].values()) == [-0.25, 0.25, 0.75, 1.25]


class TestResolver:
    def test_dodge_positions_match_group_locations(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Aggregate(), Marker())
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            assert geometry["position"] == context["loc_dict"][gkey]
            np.testing.assert_allclose(geometry["x"], [context["loc_dict"][gkey]])
            assert geometry["center"].size == 1
            assert geometry["extent"] == context["width"]

    def test_zero_width_places_points_at_slot_center(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker())  # default width=0
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            loc = context["loc_dict"][gkey]
            np.testing.assert_allclose(geometry["x"], [loc] * geometry["y"].size)

    def test_width_jitters_within_slot_fraction(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker(), width=0.5, seed=42)
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            loc = context["loc_dict"][gkey]
            # width is a fraction of the 1-unit slot
            assert np.all(np.abs(geometry["x"] - loc) <= 0.25)
            assert geometry["x"].shape == geometry["y"].shape
            assert len(set(np.round(geometry["x"], 6))) > 1

    def test_unique_id_keys_get_even_columns(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(unique_id="unique_grouping"), Marker(), width=0.9)
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        # one geometry per (group, uid); 3 uids per group on even columns
        for gkey, geometry in resolved["geometry"].items():
            # all points of one subject share its column
            assert len(set(np.round(geometry["x"], 6))) == 1
            assert geometry["extent"] == pytest.approx(0.9 / 3)
            assert geometry["uid"] == gkey[-1]
        for gkey in (0, 1):
            uids = sorted(k[-1] for k in resolved["geometry"] if k[0] == gkey)
            columns = [resolved["geometry"][(gkey, uid)]["x"][0] for uid in uids]
            np.testing.assert_allclose(columns, [gkey - 0.3, float(gkey), gkey + 0.3], atol=1e-12)
            # uid_index ranks the subject within its slot (marker cycling)
            indexes = [resolved["geometry"][(gkey, uid)]["uid_index"] for uid in uids]
            assert indexes == [0, 1, 2]

    def test_unique_id_keys_collapse_without_width(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(unique_id="unique_grouping"), Marker())  # width=0
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            assert geometry["x"][0] == context["loc_dict"][gkey[:1]]

    def test_paired_geometry_resolves_by_group_prefix(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(unique_id="unique_grouping"), Marker())
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        resolved = resolve_layers(plot._process_data(holder, y="y"), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            assert len(gkey) == 2
            assert geometry["position"] == context["loc_dict"][gkey[:1]]

    def test_unique_id_colliding_with_grouping_is_rejected(self, two_grouping):
        data, _ = two_grouping
        plot = CategoricalPlot().grouping(group="grouping_1", subgroup="grouping_2")
        plot.add(Identity(unique_id="grouping_2"), Marker())
        with pytest.raises(ValueError):
            plot._process_data(DataHolder(data), y="y")

    def test_locate_key_requires_matching_prefix(self):
        with pytest.raises(KeyError):
            locate_key(("a", "b"), {("z",): 0.0})


class TestMarkerSymbol:
    def test_list_cycles_over_uid_index(self):
        spec = {"marker": ["o", "s", "^"]}
        assert SpecPlotter._marker_symbol(spec, (0, 0), {"uid_index": 0}) == "o"
        assert SpecPlotter._marker_symbol(spec, (0, 1), {"uid_index": 1}) == "s"
        assert SpecPlotter._marker_symbol(spec, (0, 2), {"uid_index": 2}) == "^"
        # cycles past the end of the list
        assert SpecPlotter._marker_symbol(spec, (0, 3), {"uid_index": 3}) == "o"

    def test_list_without_uid_uses_first_entry(self):
        spec = {"marker": ["o", "s"]}
        assert SpecPlotter._marker_symbol(spec, (0,), {}) == "o"

    def test_dict_uses_uid_for_nested_layers(self):
        spec = {"marker": {"a": "o", "b": "s"}}
        assert SpecPlotter._marker_symbol(spec, (0, "a"), {"uid": "a"}) == "o"
        assert SpecPlotter._marker_symbol(spec, (0, "b"), {"uid": "b"}) == "s"

    def test_dict_uses_group_key_for_flat_layers(self):
        spec = {"marker": {(0,): "o", (1,): "s"}}
        assert SpecPlotter._marker_symbol(spec, (0,), {}) == "o"
        assert SpecPlotter._marker_symbol(spec, (1,), {}) == "s"

    def test_dict_missing_entry_raises(self):
        spec = {"marker": {"a": "o"}}
        with pytest.raises(ValueError, match="has no entry"):
            SpecPlotter._marker_symbol(spec, (0,), {})


class TestSummaryLine:
    def test_spec_fields(self):
        spec = SummaryLine(width=0.8, linewidth=3, linecolor="black").to_spec()

        assert spec["type"] == "summaryline"
        assert spec["width"] == 0.8
        assert spec["linewidth"] == 3
        assert spec["linecolor"] == "black"

    def test_summaryline_builds_from_metadata(self):
        spec = SummaryLine(width=0.8, linewidth=3).to_spec()
        assert build_element(spec) == SummaryLine(width=0.8, linewidth=3)

    def test_summaryline_renders_from_aggregate_center(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").add(Aggregate(), SummaryLine())
        plot.plot(y="y", data=data)

        ax = plot.plotter.axes[0]
        assert ax.has_data()
        # the line renders as horizontal error segments spanning width * slot
        segments = [segment for collection in ax.collections for segment in collection.get_segments()]
        assert len(segments) == 3  # one per group
        for segment in segments:
            np.testing.assert_allclose(segment[1] - segment[0], [0.9, 0.0], atol=1e-12)


class TestLabels:
    def teardown_method(self):
        plt.close("all")

    def test_default_labels_use_column_names(self):
        plot = LinePlot().grouping(group="grouping_1")

        labels = plot._resolved_labels(y="y", x="x")
        assert labels["ylabel"] == "y"
        assert labels["xlabel"] == "x"

    def test_default_labels_without_columns_are_blank(self):
        plot = CategoricalPlot().grouping(group="grouping_1")

        labels = plot._resolved_labels()
        assert labels["ylabel"] == ""
        assert labels["xlabel"] == ""
        assert labels["title"] == ""
        assert labels["figure_title"] == ""

    def test_none_means_no_label_and_empty_string_is_kept(self):
        plot = LinePlot().labels(ylabel=None, xlabel="")

        labels = plot._resolved_labels(y="y", x="x")
        assert labels["ylabel"] is None
        assert labels["xlabel"] == ""

    def test_metadata_carries_columns_and_labels(self):
        plot = LinePlot().grouping(group="grouping_1")
        plot.labels(ylabel="value", title="Test")

        metadata = plot.metadata(y="y", x="x")
        assert metadata["data"] == {"y": "y", "x": "x"}
        assert metadata["labels"] == {"ylabel": "value", "xlabel": "x", "title": "Test", "figure_title": ""}

    def test_labels_roundtrip_through_metadata(self):
        plot = LinePlot().grouping(group="grouping_1")
        plot.labels(ylabel=None, xlabel="", title="Test", figure_title="")

        rebuilt = LinePlot().load_metadata(plot.metadata())
        assert rebuilt._labels == plot._labels
        assert rebuilt._resolved_labels(y="y", x="x") == plot._resolved_labels(y="y", x="x")

    def test_render_default_label_is_column_name(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").add(Aggregate(err_func="sem"), Line(), Marker(), ErrorBand())
        plot.plot(y="y", x="x", data=data)

        assert plot.plotter.axes[0].get_ylabel() == "y"
        assert plot.plotter.axes[0].get_xlabel() == "x"

    def test_render_none_label_is_blank(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot()
            .grouping(group="grouping_1")
            .labels(ylabel=None)
            .add(Aggregate(err_func="sem"), Line(), Marker(), ErrorBand())
        )
        plot.plot(y="y", x="x", data=data)

        assert plot.plotter.axes[0].get_ylabel() == ""


class TestLabelFormat:
    def teardown_method(self):
        plt.close("all")

    def test_separate_ticklabel_sizes(self):
        plot = LinePlot().label_format(xticklabel_size=8, yticklabel_size=14)

        labels = plot.plot_format["labels"]
        assert labels["xticklabel_size"] == 8
        assert labels["yticklabel_size"] == 14
        assert "ticklabel_size" not in labels

    def test_render_separate_ticklabel_sizes(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot()
            .grouping(group="grouping_1")
            .label_format(xticklabel_size=8, yticklabel_size=14)
            .add(Aggregate(err_func="sem"), Line(), Marker(), ErrorBand())
        )
        plot.plot(y="y", x="x", data=data)

        ax = plot.plotter.axes[0]
        assert ax.get_xticklabels()[0].get_fontsize() == 8
        assert ax.get_yticklabels()[0].get_fontsize() == 14


class TestMetadata:
    def test_metadata_is_json_serializable(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())
        metadata = plot.metadata(y="y", x="x")

        assert metadata["version"] == 3
        assert metadata["layout"] == "continuous"
        assert metadata["grouping"]["group"] == "grouping_1"
        assert metadata["data"] == {"y": "y", "x": "x"}
        assert len(metadata["layers"]) == 1
        assert metadata["layers"][0]["transform"]["name"] == "aggregate"
        assert metadata["layers"][0]["elements"][0]["type"] == "line"
        # layers are pure specs — no geometry or columns are stored
        assert "groups" not in metadata["layers"][0]
        assert "geometry" not in metadata["layers"][0]
        assert "y" not in metadata["layers"][0]
        assert "x" not in metadata["layers"][0]
        # the whole document round-trips through json
        assert json.loads(json.dumps(metadata)) == json.loads(json.dumps(plot.metadata(y="y", x="x")))

    def test_metadata_template_is_data_free(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1").add(Aggregate(), Marker())
        metadata = plot.metadata()

        assert metadata["data"] == {"y": None, "x": None}
        assert json.loads(json.dumps(metadata))

    def test_metadata_roundtrip_via_dict(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())

        rebuilt = LinePlot().load_metadata(plot.metadata())
        assert len(rebuilt.layers) == 1
        assert rebuilt._grouping == plot._grouping
        rebuilt_layer, layer = rebuilt.layers[0], plot.layers[0]
        assert rebuilt_layer["transform"] == layer["transform"]
        assert rebuilt_layer["elements"] == layer["elements"]
        holder = DataHolder(data)
        rebuilt_geometry = rebuilt._process_data(holder, y="y", x="x")[0]["geometry"]
        for gkey, geometry in plot._process_data(holder, y="y", x="x")[0]["geometry"].items():
            for key in ("x", "center", "error_low", "error_high"):
                np.testing.assert_allclose(np.asarray(rebuilt_geometry[gkey][key]), np.asarray(geometry[key]))

    def test_metadata_roundtrip_via_file(self, one_grouping, tmp_path):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker(), width=0.5, jitter_type="dist", seed=30)
        plot.add(Aggregate(err_func="sem"), SummaryLine(), ErrorBar())
        plot.save_metadata(tmp_path / "spec_plot")
        assert (tmp_path / "spec_plot.json").exists()

        rebuilt = CategoricalPlot().load_metadata(tmp_path / "spec_plot")
        assert len(rebuilt.layers) == 2
        assert rebuilt._layout_options == plot._layout_options
        assert rebuilt.layers[1]["position"] == "dodge"
        # layer spread settings round-trip through the metadata
        assert rebuilt.layers[0]["width"] == 0.5
        assert rebuilt.layers[0]["jitter_type"] == "dist"
        assert rebuilt.layers[0]["seed"] == 30

    def test_template_shares_across_data(self, two_grouping, tmp_path):
        """GraphPad-style sharing: a saved template renders against any dataset."""
        other_data, _ = two_grouping
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar())
        plot.save_metadata(tmp_path / "template")

        rebuilt = LinePlot().load_metadata(tmp_path / "template")
        rebuilt.plot(y="y", x="x", data=other_data)

        assert rebuilt.plotter is not None
        assert rebuilt.plotter.axes[0].has_data()

    def test_load_rejects_wrong_layout(self):
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker())
        with pytest.raises(ValueError):
            CategoricalPlot().load_metadata(plot.metadata())

    def test_load_rejects_wrong_version(self):
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker())
        stale = plot.metadata()
        stale["version"] = 2
        with pytest.raises(ValueError):
            LinePlot().load_metadata(stale)

    def test_build_helpers_reject_unknowns(self):
        with pytest.raises(ValueError):
            build_transform({"name": "unknown"})
        with pytest.raises(ValueError):
            build_element({"type": "unknown"})

    def test_to_jsonable_converts_numpy(self):
        output = to_jsonable({"a": np.arange(3), "b": np.float64(1.5), "c": (np.array([1.0]),)})
        assert output["a"] == [0, 1, 2]
        assert output["b"] == 1.5
        assert output["c"] == [[1.0]]

    def test_layer_to_json_serializes_held_objects(self):
        plot = LinePlot().grouping(group="grouping_1")
        plot.add(Identity(), Marker())
        serialized = layer_to_json(plot.layers[0])

        assert serialized["transform"]["name"] == "identity"
        assert serialized["elements"][0]["type"] == "marker"
        # pure layer spec — no geometry or columns are stored
        assert "geometry" not in serialized
        assert "groups" not in serialized
        assert "y" not in serialized
        assert "x" not in serialized
        # the serialized layer is plain JSON
        json.loads(json.dumps(serialized))


class TestRenderSmoke:
    def teardown_method(self):
        plt.close("all")

    def test_render_continuous(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot()
            .grouping(group="grouping_1")
            .labels(ylabel="value")
            .add(Aggregate(err_func="sem"), Line(), Marker(), ErrorBand())
        )
        plot.plot(y="y", x="x", data=data)

        assert plot.plotter is not None
        assert len(plot.plotter.axes) == 1
        assert plot.plotter.axes[0].has_data()
        assert plot.plotter.axes[0].get_ylabel() == "value"

    def test_render_continuous_faceted(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot()
            .grouping(group="grouping_1")
            .facet(facet=True, facet_title=True)
            .add(Aggregate(err_func="sem"), Line(), ErrorBar())
        )
        plot.plot(y="y", x="x", data=data)

        assert len(plot.plotter.axes) == 3

    def test_render_categorical(self, one_grouping):
        data, _ = one_grouping
        plot = (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .labels(ylabel="value")
            .add(Identity(), Marker(), width=0.5, seed=42)
            .add(Aggregate(err_func="sem"), SummaryLine(width=0.8), ErrorBar())
        )
        plot.plot(y="y", data=data)

        assert plot.plotter is not None
        ax = plot.plotter.axes[0]
        assert ax.has_data()
        # categorical ticks come from the layout context
        assert list(ax.get_xticks()) == [0, 1, 2]

    def test_render_categorical_jitteru(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        plot = CategoricalPlot().grouping(group="grouping_1")
        plot.add(Identity(unique_id="unique_grouping"), Marker(marker=["o", "s", "^"]), width=0.9)
        plot.add(Aggregate(unique_id="unique_grouping"), SummaryLine(width=0.8))
        plot.plot(y="y", data=data)

        assert plot.plotter is not None
        ax = plot.plotter.axes[0]
        assert ax.has_data()
        # one scatter collection per (group, uid): 2 groups x 3 subjects
        scatters = [
            collection for collection in ax.collections if isinstance(collection, matplotlib.collections.PathCollection)
        ]
        assert len(scatters) == 6
        # subjects sit on even columns within each group slot
        columns = {round(collection.get_offsets()[0, 0], 6) for collection in scatters}
        assert columns == {-0.3, 0.0, 0.3, 0.7, 1.0, 1.3}
        # three marker symbols cycle over the subject columns
        symbols = {tuple(np.round(collection.get_paths()[0].vertices.flatten(), 6)) for collection in scatters}
        assert len(symbols) == 3

    def test_render_categorical_bars(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot().grouping(group="grouping_1").add(Aggregate(), Bar(), Fill())
        plot.plot(y="y", data=data)

        assert plot.plotter.axes[0].patches

    def test_render_numpy_arrays_continuous(self):
        plot = LinePlot().add(Identity(), Line(), Marker())
        plot.plot(y=np.random.normal(size=30), x=np.arange(30.0))

        assert plot.plotter is not None
        assert plot.plotter.axes[0].has_data()

    def test_render_numpy_array_categorical_jitter(self):
        plot = CategoricalPlot().add(Identity(), Marker(), width=0.9)
        plot.plot(y=np.random.normal(size=50))

        assert plot.plotter is not None
        assert plot.plotter.axes[0].has_data()
