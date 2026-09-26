"""Tests for the spec-based plot API (lithos.plotting.spec)."""

import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lithos.plotting.elements import Bar, ErrorBand, ErrorBar, Fill, Line, Marker
from lithos.plotting.spec import CategoricalPlot, LinePlot
from lithos.plotting.spec.metadata import (
    build_element,
    build_transform,
    layer_to_json,
    to_jsonable,
)
from lithos.plotting.spec.resolver import locate_key, resolve_layers
from lithos.plotting.transforms import Aggregate, Identity


class TestAddValidation:
    def test_rejects_non_transform(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data)
        with pytest.raises(TypeError):
            plot.add("mean", Line(), y="y")

    def test_rejects_non_element(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data)
        with pytest.raises(TypeError):
            plot.add(Identity(), "marker", y="y")

    def test_requires_at_least_one_element(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data)
        with pytest.raises(ValueError):
            plot.add(Identity(), y="y")

    def test_rejects_unsupported_position(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot(data)
        with pytest.raises(ValueError):
            plot.add(Identity(), Marker(), y="y", position="stack")

    def test_rejects_jitter_on_line_layout(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data)
        with pytest.raises(ValueError):
            plot.add(Identity(), Marker(), y="y", position="jitter")


class TestAddLayers:
    def test_layers_hold_raw_objects(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        identity = Identity()
        plot.add(identity, Marker(), y="y", x="x")

        assert len(plot.layers) == 1
        layer = plot.layers[0]
        assert layer["transform"] is identity
        assert layer["transform"].name == "identity"
        assert all(isinstance(element, Marker) for element in layer["elements"])
        # no data is processed at .add() time
        assert "geometry" not in layer

    def test_geometry_honors_grouping_set_after_add(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data)
        plot.add(Identity(), Marker(), y="y", x="x")
        plot.grouping(group="grouping_1")

        processed = plot._process_data()
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
        plot = LinePlot(data).plot_data(y="y", x="x")
        plot.add(Identity(), Marker())

        geometry = plot._process_data()[0]["geometry"]
        assert len(geometry) == 1
        single = geometry[("",)]
        assert single["n"] == 90
        assert single["x"].shape == (90,)
        assert single["y"].shape == (90,)

    def test_chained_layers_are_independent(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        plot.add(Identity(), Marker(), y="y", x="x")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar(), y="y", x="x")

        assert len(plot.layers) == 2
        assert plot.layers[0]["transform"].name == "identity"
        assert plot.layers[1]["transform"].name == "aggregate"
        processed = plot._process_data()
        assert processed[0]["geometry"] is not processed[1]["geometry"]

    def test_plot_level_default_columns(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1").plot_data(y="y", x="x")
        plot.add(Identity(), Marker())
        layer = plot._process_data()[0]
        assert layer["y"] == "y"
        assert layer["x"] == "x"

    def test_layer_column_overrides_plot_default(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1").plot_data(y="y")
        plot.add(Identity(), Marker(), y="x", x="y")
        layer = plot._process_data()[0]
        assert layer["y"] == "x"
        assert layer["x"] == "y"

    def test_position_defaults_per_layout(self, one_grouping):
        data, _ = one_grouping
        line_layer = LinePlot(data).add(Identity(), Marker(), y="y", x="x").layers[0]
        cat_layer = CategoricalPlot(data).add(Identity(), Marker(), y="y").layers[0]
        assert line_layer["position"] == "passthrough"
        assert cat_layer["position"] == "dodge"


class TestLayoutContext:
    def test_line_context_facet(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1").facet(facet=True)
        context = plot._layout_context()
        assert context["layout"] == "continuous"
        assert context["facet"] is True
        assert sorted(context["loc_dict"].values()) == [0, 1, 2]

    def test_categorical_context_positions(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1").spacing(group_spacing=0.9)
        context = plot._layout_context()
        assert context["layout"] == "categorical"
        assert context["width"] == 1.0
        assert context["ticks"] == [0, 1, 2]
        assert sorted(context["loc_dict"].values()) == [0.0, 1.0, 2.0]

    def test_categorical_subgroup_context(self, two_grouping):
        data, _ = two_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1", subgroup="grouping_2")
        context = plot._layout_context()
        assert context["width"] < 1.0
        assert len(context["subticks"]) == len(context["unique_groups"])


class TestResolver:
    def test_dodge_positions_match_group_locations(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1")
        plot.add(Aggregate(), Marker(), y="y")
        context = plot._layout_context()
        resolved = resolve_layers(plot._process_data(), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            assert geometry["position"] == context["loc_dict"][gkey]
            np.testing.assert_allclose(geometry["x"], [context["loc_dict"][gkey]])
            assert geometry["center"].size == 1

    def test_jitter_positions_stay_within_group_width(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1")
        plot.add(Identity(), Marker(), y="y", position="jitter", seed=42)
        context = plot._layout_context()
        resolved = resolve_layers(plot._process_data(), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            loc = context["loc_dict"][gkey]
            assert np.all(geometry["x"] >= loc - context["width"])
            assert np.all(geometry["x"] <= loc + context["width"])
            assert geometry["x"].shape == geometry["y"].shape

    def test_paired_geometry_resolves_by_group_prefix(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        plot = CategoricalPlot(data).grouping(group="grouping_1")
        plot.add(Identity(unique_id="unique_grouping"), Marker(), y="y", position="dodge")
        context = plot._layout_context()
        resolved = resolve_layers(plot._process_data(), context)[0]
        for gkey, geometry in resolved["geometry"].items():
            assert len(gkey) == 2
            assert geometry["position"] == context["loc_dict"][gkey[:1]]

    def test_locate_key_requires_matching_prefix(self):
        with pytest.raises(KeyError):
            locate_key(("a", "b"), {("z",): 0.0})


class TestMetadata:
    def test_metadata_is_json_serializable(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar(), y="y", x="x")
        metadata = plot.metadata()

        assert metadata["version"] == 2
        assert metadata["layout"] == "continuous"
        assert metadata["grouping"]["group"] == "grouping_1"
        assert len(metadata["layers"]) == 1
        assert metadata["layers"][0]["transform"]["name"] == "aggregate"
        assert metadata["layers"][0]["elements"][0]["type"] == "line"
        # layers are pure specs — no geometry is stored
        assert "groups" not in metadata["layers"][0]
        assert "geometry" not in metadata["layers"][0]
        # the whole document round-trips through json
        assert json.loads(json.dumps(metadata)) == json.loads(json.dumps(plot.metadata()))

    def test_metadata_roundtrip_via_dict(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        plot.add(Aggregate(err_func="sem"), Line(), ErrorBar(), y="y", x="x")

        rebuilt = LinePlot(data).load_metadata(plot.metadata())
        assert len(rebuilt.layers) == 1
        assert rebuilt._grouping == plot._grouping
        rebuilt_layer, layer = rebuilt.layers[0], plot.layers[0]
        assert rebuilt_layer["transform"] == layer["transform"]
        assert rebuilt_layer["elements"] == layer["elements"]
        rebuilt_geometry = rebuilt._process_data()[0]["geometry"]
        for gkey, geometry in plot._process_data()[0]["geometry"].items():
            for key in ("x", "center", "error_low", "error_high"):
                np.testing.assert_allclose(np.asarray(rebuilt_geometry[gkey][key]), np.asarray(geometry[key]))

    def test_metadata_roundtrip_via_file(self, one_grouping, tmp_path):
        data, _ = one_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1")
        plot.add(Identity(), Marker(), y="y", position="jitter")
        plot.add(Aggregate(err_func="sem"), Marker(), ErrorBar(), y="y")
        plot.save_metadata(tmp_path / "spec_plot")
        assert (tmp_path / "spec_plot.json").exists()

        rebuilt = CategoricalPlot(data).load_metadata(tmp_path / "spec_plot")
        assert len(rebuilt.layers) == 2
        assert rebuilt._layout_options == plot._layout_options
        assert rebuilt.layers[1]["position"] == "dodge"

    def test_load_rejects_wrong_layout(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        plot.add(Identity(), Marker(), y="y", x="x")
        with pytest.raises(ValueError):
            CategoricalPlot(data).load_metadata(plot.metadata())

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

    def test_layer_to_json_serializes_held_objects(self, one_grouping):
        data, _ = one_grouping
        plot = LinePlot(data).grouping(group="grouping_1")
        plot.add(Identity(), Marker(), y="y", x="x")
        serialized = layer_to_json(plot.layers[0])

        assert serialized["transform"]["name"] == "identity"
        assert serialized["elements"][0]["type"] == "marker"
        # pure layer spec — no geometry is stored
        assert "geometry" not in serialized
        assert "groups" not in serialized
        # the serialized layer is plain JSON
        json.loads(json.dumps(serialized))


class TestRenderSmoke:
    def teardown_method(self):
        plt.close("all")

    def test_render_continuous(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot(data)
            .grouping(group="grouping_1")
            .plot_data(y="y", x="x", ylabel="value")
            .add(Aggregate(err_func="sem"), Line(), Marker(), ErrorBand(), y="y", x="x")
        )
        plot.plot()

        assert plot.plotter is not None
        assert len(plot.plotter.axes) == 1
        assert plot.plotter.axes[0].has_data()
        assert plot.plotter.axes[0].get_ylabel() == "value"

    def test_render_continuous_faceted(self, one_grouping):
        data, _ = one_grouping
        plot = (
            LinePlot(data)
            .grouping(group="grouping_1")
            .facet(facet=True, facet_title=True)
            .add(Aggregate(err_func="sem"), Line(), ErrorBar(), y="y", x="x")
        )
        plot.plot()

        assert len(plot.plotter.axes) == 3

    def test_render_categorical(self, one_grouping):
        data, _ = one_grouping
        plot = (
            CategoricalPlot(data)
            .grouping(group="grouping_1")
            .plot_data(y="y", ylabel="value")
            .add(Identity(), Marker(), position="jitter")
            .add(Aggregate(err_func="sem"), Marker(), ErrorBar())
        )
        plot.plot()

        assert plot.plotter is not None
        ax = plot.plotter.axes[0]
        assert ax.has_data()
        # categorical ticks come from the layout context
        assert list(ax.get_xticks()) == [0, 1, 2]

    def test_render_categorical_bars(self, one_grouping):
        data, _ = one_grouping
        plot = CategoricalPlot(data).grouping(group="grouping_1").plot_data(y="y")
        plot.add(Aggregate(), Bar(), Fill())
        plot.plot()

        assert plot.plotter.axes[0].patches
