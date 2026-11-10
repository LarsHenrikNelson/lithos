"""Composition tests for the spec-API sugar verbs (the Phase 2.5 method facades).

A verb is a thin factory over ``.add()``, so the contract under test is the
produced layer list itself: each verb must leave exactly the layers of the
equivalent hand-written ``.add(...)`` composition - same transform fields,
same element values, same layer kwargs. Role overrides, intent fan-out and
layer-kwarg forwarding pin the argument contract; geometry correctness is
inherited from the parity suite (``tests/processing/``, the same
transforms), plus one end-to-end render smoke test per verb and layout.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import pytest

from lithos.plotting.elements import Bar, ErrorBand, Fill, Line, Marker
from lithos.plotting.spec import CategoricalPlot, LinePlot
from lithos.plotting.transforms import ECDF, KDE, Fit, Histogram, Identity


class TestScatterVerb:
    """scatter == .add(Identity, Line?, Marker) with the pairing knob on the transform."""

    def test_defaults_to_points_only(self):
        assert LinePlot().scatter().layers == LinePlot().add(Identity(), Marker()).layers

    def test_connect_adds_the_paired_connector(self):
        plot = LinePlot().scatter(unique_id="mouse", connect=True)
        assert plot.layers == LinePlot().add(Identity(unique_id="mouse"), Line(), Marker()).layers

    def test_false_drops_the_connector_even_with_connect(self):
        plot = LinePlot().scatter(connect=True, line=False)
        assert plot.layers == LinePlot().add(Identity(), Marker()).layers

    def test_explicit_line_role_without_connect(self):
        plot = LinePlot().scatter(line=Line(linestyle="--"))
        assert plot.layers == LinePlot().add(Identity(), Line(linestyle="--"), Marker()).layers

    def test_layer_kwargs_forward_verbatim(self):
        plot = CategoricalPlot().scatter(width=0.5, jitter_type="dist", seed=30, position="passthrough")
        expected = CategoricalPlot().add(
            Identity(), Marker(), width=0.5, jitter_type="dist", seed=30, position="passthrough"
        )
        assert plot.layers == expected.layers

    def test_marker_false_surfaces_the_add_error(self):
        with pytest.raises(ValueError, match="at least one element"):
            LinePlot().scatter(marker=False)

    def test_role_override_wins_over_intent(self):
        explicit = Marker(marker="X")
        plot = LinePlot().scatter(color="red", alpha=0.1, marker=explicit)
        assert plot.layers == LinePlot().add(Identity(), explicit).layers

    def test_intent_fans_out_primary_fields(self):
        plot = LinePlot().scatter(color="red", alpha=0.1)
        assert plot.layers[0]["elements"] == [Marker(markercolor="red", alpha=0.1)]


class TestKdeVerb:
    """kde == .add(KDE, Fill, Line?) with a layout-aware line default."""

    def test_continuous_default_fills_under_the_curve(self):
        assert LinePlot().kde().layers == LinePlot().add(KDE(), Fill(), Line()).layers

    def test_categorical_default_is_the_violin_fill(self):
        assert CategoricalPlot().kde().layers == CategoricalPlot().add(KDE(), Fill()).layers

    def test_stat_kwargs_forward_verbatim(self):
        plot = LinePlot().kde(bw="silverman", kde_length=64)
        assert plot.layers == LinePlot().add(KDE(bw="silverman", kde_length=64), Fill(), Line()).layers

    def test_density_kwargs_forward_verbatim(self):
        plot = LinePlot().kde(unique_id="mouse", agg_func="mean", err_func="sem")
        expected = LinePlot().add(KDE(unique_id="mouse", agg_func="mean", err_func="sem"), Fill(), Line())
        assert plot.layers == expected.layers

    def test_fill_false_keeps_the_outline_only(self):
        assert LinePlot().kde(fill=False).layers == LinePlot().add(KDE(), Line()).layers

    def test_intent_reaches_both_roles(self):
        plot = LinePlot().kde(color="red", alpha=0.3)
        assert plot.layers[0]["elements"] == [
            Fill(fillcolor="red", fillalpha=0.3),
            Line(linecolor="red", linealpha=0.3),
        ]

    def test_style_forwards_the_violin_form(self):
        plot = CategoricalPlot().kde(style="left")
        assert plot.layers == CategoricalPlot().add(KDE(), Fill(), style="left").layers


class TestHistVerb:
    """hist == .add(Histogram, Fill, Bar) with the step outline as an optional role."""

    def test_default_draws_faces_and_outlines(self):
        assert LinePlot().hist().layers == LinePlot().add(Histogram(), Fill(), Bar()).layers

    def test_stat_kwargs_forward_verbatim(self):
        plot = LinePlot().hist(bins=10, bin_limits="common", stat="count")
        expected = LinePlot().add(Histogram(bins=10, bin_limits="common", stat="count"), Fill(), Bar())
        assert plot.layers == expected.layers

    def test_step_outline_is_role_composition(self):
        plot = LinePlot().hist(fill=False, bar=False, step=Line())
        assert plot.layers == LinePlot().add(Histogram(), Line()).layers

    def test_intent_reaches_faces_and_outlines(self):
        plot = LinePlot().hist(color="blue", alpha=0.6)
        assert plot.layers[0]["elements"] == [
            Fill(fillcolor="blue", fillalpha=0.6),
            Bar(edgecolor="blue", edge_alpha=0.6),
        ]


class TestEcdfVerb:
    """ecdf == .add(ECDF, Fill, Line); no axis side effects (the probability axis follows the value column)."""

    def test_default_composition(self):
        assert LinePlot().ecdf().layers == LinePlot().add(ECDF(), Fill(), Line()).layers

    def test_variant_kwargs_forward_verbatim(self):
        plot = LinePlot().ecdf(ecdf_type="spline", ecdf_args={"size": 100})
        expected = LinePlot().add(ECDF(ecdf_type="spline", ecdf_args={"size": 100}), Fill(), Line())
        assert plot.layers == expected.layers

    def test_no_axis_side_effect(self):
        plot = LinePlot()
        plot.ecdf()
        assert plot.plot_format["axis"]["ylim"] == (None, None)

    def test_line_only_composition(self):
        assert LinePlot().ecdf(fill=False).layers == LinePlot().add(ECDF(), Line()).layers


class TestFitVerb:
    """fit == .add(Fit, ErrorBand?, Line) with the band tied to ci_func."""

    def test_default_draws_band_and_line(self):
        assert LinePlot().fit().layers == LinePlot().add(Fit(), ErrorBand(), Line()).layers

    def test_no_ci_drops_the_band_by_default(self):
        plot = LinePlot().fit(ci_func=None)
        assert plot.layers == LinePlot().add(Fit(ci_func=None), Line()).layers

    def test_band_false_draws_the_bare_fit(self):
        plot = LinePlot().fit(band=False)
        assert plot.layers == LinePlot().add(Fit(), Line()).layers

    def test_stats_forward_verbatim(self):
        plot = LinePlot().fit(
            fit_func="polynomial",
            fit_args={"degree": 2},
            ci_func="pi",
            unique_id="mouse",
            agg_func="mean",
            err_func="sem",
        )
        expected = LinePlot().add(
            Fit(
                fit_func="polynomial",
                fit_args={"degree": 2},
                ci_func="pi",
                unique_id="mouse",
                agg_func="mean",
                err_func="sem",
            ),
            ErrorBand(),
            Line(),
        )
        assert plot.layers == expected.layers

    def test_intent_reaches_band_and_line(self):
        plot = LinePlot().fit(color="green", alpha=0.25)
        assert plot.layers[0]["elements"] == [
            ErrorBand(fillcolor="green", fillalpha=0.25),
            Line(linecolor="green", linealpha=0.25),
        ]


class TestVerbMetadataReplay:
    """Verbs go through .add(), so their layers serialize and reload identically."""

    def test_metadata_round_trip_rebuilds_verb_layers(self):
        plot = LinePlot().grouping(group="grouping_1").scatter(color="red", alpha=0.4)
        rebuilt = LinePlot().load_metadata(plot.metadata())
        assert rebuilt.layers == plot.layers


class TestVerbRenderSmoke:
    """One end-to-end render per verb and layout on the Agg backend."""

    def test_scatter_renders_continuous(self, one_grouping):
        data, _ = one_grouping
        LinePlot().grouping(group="grouping_1").scatter().plot(y="y", x="x", data=data)

    def test_scatter_renders_categorical(self, one_grouping):
        data, _ = one_grouping
        CategoricalPlot().grouping(group="grouping_1").scatter().plot(y="y", data=data)

    def test_paired_scatter_renders(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        (
            CategoricalPlot()
            .grouping(group="grouping_1")
            .scatter(unique_id="unique_grouping", connect=True)
            .plot(y="y", x="x", data=data)
        )

    def test_kde_renders_continuous(self, one_grouping):
        data, _ = one_grouping
        LinePlot().grouping(group="grouping_1").kde().plot(y="y", data=data)

    def test_kde_renders_violin(self, one_grouping):
        data, _ = one_grouping
        CategoricalPlot().grouping(group="grouping_1").kde().plot(y="y", data=data)

    def test_kde_split_violin_renders(self, two_grouping):
        data, _ = two_grouping
        (
            CategoricalPlot()
            .grouping(group="grouping_1", subgroup="grouping_2")
            .kde(style="alternate")
            .plot(y="y", data=data)
        )

    def test_hist_renders(self, one_grouping):
        data, _ = one_grouping
        LinePlot().grouping(group="grouping_1").hist(bins=10).plot(y="y", data=data)

    def test_stacked_hist_renders(self, one_grouping):
        data, _ = one_grouping
        (
            LinePlot()
            .grouping(group="grouping_1")
            .hist(bins=10, bin_limits="common", style="stack")
            .plot(y="y", data=data)
        )

    def test_ecdf_renders(self, one_grouping):
        data, _ = one_grouping
        LinePlot().grouping(group="grouping_1").ecdf().plot(y="y", data=data)

    def test_fit_renders(self, one_grouping):
        data, _ = one_grouping
        LinePlot().grouping(group="grouping_1").fit().plot(y="y", x="x", data=data)
