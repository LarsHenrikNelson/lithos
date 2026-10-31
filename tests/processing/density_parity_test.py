import numpy as np
import pytest

from lithos import KDE, CategoricalPlot, Histogram, LinePlot
from lithos.plotting.elements import Fill
from lithos.plotting.processing import CategoricalProcessor, LineProcessor
from lithos.plotting.spec import CategoricalPlot as SpecCategoricalPlot
from lithos.plotting.spec import LinePlot as SpecLinePlot
from lithos.plotting.spec.resolver import resolve_layers
from lithos.utils import DataHolder


def _positive_data():
    rng = np.random.default_rng(11)
    vals = np.concatenate([rng.uniform(1.0, 40.0, size=40), rng.uniform(5.0, 60.0, size=40)])
    return {"grouping_1": [0] * 40 + [1] * 40, "y": vals.tolist()}


def _levels(subgroup):
    return ("grouping_1",) if subgroup is None else ("grouping_1", "grouping_2")


def _as_group_key(label):
    return tuple(label) if isinstance(label, tuple) else (label,)


class TestKDEParity:
    """Parity: ``KDE`` transform geometry vs the legacy ``_kde`` processor output.

    The legacy ``LinePlot.kde()`` and the new ``KDE`` transform share the stats
    layer, so their processed grid/density/error values must match for the
    same data: the per-group KDE, the per-subject KDEs (``unique_id``,
    ``agg_func=None``) and the aggregated per-subject KDEs (``unique_id`` +
    ``agg_func``/``err_func`` on the shared grid).

    ``bw`` is passed explicitly on both sides: the legacy method defaults to
    ``"silverman"``, the transform to ``"ISJ"`` — parity pins the shared
    computation, not the defaults. Only the matching axis transform is ever
    set: the legacy ``_kde`` crosses the transforms when both are given (the
    bug fixed in the new API), so parity with both set would compare against
    wrongly-transformed legacy geometry.
    """

    def _legacy(self, data, subgroup, **kde_kwargs):
        plot = LinePlot(data).grouping(group="grouping_1", subgroup=subgroup).kde(**kde_kwargs).plot_data(y="y")
        processor = LineProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping", None),
            ("two_grouping", "grouping_2"),
        ],
    )
    def test_group_kde_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup, bw="ISJ")
        geometry = KDE(bw="ISJ")(DataHolder(data), y="y", levels=_levels(subgroup))

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            assert legacy.error_data[i] is None

    def test_group_kde_transform_matches_legacy(self):
        # only the matching axis transform is set (legacy crosses them when both are given)
        data = _positive_data()
        plot = LinePlot(data).grouping(group="grouping_1").transform(ytransform="log10").kde(bw="ISJ").plot_data(y="y")
        processor = LineProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        legacy = output[0]
        geometry = KDE(bw="ISJ")(DataHolder(data), y="y", levels=("grouping_1",), ytransform="log10")

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_per_subject_kdes_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = _levels(subgroup)
        legacy = self._legacy(data, subgroup, bw="ISJ", unique_id="unique_grouping", agg_func=None)
        geometry = KDE(bw="ISJ", unique_id="unique_grouping")(DataHolder(data), y="y", levels=levels)

        # legacy appends one KDE per subject (labeled by the group key); the
        # transform nests the subject into the key. Both order subjects by
        # np.unique(uid) within groups in data order, so entries line up.
        items = list(geometry.items())
        assert len(legacy.x_data) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[: len(levels)] == _as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_aggregated_kdes_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        # kde_length passed explicitly: the legacy method defaults it to 256
        # with agg_func, the transform derives it from the data size
        legacy = self._legacy(
            data, subgroup, bw="ISJ", kde_length=256, unique_id="unique_grouping", agg_func="mean", err_func="sem"
        )
        geometry = KDE(bw="ISJ", kde_length=256, unique_id="unique_grouping", agg_func="mean", err_func="sem")(
            DataHolder(data), y="y", levels=_levels(subgroup)
        )

        assert len(legacy.x_data) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            # shared grid, aggregated density and per-grid-point errors
            np.testing.assert_allclose(legacy.x_data[i], g["x"])
            np.testing.assert_allclose(legacy.y_data[i], g["y"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_low"])
            np.testing.assert_allclose(legacy.error_data[i], g["error_high"])


class TestHistogramParity:
    """Parity: ``Histogram`` transform geometry vs the legacy ``_hist`` processor output.

    ``nbins``/``stat``/``bin_limits`` are passed explicitly on both sides: the
    legacy method defaults to ``stat="count"``, the transform to
    ``"density"``. The legacy rectangle geometry carries bin *left* edges;
    the transform emits the full ``edges`` array (``centers``/``binwidth``
    derived from it).

    ``bin_limits="common"`` pins the shared computation exactly: with
    ``bin_limits=None`` the legacy ``_hist`` computes the auto bin range from
    the *first* group and reuses it for every later group (whose values can
    then fall outside the bins), while the transform computes true per-group
    ranges — a deliberate fix, so per-group-range parity cannot match. The
    legacy per-subject branch also double-truncates the bin edges (it appends
    left edges where every other branch appends full edges), silently
    dropping the last bin; the per-subject parity below pins the shared
    heights and the surviving edges.
    """

    def _legacy(self, data, subgroup, **hist_kwargs):
        plot = LinePlot(data).grouping(group="grouping_1", subgroup=subgroup).hist(**hist_kwargs).plot_data(y="y")
        processor = LineProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping", None),
            ("two_grouping", "grouping_2"),
        ],
    )
    def test_group_histogram_matches_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, subgroup, nbins=10, stat="density", bin_limits="common")
        geometry = Histogram(bins=10, stat="density", bin_limits="common")(
            DataHolder(data), y="y", levels=_levels(subgroup)
        )

        assert len(legacy.heights) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.heights[i], g["height"])
            np.testing.assert_allclose(legacy.bins[i], g["edges"][:-1])
            np.testing.assert_allclose(legacy.binwidths[i], g["binwidth"])
            np.testing.assert_allclose(legacy.bottoms[i], 0.0)

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_per_subject_histograms_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = _levels(subgroup)
        legacy = self._legacy(
            data, subgroup, nbins=10, stat="density", bin_limits="common", unique_id="unique_grouping"
        )
        geometry = Histogram(bins=10, stat="density", bin_limits="common", unique_id="unique_grouping")(
            DataHolder(data), y="y", levels=levels
        )

        # per-subject entries on the shared common edges, ordered like the KDE parity
        items = list(geometry.items())
        assert len(legacy.heights) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[: len(levels)] == _as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.heights[i], g["height"])
            # legacy double-truncates the per-subject bin edges (the nested
            # branch appends left edges, RectanglePlotData truncates again),
            # silently dropping the last bin; the transform keeps every bin
            np.testing.assert_allclose(legacy.bins[i], g["edges"][:-2])
            np.testing.assert_allclose(legacy.binwidths[i], g["binwidth"][:-1])

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_aggregated_histograms_match_legacy(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(
            data,
            subgroup,
            nbins=10,
            stat="density",
            bin_limits="common",
            unique_id="unique_grouping",
            agg_func="mean",
        )
        geometry = Histogram(
            bins=10, stat="density", bin_limits="common", unique_id="unique_grouping", agg_func="mean"
        )(DataHolder(data), y="y", levels=_levels(subgroup))

        assert len(legacy.heights) == len(geometry)
        for i, label in enumerate(legacy.group_labels):
            g = geometry[_as_group_key(label)]
            np.testing.assert_allclose(legacy.heights[i], g["height"])
            np.testing.assert_allclose(legacy.bins[i], g["edges"][:-1])
            # legacy hist aggregation carries no errors
            assert g["error_low"] is None
            assert g["error_high"] is None


class TestViolinParity:
    """Parity: resolved ``KDE`` violin geometry vs the legacy ``_violin`` processor output.

    The legacy ``CategoricalPlot.violin()`` and the new resolver share the
    stats layer and the normalization (the density is scaled so its maximum
    spans the layer width across the slot: legacy ``width/2`` for the ``full``
    style, resolver ``width * extent / 2``). ``bw``/``kde_length`` are passed
    explicitly (legacy defaults ``bw="silverman"``, ``kde_length=128``; the
    transform defaults ``"ISJ"``/``None``).

    Flat grouping only: the legacy violin width is absolute axis units (the
    spec resolver scales by the slot extent, so subgrouped slots differ) and
    the legacy nested ``split`` ``uwidth`` formula differs from the
    even-column extent - both documented divergences. The legacy nested
    ``overlap`` placement is reproduced by ``style="overlap"`` below.
    """

    def _legacy(self, data, **violin_kwargs):
        plot = CategoricalPlot(data).grouping(group="grouping_1").violin(**violin_kwargs).plot_data(y="y")
        processor = CategoricalProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    def _resolved(self, data, transform, style=None):
        plot = SpecCategoricalPlot().grouping(group="grouping_1").add(transform, Fill(), style=style)
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        return resolve_layers(plot._process_data(holder, y="y"), context)[0]["geometry"]

    @pytest.mark.parametrize("fixture", ["one_grouping", "two_grouping"])
    def test_group_violin_matches_legacy(self, request, fixture):
        data, _ = request.getfixturevalue(fixture)
        legacy = self._legacy(data, bw="ISJ", kde_length=128)
        resolved = self._resolved(data, KDE(bw="ISJ", kde_length=128))

        assert len(legacy.x_data) == len(resolved)
        for i, label in enumerate(legacy.group_labels):
            g = resolved[_as_group_key(label)]
            assert g["value_axis"] == "y"
            # values on the value axis
            np.testing.assert_allclose(legacy.x_data[i], g["y"])
            # legacy half-spread (width 0.9 halved for the full style) == resolver spread
            np.testing.assert_allclose(legacy.y_data[i], g["spread"])
            assert legacy.location[i] == g["position"]

    def test_aggregated_violin_matches_legacy(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        legacy = self._legacy(data, bw="ISJ", kde_length=128, unique_id="unique_grouping", agg_func="mean")
        resolved = self._resolved(data, KDE(bw="ISJ", kde_length=128, unique_id="unique_grouping", agg_func="mean"))

        assert len(legacy.x_data) == len(resolved)
        for i, label in enumerate(legacy.group_labels):
            g = resolved[_as_group_key(label)]
            # shared grid aggregated across subjects, normalized to the slot
            np.testing.assert_allclose(legacy.x_data[i], g["y"])
            np.testing.assert_allclose(legacy.y_data[i], g["spread"])
            assert legacy.location[i] == g["position"]

    @pytest.mark.parametrize("style", ["left", "right", "alternate"])
    def test_split_violin_styles_match_legacy(self, one_grouping, style):
        data, _ = one_grouping
        legacy = self._legacy(data, bw="ISJ", kde_length=128, style=style)
        resolved = self._resolved(data, KDE(bw="ISJ", kde_length=128), style=style)

        # the legacy styles keep the un-halved width; with m = ymax/2 the
        # half violin spans [loc + m - y, loc + m] (left) / [loc - m, loc - m + y]
        # (right), which the resolver expresses as spread_low/spread_high offsets
        assert len(legacy.x_data) == len(resolved)
        for i, label in enumerate(legacy.group_labels):
            g = resolved[_as_group_key(label)]
            m = legacy.y_data[i].max() / 2.0
            side = ("left" if i % 2 == 0 else "right") if style == "alternate" else style
            np.testing.assert_allclose(legacy.x_data[i], g["y"])
            if side == "left":
                np.testing.assert_allclose(g["spread_high"], m)
                np.testing.assert_allclose(g["spread_low"], m - legacy.y_data[i])
            else:
                np.testing.assert_allclose(g["spread_low"], -m)
                np.testing.assert_allclose(g["spread_high"], legacy.y_data[i] - m)
            assert legacy.location[i] == g["position"]

    def test_overlap_violins_match_legacy(self, one_grouping_with_unique_ids):
        data, _ = one_grouping_with_unique_ids
        legacy = self._legacy(data, bw="ISJ", kde_length=128, unique_id="unique_grouping", unique_style="overlap")
        resolved = self._resolved(data, KDE(bw="ISJ", kde_length=128, unique_id="unique_grouping"), style="overlap")

        # per-subject violins in data order, all overlaid on the slot center
        # at the full (halved-width) normalization
        items = list(resolved.items())
        assert len(legacy.x_data) == len(items)
        for i, (gkey, g) in enumerate(items):
            assert gkey[:1] == _as_group_key(legacy.group_labels[i])
            np.testing.assert_allclose(legacy.x_data[i], g["y"])
            np.testing.assert_allclose(legacy.y_data[i], g["spread"])
            assert legacy.location[i] == g["position"]


class TestHistStackParity:
    """Parity: resolved ``Histogram`` stacking (``style``) vs legacy ``hist_type`` post-processing.

    ``bin_limits="common"`` on both sides: stacking compares heights across
    groups, so the bins must match (the legacy ``_post_process_density``
    stacks by bin index regardless; the resolver raises on mismatched edges).
    Groups stack in geometry order within each facet - the legacy row order.
    """

    def _legacy(self, data, **hist_kwargs):
        plot = LinePlot(data).grouping(group="grouping_1").hist(**hist_kwargs).plot_data(y="y")
        processor = LineProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        return output[0]

    def _resolved(self, data, style):
        plot = (
            SpecLinePlot()
            .grouping(group="grouping_1")
            .add(Histogram(bins=10, stat="density", bin_limits="common"), Fill(), style=style)
        )
        holder = DataHolder(data)
        context = plot._layout_context(holder)
        return resolve_layers(plot._process_data(holder, y="y"), context)[0]["geometry"]

    @pytest.mark.parametrize("style", ["stack", "fill"])
    def test_stacked_heights_and_bottoms_match_legacy(self, one_grouping, style):
        data, _ = one_grouping
        legacy = self._legacy(data, nbins=10, stat="density", bin_limits="common", hist_type=style)
        resolved = self._resolved(data, style)

        assert len(legacy.heights) == len(resolved)
        for i, label in enumerate(legacy.group_labels):
            g = resolved[_as_group_key(label)]
            np.testing.assert_allclose(legacy.heights[i], g["height"])
            np.testing.assert_allclose(legacy.bottoms[i], g["bottom"])
