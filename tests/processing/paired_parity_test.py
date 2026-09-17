import numpy as np
import pytest

from lithos import CategoricalPlot, Identity
from lithos.plotting.processing import CategoricalProcessor
from lithos.utils import DataHolder


class TestPairedParity:
    """Parity: Identity(unique_id=...) per-subject geometry vs legacy _paired.

    The new-API paired decomposition is Identity(unique_id=...) + Line/Marker;
    the per-subject y series it produces must match the legacy pivot-based
    _paired connector geometry for the same data (raw branch, agg_func=None).
    """

    @pytest.mark.parametrize(
        "fixture, subgroup",
        [
            ("one_grouping_with_unique_ids", None),
            ("two_grouping_with_unique_ids", "grouping_2"),
        ],
    )
    def test_identity_matches_legacy_paired(self, request, fixture, subgroup):
        data, _ = request.getfixturevalue(fixture)
        levels = ("grouping_1",) if subgroup is None else ("grouping_1", "grouping_2")

        plot = (
            CategoricalPlot(data)
            .grouping(group="grouping_1", subgroup=subgroup)
            .paired(unique_id="unique_grouping", index="x")
            .plot_data(y="y")
        )
        processor = CategoricalProcessor(markers=("o", "X", "^", "s", "d"), hatches=("/", "o", "-", "*", "+"))
        output, _ = processor(plot.data, plot.metadata())
        legacy = output[0]  # MarkerLinePlotData from legacy _paired

        geometry = Identity(unique_id="unique_grouping")(DataHolder(data), y="y", x="x", levels=levels)

        n_levels = len(levels)
        group_keys = {key[:n_levels] for key in geometry}
        assert len(legacy.y_data) == len(group_keys)

        for i, gkey in enumerate(legacy.group_labels):
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            gkey = tuple(gkey)
            # legacy y_data[i] is (n_pairs, n_subjects): column j is subject j (uid ascending)
            y_matrix = np.asarray(legacy.y_data[i])
            uids = sorted({k[-1] for k in geometry if k[:n_levels] == gkey})
            assert y_matrix.shape == (30, len(uids))
            for j, uid in enumerate(uids):
                # both order the subject series by ascending order (x) values
                np.testing.assert_allclose(y_matrix[:, j], geometry[gkey + (uid,)]["y"])
