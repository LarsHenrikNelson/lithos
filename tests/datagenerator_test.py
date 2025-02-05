import pytest

from lithos.utils import create_synthetic_data


@pytest.mark.parametrize(
    "n_groups, n_subgroups, n_unique_ids, n_points",
    [(1, 0, 0, 30), (1, 3, 0, 50), (2, 2, 0, 70), (2, 2, 1, 5)],
)
def test_create_synthetic_data(n_groups, n_subgroups, n_unique_ids, n_points):
    # Test creating synthetic data
    data = create_synthetic_data(n_groups, n_subgroups, n_unique_ids, n_points)
    group = set(data["grouping_1"])
    temp = len(group)

    assert len(group) == n_groups
    if n_subgroups != 0:
        subgroup = set(data["grouping_2"])
        assert len(subgroup) == n_subgroups
        temp *= len(subgroup)

    assert len(data["y"]) / temp == n_points
