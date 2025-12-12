import numpy as np

import pytest

from lithos.utils import DataHolder


@pytest.mark.parametrize(
    "test_groupings",
    [
        "one_grouping",
        "one_grouping_with_unique_ids",
        "two_grouping",
        "two_grouping_with_unique_ids",
    ],
)
class DataHolderTestClass:
    @pytest.fixture
    def _fixt(self, test_groupings, request) -> tuple[dict, tuple[int, int, int, int]]:
        data, x = request.getfixturevalue(test_groupings)
        return data, x

    def get_rows(self, x):
        rows = 1
        for i in x:
            if i > 0:
                rows *= i
        return rows

    def get_cols(self, x):
        return sum(True for i in x if i > 0) + 1

    def get_unique_groups(self, x):
        groups = 1
        for i in x[:3]:
            if i > 0:
                groups *= i
        return groups

    def get_unique_group_names(self, data, cols):
        names = set()
        names = set(zip(*[data[i] for i in cols]))
        return names

    def test_shape(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]) -> None:
        df, x = fixt
        cols = self.get_cols(x)
        rows = self.get_rows(x)
        assert df.shape == (rows, cols)

    def test_size(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]) -> None:
        df, x = fixt
        cols = self.get_cols(x)
        rows = self.get_rows(x)
        assert df.size == rows * cols

    def test_len(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]) -> None:
        df, x = fixt
        cols = self.get_cols(x)
        rows = self.get_rows(x)
        assert len(df) == rows * cols


class StringColumnsDataHolder(DataHolderTestClass):

    def get_column_names(self, x):
        cols = sum(True for i in x[:3] if i > 0)
        if cols == 1:
            return ("grouping_1",)
        elif cols == 2:
            if x[2] == 0:
                return ("grouping_1", "grouping_2")
            else:
                return ("grouping_1", "unique_grouping")
        elif cols == 3:
            return ("grouping_1", "grouping_2", "unique_grouping")

    def test_max(
        self,
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
        _fixt: tuple[dict, tuple[int, int, int, int]],
    ) -> None:
        df1, _ = _fixt
        df2, _ = fixt
        assert max(df1["y"]) == df2.max("y")

    def test_min(
        self,
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
        _fixt: tuple[dict, tuple[int, int, int, int]],
    ) -> None:
        df1, _ = _fixt
        df2, _ = fixt
        assert min(df1["y"]) == df2.min("y")

    def test_groups(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]) -> None:
        df, x = fixt
        names = self.get_column_names(x)
        gb = df.groups(names)
        ngroups = self.get_unique_groups(x)
        assert len(gb) == ngroups

        group_names = self.get_unique_group_names(df, names)

        assert len(group_names.difference(gb.keys())) == 0

    def test_groupby(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        df, x = fixt
        names = self.get_column_names(x)
        gb = df.groupby("y", names)
        ngroups = self.get_unique_groups(x)

        assert len(gb) == ngroups

    def test_contains(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        df, x = fixt
        names = self.get_column_names(x)
        for i in names:
            assert i in df

    def test_getitem(
        self,
        _fixt: tuple[dict, tuple[int, int, int, int]],
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
    ):
        data1, _ = _fixt
        data2, _ = fixt
        indices = np.arange(5)

        temp = data2[indices, "y"]

        assert list(temp) == [data1["y"][i] for i in indices]
