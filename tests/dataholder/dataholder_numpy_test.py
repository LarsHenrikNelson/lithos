import numpy as np
import pytest

from .dataholder_test_class import DataHolderTestClass
from lithos.utils import DataHolder


class TestDataHolderNumpy(DataHolderTestClass):
    @pytest.fixture
    def fixt(
        self, _fixt: tuple[dict, tuple[int, int, int, int]]
    ) -> tuple[DataHolder, tuple[int, int, int, int]]:
        data, x = _fixt
        new_data = np.zeros((data["y"].size, len(data)))
        for index, i in enumerate(data.values()):
            new_data[:, index] = i
        data = DataHolder(new_data)
        return data, x

    def get_column_names(self, x):
        cols = sum(True for i in x[:3] if i > 0)
        if cols == 1:
            return (2,)
        elif cols == 2:
            if x[2] == 0:
                return (2, 3)
            else:
                return (2, 3)
        elif cols == 3:
            return (2, 3, 4)

    def test_max(
        self,
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
        _fixt: tuple[dict, tuple[int, int, int, int]],
    ) -> None:
        df1, _ = _fixt
        df2, _ = fixt
        assert max(df1["y"]) == df2.max(0)

    def test_min(
        self,
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
        _fixt: tuple[dict, tuple[int, int, int, int]],
    ) -> None:
        df1, _ = _fixt
        df2, _ = fixt
        assert min(df1["y"]) == df2.min(0)

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
        gb = df.groupby(0, names)
        ngroups = self.get_unique_groups(x)

        assert len(gb) == ngroups

    def test_contains(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        df, x = fixt
        names = self.get_column_names(x)
        for i in names:
            assert i in df

    def test_get_container_type(
        self, fixt: tuple[DataHolder, tuple[int, int, int, int]]
    ):
        data, _ = fixt

        assert data._get_container_type() == "numpy"

    def test_get_numpy_index(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        data, _ = fixt
        indices = np.arange(5)
        temp = data._numpy_index((indices, 0))
        assert list(temp) == list(data[0][indices])

    def test_getitem(
        self,
        _fixt: tuple[dict, tuple[int, int, int, int]],
        fixt: tuple[DataHolder, tuple[int, int, int, int]],
    ):
        data1, _ = _fixt
        data2, _ = fixt
        indices = np.arange(5)

        temp = data2[indices, 0]

        assert list(temp) == [data1["y"][i] for i in indices]
