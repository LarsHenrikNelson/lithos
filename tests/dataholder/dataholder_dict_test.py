import numpy as np
import pytest

from .dataholder_test_class import StringColumnsDataHolder
from lithos.utils import DataHolder


class TestDataHolderDict(StringColumnsDataHolder):
    @pytest.fixture
    def fixt(
        self, _fixt: tuple[dict, tuple[int, int, int, int]]
    ) -> tuple[DataHolder, tuple[int, int, int, int]]:
        data, x = _fixt
        data = DataHolder(data)
        return data, x

    def test_get_container_type(
        self, fixt: tuple[DataHolder, tuple[int, int, int, int]]
    ):
        data, _ = fixt

        assert data._get_container_type() == "dict"

    def test_get_dict_index(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        data, _ = fixt
        indices = np.arange(5)
        temp = data._dict_index((indices, "y"))
        assert list(temp) == list(data["y"][indices])
