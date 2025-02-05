import pandas as pd
import numpy as np
import pytest

from .dataholder_test_class import StringColumnsDataHolder
from lithos.utils import DataHolder


class TestDataHolderPandas(StringColumnsDataHolder):
    @pytest.fixture
    def fixt(
        self, _fixt: tuple[dict, tuple[int, int, int, int]]
    ) -> tuple[DataHolder, tuple[int, int, int, int]]:
        data, x = _fixt
        data = DataHolder(pd.DataFrame(data))
        return data, x

    def test_get_container_type(
        self, fixt: tuple[DataHolder, tuple[int, int, int, int]]
    ):
        data, _ = fixt

        assert data._get_container_type() == "pandas"

    def test_get_pandas_index(self, fixt: tuple[DataHolder, tuple[int, int, int, int]]):
        data, _ = fixt
        indices = np.arange(5)
        temp = data._pandas_index((indices, "y"))
        assert list(temp) == list(data["y"][indices])
