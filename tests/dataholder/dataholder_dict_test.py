import numpy as np
import pytest

from lithos.utils import DataHolder

from .dataholder_test_class import StringColumnsDataHolder


class TestDataHolderDict(StringColumnsDataHolder):
    @pytest.fixture
    def fixt(self, _fixt: tuple[dict, tuple[int, int, int, int]]) -> tuple[DataHolder, tuple[int, int, int, int]]:
        data, x = _fixt
        data = DataHolder(data)
        return data, x
