import pytest

from lithos.plotting.processing import (
    BaseProcessor,
)


@pytest.mark.parametrize(
    "test_groupings",
    [
        "one_grouping",
        "one_grouping_with_unique_ids",
        "two_grouping",
        "two_grouping_with_unique_ids",
    ],
)
class BaseProcessorTestClass:
    @pytest.fixture
    def _fixt(self, test_groupings, request) -> tuple[dict, tuple[int, int, int, int]]:
        data, x = request.getfixturevalue(test_groupings)
        return data, x

    @pytest.mark.parametrize(
        "markers, hatches",
        [
            ("o", "X", "^", "s", "d"),
            ("/", "o", "-", "*", "+"),
        ],
    )

    def test_zorder(self, markers, hatches):
        # Test the _set_zorder method of the LineProcessor class
        line_processor = BaseProcessor(markers=markers, hatches=hatches)
        zorder_dict = line_processor._set_zorder()
