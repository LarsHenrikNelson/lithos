import pytest
import shutil

from lithos.utils import metadata_utils, create_synthetic_data

pytest_plugins = "pytester"


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        hdir = metadata_utils.home_dir()

        shutil.rmtree(hdir)

    request.addfinalizer(remove_test_dir)


@pytest.fixture(scope="session")
def one_grouping():
    data = create_synthetic_data(1, 0, 0, 30)
    return data, (1, 0, 0, 30)


@pytest.fixture(scope="session")
def one_grouping_with_unique_ids():
    data = create_synthetic_data(1, 0, 3, 30)
    return data, (1, 0, 3, 30)


@pytest.fixture(scope="session")
def two_grouping():
    data = create_synthetic_data(2, 2, 0, 30)
    return data, (2, 2, 0, 30)


@pytest.fixture(scope="session")
def two_grouping_with_unique_ids():
    data = create_synthetic_data(2, 2, 3, 30)
    return data, (2, 2, 3, 30)
