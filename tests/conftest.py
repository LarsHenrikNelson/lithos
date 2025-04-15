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
    data = create_synthetic_data(3, 0, 0, 30)
    return data, (3, 0, 0, 30)


@pytest.fixture(scope="session")
def one_grouping_with_unique_ids():
    data = create_synthetic_data(2, 0, 3, 30)
    return data, (2, 0, 3, 30)


@pytest.fixture(scope="session")
def two_grouping():
    data = create_synthetic_data(2, 2, 0, 30)
    return data, (2, 2, 0, 30)


@pytest.fixture(scope="session")
def two_grouping_with_unique_ids():
    output = (2, 3, 3, 30)
    data = create_synthetic_data(*output)
    return data, output


@pytest.fixture(scope="session")
def metadata_core_one_categorical():
    metadata = {
        "grouping": {
            "group": "grouping_1",
            "subgroup": None,
            "group_order": None,
            "subgroup_order": None,
            "group_spacing": 0.9,
        }
    }
    return metadata


@pytest.fixture(scope="session")
def metadata_core_two_categorical():
    metadata = {
        "grouping": {
            "group": "grouping_1",
            "subgroup": "grouping_2",
            "group_order": None,
            "subgroup_order": None,
            "group_spacing": 0.9,
        }
    }
    return metadata


@pytest.fixture(scope="session")
def y_only():
    metadata = {
        "data": {
            "y": "y",
            "x": None,
            "ylabel": "test",
            "xlabel": "",
            "title": "Test",
            "figure_title": "",
        }
    }
    return metadata


@pytest.fixture(scope="session")
def x_and_y():
    metadata = {
        "data": {
            "y": "y",
            "x": "x",
            "ylabel": "test",
            "xlabel": "",
            "title": "Test",
            "figure_title": "",
        }
    }
    return metadata


@pytest.fixture(scope="session")
def metadata_format():
    metadata = {
        "format": {
            "labels": {
                "labelsize": 22,
                "titlesize": 22,
                "font": "DejaVu Sans",
                "ticklabel_size": 20,
                "title_fontweight": "bold",
                "label_fontweight": "normal",
                "tick_fontweight": "light",
                "xlabel_rotation": "vertical",
                "ylabel_rotation": "vertical",
                "xtick_rotation": "horizontal",
                "ytick_rotation": "horizontal",
            },
            "axis": {
                "yscale": "linear",
                "xscale": "linear",
                "ylim": (None, None),
                "xlim": (None, None),
                "yaxis_lim": None,
                "xaxis_lim": None,
                "ydecimals": None,
                "xdecimals": None,
                "xunits": None,
                "yunits": None,
                "xformat": "f",
                "yformat": "f",
            },
            "axis_format": {
                "tickwidth": 0.5,
                "ticklength": 2,
                "linewidth": {"left": 0.5, "bottom": 0.5, "top": 0, "right": 0},
                "minor_tickwidth": 1.5,
                "minor_ticklength": 2.5,
                "yminorticks": 0,
                "xminorticks": 0,
                "xsteps": (5, 0, 5),
                "ysteps": (5, 0, 5),
                "style": "lithos",
                "truncate_xaxis": False,
                "truncate_yaxis": False,
            },
            "figure": {
                "gridspec_kw": None,
                "margins": 0.05,
                "aspect": 1.0,
                "figsize": None,
                "nrows": None,
                "ncols": None,
                "projection": "rectilinear",
            },
            "grid": {
                "ygrid": 0,
                "xgrid": 0,
                "yminor_grid": 0,
                "xminor_grid": 0,
                "linestyle": "solid",
                "minor_linestyle": "solid",
            },
        },
        "transforms": {
            "ytransform": None,
            "back_transform_yticks": False,
            "xtransform": None,
            "back_transform_xticks": False,
        },
    }
    return metadata
