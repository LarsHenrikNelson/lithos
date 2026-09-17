import json

import pytest

from lithos.plotting.elements import (
    Annotation,
    Bar,
    Element,
    ErrorBand,
    ErrorBar,
    Fill,
    Line,
    Marker,
    Significance,
)

ALL_ELEMENTS = [
    (Line, "line"),
    (Marker, "marker"),
    (Bar, "bar"),
    (Fill, "fill"),
    (ErrorBand, "errorband"),
    (ErrorBar, "errorbar"),
    (Annotation, "annotation"),
    (Significance, "significance"),
]


@pytest.mark.parametrize("cls,type_name", ALL_ELEMENTS)
def test_element_is_subclass_and_has_type(cls, type_name):
    elem = cls()
    assert isinstance(elem, Element)
    assert elem.type == type_name


@pytest.mark.parametrize("cls,_", ALL_ELEMENTS)
def test_element_to_spec_is_json_serializable(cls, _):
    elem = cls()
    spec = elem.to_spec()
    # round trip through JSON keeps the structure intact
    loaded = json.loads(json.dumps(spec))
    assert loaded["type"] == spec["type"]
    assert isinstance(loaded, dict)


@pytest.mark.parametrize("cls,_", ALL_ELEMENTS)
def test_element_zorder_default(cls, _):
    assert cls().zorder is None


@pytest.mark.parametrize("cls,_", ALL_ELEMENTS)
def test_element_to_spec_includes_zorder(cls, _):
    assert cls().to_spec()["zorder"] is None


def test_line_has_no_fill_fields():
    # fills are handled exclusively by Fill/ErrorBand in the new API
    spec = Line().to_spec()
    assert "fillcolor" not in spec
    assert "fillalpha" not in spec
    assert "fill_between" not in spec


def test_bar_has_no_fill_fields():
    # fill styling (incl. hatch) lives exclusively on Fill
    spec = Bar().to_spec()
    assert "facecolor" not in spec
    assert "fillcolor" not in spec
    assert "fillalpha" not in spec
    assert "hatch" not in spec
    assert "alpha" not in spec


def test_errorband_has_no_fill_to_baseline():
    # ErrorBand is strictly the center +/- error band; general fills are Fill
    assert "fill_to_baseline" not in ErrorBand().to_spec()


def test_fill_fields_and_hatch():
    spec = Fill().to_spec()
    assert spec["fillcolor"] == "glasbey_category10"
    assert spec["fillalpha"] == 0.5
    assert spec["hatch"] is None
    spec = Fill(fillcolor="red", fillalpha=0.3, hatch="//").to_spec()
    assert spec["fillcolor"] == "red"
    assert spec["fillalpha"] == 0.3
    assert spec["hatch"] == "//"


def test_element_to_spec_includes_customized_fields():
    spec = Marker(marker="X", markersize=8, zorder=3).to_spec()
    assert spec["marker"] == "X"
    assert spec["markersize"] == 8
    assert spec["zorder"] == 3


def test_significance_fields():
    spec = Significance(text="**", x1=0.9, x2=2.1, y=5.0).to_spec()
    assert spec["text"] == "**"
    assert spec["x1"] == 0.9
    assert spec["x2"] == 2.1
    assert spec["y"] == 5.0
