from typing import NamedTuple, TypeAlias

from .basic_types import AlphaRange, Error, Agg


class Group(tuple):
    def __new__(cls, *items):
        return super().__new__(cls, items)

    def _asdict(self):
        return {"group": tuple(self)}


class Subgroup(tuple):
    def __new__(cls, *items):
        return super().__new__(cls, items)

        def _asdict(self):
            return {"subgroup": tuple(self)}


class UniqueGroups(tuple):
    def __new__(cls, *items):
        return super().__new__(cls, items)

    def _asdict(self):
        return {"unique_groups": (self)}


ColorParameters: TypeAlias = (
    str | dict[str | int, str] | Group | Subgroup | UniqueGroups | None
)


# For new API for density, scatter, percent, aggregate functions
# density: kde, hist -> line
# aggregate: jitter, summary, box, bar, line (paired), marker
# scatter: marker
# percent: hist, kde, categorical -> stacked or percent


class Line(NamedTuple):
    color: ColorParameters
    style: str = "-"
    width: float | int = 1.5
    alpha: AlphaRange = 1.0
    aggregate: Agg | None = None
    type: str = "line"


class Marker(NamedTuple):
    color: ColorParameters = "glasbey_category10"
    edgecolor: ColorParameters = "black"
    style: str = "o"
    size: float | int | str = 5
    alpha: AlphaRange = 1.0
    edgealpha: AlphaRange = 1.0
    edgewidth: float | int = 1.0
    aggregate: Agg | None = None
    type: str = "marker"


class ErrorBand(NamedTuple):
    color: ColorParameters = "glasbey_category10"
    alpha: AlphaRange = 0.5
    edgecolor: ColorParameters = "none"
    edgealpha: AlphaRange = 1.0
    error: Error = "sem"
    type: str = "errorband"


class ErrorBar(NamedTuple):
    color: ColorParameters = "glasbey_category10"
    alpha: AlphaRange = 1.0
    error: Error = "sem"
    type: str = "errorband"


class FillUnder(NamedTuple):
    color: ColorParameters = "glasbey_category10"
    alpha: AlphaRange = 0.5
    type: str = "fillunder"


class Bar(NamedTuple):
    linecolor: ColorParameters = "glasbey_category10"
    fillcolor: ColorParameters = "glasbey_category10"
    linealpha: AlphaRange = 1.0
    fillalpha: AlphaRange = 0.5
    aggregate: Agg | None = "mean"
    type: str = "bar"
