from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, NamedTuple

import numpy as np

Kernels: TypeAlias = Literal[
    "gaussian",
    "exponential",
    "box",
    "tri",
    "epa",
    "biweight",
    "triweight",
    "tricube",
    "cosine",
]

BW: TypeAlias = float | Literal["ISJ", "silverman", "scott"]
KDEType: TypeAlias = Literal["fft", "tree"]
Levels: TypeAlias = str | int | float


@dataclass
class ValueRange:
    lo: float
    hi: float


class Group(NamedTuple):
    group: list


class Subgroup(NamedTuple):
    subgroup: list


class UniqueGroups(NamedTuple):
    unique_groups: list


AlphaRange: TypeAlias = Annotated[float, ValueRange(0.0, 1.0)]
CountPlotTypes: TypeAlias = Literal["percent", "count"]
ColorParameters: TypeAlias = str | dict[str, str] | None
TransformFuncs: TypeAlias = Literal[
    "log10", "log2", "ln", "inverse", "ninverse", "sqrt"
]
AggFuncs: TypeAlias = Literal[
    "mean", "periodic_mean", "nanmean", "median", "nanmedian", "gmean", "hmean"
]
ErrorFuncs: TypeAlias = Literal[
    "sem",
    "ci",
    "periodic_std",
    "periodic_sem",
    "std",
    "nanstd",
    "var",
    "nanvar",
    "mad",
    "gstd",
]
Error: TypeAlias = ErrorFuncs | callable | None
Agg: TypeAlias = AggFuncs | callable
Transform: TypeAlias = TransformFuncs | None
BinType: TypeAlias = Literal["density", "percent"]
CapStyle: TypeAlias = Literal["butt", "round", "projecting"]
SavePath: TypeAlias = str | Path | BytesIO | StringIO
FitFunc: TypeAlias = callable | Literal["linear", "sine", "polynomial"]


class Line(NamedTuple):
    plot_type: str = "line"
    linecolor: ColorParameters = "glasbey_category10"
    linewidth: float = 1.0
    linestyle: str = "-"
    linealpha: float = 1.0


class MarkerLine(NamedTuple):
    plot_type: str = "marker"
    linecolor: ColorParameters = "glasbey_category10"
    linewidth: float = 1.0
    linestyle: str = "-"
    linealpha: float = 1.0
    marker: str = "o"
    markersize: float = 1
    markerfacecolor: ColorParameters | tuple[str, str] = "glasbey_category10"
    markeredgecolor: ColorParameters | tuple[str, str] = "glasbey_category10"


class FillBetweenLine(NamedTuple):
    plot_type: str = "fill_between"
    linecolor: ColorParameters = "glasbey_category10"
    linewidth: float = 1.0
    linestyle: str = "-"
    linealpha: float = 1.0
    fill_alpha: AlphaRange = 0.5
    fillcolor: ColorParameters | tuple[str, str] = "glasbey_category10"


class FillUnderLine(NamedTuple):
    plot_type: str = "fill_under"
    linecolor: ColorParameters = "glasbey_category10"
    linewidth: float = 1.0
    linestyle: str = "-"
    linealpha: float = 1.0
    linewidth: float = 1.0
    linestyle: str = "-"
    linealpha: float = 1.0
    fill_alpha: AlphaRange = 1.0
    fillcolor: ColorParameters | tuple[str, str] = "glasbey_category10"


@dataclass
class PlotData:
    group_labels: list[str]
    zorder: list[int]


@dataclass
class RectanglePlotData(PlotData):
    heights: list[float]
    bottoms: list[float]
    bins: list[float]
    binwidths: list[float]
    fillcolors: list[str]
    edgecolors: list[str]
    fill_alpha: float
    edge_alpha: float
    hatches: list[str]
    linewidth: float
    axis: Literal["x", "y"]
    facet_index: None | list[int] = None
    plot_type: str = "rectangle"


FillType: TypeAlias = Literal[None, "fill_under", "fill_between"]


@dataclass
class LinePlotData(PlotData):
    facet_index: list[int]
    x_data: list
    y_data: list
    linecolor: list[str | None] | None = None
    linestyle: list[str | None] | None = None
    linealpha: float | None = None
    linewidth: list[float | None] | None = None
    plot_type: str = "line"
    agg_func: Agg | None = None
    direction: Literal["xy"] = "y"


@dataclass
class ErrorLinePlotData(LinePlotData):
    error_data: list = None
    err_func: Error = "sem"


@dataclass
class MarkerLinePlotData(ErrorLinePlotData):
    marker: list[str | None] | None = None
    markerfacecolor: list[str | None] | None = None
    markeredgecolor: list[str | None] | None = None
    markersize: float | None = None
    plot_type: str = "marker_line"


@dataclass
class FillBetweenPlotData(ErrorLinePlotData):
    fillcolor: list[str | None] | None = None
    fill_alpha: float | None = None
    plot_type: str = "fill_between_line"


@dataclass
class FillUnderPlotData(LinePlotData):
    fillcolor: list[str | None] | None = None
    fill_alpha: float | None = None
    plot_type: str = "fill_under_line"


@dataclass
class JitterPlotData(PlotData):
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    alpha: float
    edge_alpha: float
    plot_type: str = "jitter"


@dataclass
class ScatterPlotData(PlotData):
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    alpha: float
    linewidth: float | int
    edge_alpha: float
    facet_index: list[int]
    plot_type: str = "scatter"


@dataclass
class SummaryPlotData(PlotData):
    x_data: list
    y_data: list
    error_data: list
    widths: list
    colors: list
    linewidth: float
    alpha: float
    capstyle: str
    capsize: float
    plot_type: str = "summary"


@dataclass
class BoxPlotData(PlotData):
    x_data: list
    y_data: list
    facecolors: list[str]
    edgecolors: list[str]
    alpha: float
    linealpha: float
    fliers: bool
    linewidth: float
    width: float
    show_ci: bool
    showmeans: bool
    plot_type: str = "box"


@dataclass
class ViolinPlotData(PlotData):
    x_data: list
    y_data: list
    location: list[float]
    facecolors: list[str]
    edgecolors: list[str]
    alpha: float
    edge_alpha: float
    linewidth: float
    plot_type: str = "violin"
