from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, NamedTuple

import numpy as np


@dataclass
class PlotData:
    group_labels: list[str]
    zorder: list[int]
    direction: Literal["vertical", "horizontal"]


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
    facet_index: None | list[int] = None
    stacked: bool = False
    plot_type: str = "rectangle"


@dataclass
class LinePlotData(PlotData):
    x_data: list
    y_data: list
    error_data: list
    facet_index: list[int]
    linecolor: list[str | None] | None = None
    linewidth: list[float | None] | None = None
    linestyle: list[str | None] | None = None
    linealpha: float | None = None
    marker: list[str | None] | None = None
    markersize: float | None = None
    markerfacecolor: list[str | None] | None = None
    markeredgecolor: list[str | None] | None = None
    fill_between: bool = False
    fillcolor: list[str | None] | None = None
    fillalpha: float | None = None
    fill_under: bool = False
    plot_type: str = "line"


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
    style: str
    plot_type: str = "violin"


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
CIFunc: TypeAlias = Literal["ci", "pi", "none"]
HistTypes: TypeAlias  = Literal["bar", "step", "stack", "fill"]


class MarkerLine(NamedTuple):
    marker: str
    markestyle: str
    markersize: float
    markerfacecolor: ColorParameters | tuple[str, str] = None
    markerfacedge: ColorParameters | tuple[str, str] = None


class FillBetweenLine(NamedTuple):
    fill_alpha: AlphaRange = 0.5
    fillcolor: ColorParameters | tuple[str, str] = "glaseby_category10"


class FillUnderLine(NamedTuple):
    fill_alpha: AlphaRange = 1.0
    fillcolor: ColorParameters | tuple[str, str] = "glaseby_category10"
