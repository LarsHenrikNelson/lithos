from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import numpy as np


@dataclass
class RectanglePlotData:
    width: float
    height: float
    tops: list[float]
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
    plot_type: str = "rectangle"


@dataclass
class LinePlotData:
    x_data: list
    y_data: list
    error_data: list
    facet_index: list[int]
    marker: list[str | None] | None = None
    linecolor: list[str | None] | None = None
    linewidth: list[float | None] | None = None
    linestyle: list[str | None] | None = None
    markerfacecolor: list[str | None] | None = None
    markeredgecolor: list[str | None] | None = None
    fill_between: bool = False
    fb_direction: Literal["x" "y"] = "y"
    markersize: float | None = None
    fillalpha: float | None = None
    linealpha: float | None = None
    plot_type: str = "line"


@dataclass
class ScatterPlotData:
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    alpha: float
    edge_alpha: float
    plot_type: str = "scatter"


@dataclass
class SummaryPlotData:
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
class BoxPlotData:
    x_data: list
    y_data: list
    facecolors: list[str]
    edgecolors: list[str]
    alpha: float
    line_alpha: float
    fliers: bool
    linewidth: float
    width: float
    show_ci: bool
    showmeans: bool
    plot_type: str = "box"


@dataclass
class ViolinPlotData:
    x_data: list
    y_data: list
    facecolors: list[str]
    edgecolors: list[str]
    alpha: float
    edge_alpha: float
    linewidth: float
    width: list[float]
    showmeans: bool
    showmedians: bool
    showextrema: bool
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


AlphaRange: TypeAlias = Annotated[float, ValueRange(0.0, 1.0)]
ColorParameters: TypeAlias = str | dict[str, str] | None
CountPlotTypes: TypeAlias = Literal["percent", "count"]
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
