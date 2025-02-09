from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Literal

import numpy as np


@dataclass
class RectanglePlotData:
    plot_type: str = "rectangle"
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


@dataclass
class LinePlotData:
    plot_type: str = "line"
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


@dataclass
class ScatterPlotData:
    plot_type: str = "scatter"
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    alpha: float
    edge_alpha: float


@dataclass
class SummaryPlotData:
    plot_type: str = "summary"
    x_data: list
    y_data: list
    error_data: list
    widths: list
    colors: list
    linewidth: float
    alpha: float
    capstyle: str
    capsize: float


@dataclass
class BoxData:
    plot_type: str = "box"
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


@dataclass
class ViolinPlotData:
    plot_type: str = "violin"
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


type Kernels = Literal[
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

type BW = float | Literal["ISJ", "silverman", "scott"]
type KDEType = Literal["fft", "tree"]
type Levels = str | int | float


@dataclass
class ValueRange:
    lo: float
    hi: float


type AlphaRange = Annotated[float, ValueRange(0.0, 1.0)]
type ColorParameters = str | dict[str, str] | None
type CountPlotTypes = Literal["percent", "count"]
type TransformFuncs = Literal["log10", "log2", "ln", "inverse", "ninverse", "sqrt"]
type AggFuncs = Literal[
    "mean", "periodic_mean", "nanmean", "median", "nanmedian", "gmean", "hmean"
]
type ErrorFuncs = Literal[
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
type Error = ErrorFuncs | callable | None
type Agg = AggFuncs | callable
type Transform = TransformFuncs | None
type BinType = Literal["density", "percent"]
type CapStyle = Literal["butt", "round", "projecting"]
type SavePath = str | Path | BytesIO | StringIO
