from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import numpy as np


class RectanglePlotter(TypedDict):
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


class LinePlotter(TypedDict):
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


class ScatterPlotter(TypedDict):
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markersize: list[float]
    alpha: float
    edge_alpha: float


class SummaryPlotter(TypedDict):
    x_data: list
    y_data: list
    error_data: list
    widths: list
    colors: list
    linewidth: float
    alpha: float
    capstyle: str
    capsize: float


class BoxPlotter(TypedDict):
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


class ViolinPlotter(TypedDict):
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


class HistPlotter(TypedDict):
    x_data: list
    y_data: list
    facecolors: list[str]
    edgecolors: list[str]
    alpha: float
    facet_index: list[int]


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
