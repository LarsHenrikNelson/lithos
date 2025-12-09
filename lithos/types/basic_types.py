from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Callable, Literal, NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from .plot_input import Group, Subgroup, UniqueGroups

Direction: TypeAlias = Literal["vertical", "horizontal"]

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
Levels: TypeAlias = tuple


ProcessingOutput: TypeAlias = (
    None
    | str
    | dict
    | tuple[int | float, int | float, int | float]
    | tuple[int | float, int | float, int | float, int | float]
    | tuple[tuple[int | float, int | float, int | float] | str, int | float]
    | tuple[tuple[int | float, int | float, int | float, int | float], int | float]
)

InputData: TypeAlias = (
    dict[str | int, list[int | float | str] | np.ndarray] | pd.DataFrame | np.ndarray
)

Grouping: TypeAlias = list[str | int | float] | tuple[str | int | float] | Group | None
Subgrouping: TypeAlias = (
    list[str | int | float] | tuple[str | int | float] | Subgroup | None
)
UniqueGrouping: TypeAlias = (
    list[str | int | float] | tuple[str | int | float] | UniqueGroups | None
)

NBins: TypeAlias = (
    int | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
)

AlphaRange: TypeAlias = Annotated[float, "Value between 0.0 and 1.0"]
CountPlotTypes: TypeAlias = Literal["percent", "count"]

TransformFuncs: TypeAlias = Literal[
    "log10", "log2", "ln", "inverse", "ninverse", "sqrt"
]
AggFuncs: TypeAlias = Literal[
    "mean", "periodic_mean", "nanmean", "median", "nanmedian", "gmean", "hmean", "count"
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
Error: TypeAlias = ErrorFuncs | Callable | None
Agg: TypeAlias = AggFuncs | Callable
Transform: TypeAlias = TransformFuncs | None
BinType: TypeAlias = Literal["density", "percent"]
CapStyle: TypeAlias = Literal["butt", "round", "projecting"]
SavePath: TypeAlias = str | Path | BytesIO | StringIO
FitFunc: TypeAlias = Callable | Literal["linear", "sine", "polynomial"]
CIFunc: TypeAlias = Literal["ci", "pi", "none"]
HistType: TypeAlias = Literal["bar", "step", "stack", "fill"]
JitterType: TypeAlias = Literal["fill", "dist"]
HistBinLimits: TypeAlias = tuple[float, float] | Literal["common"] | None
HistStat: TypeAlias = Literal["density", "probability", "count"]
