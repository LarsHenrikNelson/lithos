from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, NamedTuple, TypeAlias

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

NBins: TypeAlias = (
    int | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
)

CountPlotTypes: TypeAlias = Literal["percent", "count"]

TransformFuncs: TypeAlias = Literal[
    "log10", "log2", "ln", "inverse", "ninverse", "sqrt"
]

Transform: TypeAlias = TransformFuncs | Callable | None
BinType: TypeAlias = Literal["density", "percent"]
CapStyle: TypeAlias = Literal["butt", "round", "projecting"]
SavePath: TypeAlias = str | Path | BytesIO | StringIO
FitFunc: TypeAlias = Callable | Literal["linear", "sine", "polynomial"]
CIFunc: TypeAlias = Literal["ci", "pi", "none"]
HistType: TypeAlias = Literal["bar", "step", "stack", "fill"]
JitterType: TypeAlias = Literal["fill", "dist"]
HistBinLimits: TypeAlias = tuple[float, float] | Literal["common"] | None
HistStat: TypeAlias = Literal["density", "probability", "count"]
CatLabelTypes: TypeAlias = Literal["style1", "style2", "style3"]
CategoricalLabels: TypeAlias = CatLabelTypes | dict[CatLabelTypes, Any]
