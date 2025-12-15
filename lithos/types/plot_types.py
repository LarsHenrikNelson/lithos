from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .basic_types import Direction


@dataclass
class PlotData:
    group_labels: list[str]
    zorder: list[int]
    direction: Direction


@dataclass
class RectanglePlotData(PlotData):
    heights: list[float]
    bottoms: list[float]
    bins: list[float]
    binwidths: list
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
class MarkerLinePlotData(PlotData):
    x_data: list
    y_data: list
    facet_index: list[int]
    linecolor: list[str | None] | None = None
    linewidth: list[float | None] | None = None
    linestyle: list[str | None] | None = None
    linealpha: float | None = None
    marker: list[str | None] | None = None
    markersize: list[float] | None = None
    markerfacecolor: list[str | None] | None = None
    markeredgecolor: list[str | None] | None = None
    plot_type: str = "marker_line"


@dataclass
class JitterPlotData(PlotData):
    x_data: list[np.ndarray]
    y_data: list[np.ndarray]
    marker: list[str]
    markerfacecolor: list[str]
    markeredgecolor: list[str]
    markeredgewidth: list[float] | float | str
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
    edge_alpha: float
    fliers: bool | str
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


PlotTypes: TypeAlias = (
    ViolinPlotData
    | BoxPlotData
    | MarkerLinePlotData
    | SummaryPlotData
    | JitterPlotData
    | ScatterPlotData
    | LinePlotData
    | RectanglePlotData
)
