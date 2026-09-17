from importlib.metadata import PackageNotFoundError, version

from .plotting import CategoricalPlot, LinePlot
from .plotting.elements import (
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
from .plotting.transforms import (
    Aggregate,
    Density,
    Fit,
    Identity,
    Summary,
    Transform,
)
from .stats import *
from .types.plot_input import Group, Subgroup, UniqueGroups

try:
    __version__ = version("lithos")
except PackageNotFoundError:
    pass
