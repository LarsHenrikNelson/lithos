from importlib.metadata import version, PackageNotFoundError

from .plotting import CategoricalPlot, LinePlot
from .stats import *
from .plotting.types import Group, Subgroup, UniqueGroups

try:
    __version__ = version("lithos")
except PackageNotFoundError:
    pass
