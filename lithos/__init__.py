from importlib.metadata import version, PackageNotFoundError

from .plotting import CategoricalPlot, LinePlot
from .stats import *
from .types.plot_input import Group, Subgroup, UniqueGroups

try:
    __version__ = version("lithos")
except PackageNotFoundError:
    pass
