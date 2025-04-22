from importlib.metadata import version, PackageNotFoundError

from .plotting import CategoricalPlot, LinePlot
from .stats import *

try:
    __version__ = version("lithos")
except PackageNotFoundError:
    pass
