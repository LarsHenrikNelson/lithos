"""Element dataclasses for the new plot API.

Elements describe *how* processed geometry should be rendered on the final
plot. They carry no computation logic and no knowledge of the backend; each
element is a plain-formatting spec that can be serialized to a dict (and thus
into metadata or passed straight to a non-matplotlib renderer such as a JS
frontend).

Aggregation and error computation live in the transforms (see
``lithos.plotting.transforms``), never in elements. Elements such as
``ErrorBand``, and ``ErrorBar`` merely render the error/quantile
geometry produced by a transform.

Fills are exclusively :class:`Fill` (including hatching): it is used for
density/hist fill-under curves and bar faces. :class:`Bar` carries only
outline/structure styling, and :class:`ErrorBand` is strictly the band around
a transform's center +/- error bounds.

Field names deliberately match the legacy method keyword arguments so the
serialized spec can be consumed by the existing processing helpers
(``preprocess_args`` recognizes any key containing ``color``, plus the
``marker`` / ``linestyle`` / ``hatch`` / ``barwidth`` keys) with no renaming
round trip.
"""

from dataclasses import asdict, dataclass

from ..types.basic_types import CapStyle
from ..types.plot_input import AlphaRange, ColorParameters

__all__ = [
    "Annotation",
    "Bar",
    "Element",
    "ErrorBand",
    "ErrorBar",
    "Fill",
    "Line",
    "Marker",
    "Significance",
]


@dataclass
class Element:
    """Base class for all formatting elements.

    Subclasses add their own fields; all fields must keep a default so that
    element specs can be built positionally and stay JSON-serializable.
    """

    type: str = "element"
    zorder: int | float | None = None

    def to_spec(self) -> dict:
        """Return a plain nested dict suitable for metadata/JSON export."""
        return asdict(self)


@dataclass
class Line(Element):
    """A line joining a sequence of (x, y) points per group.

    Also serves as the connector for paired plots; no separate connector
    element exists.
    """

    type: str = "line"
    linecolor: ColorParameters = "glasbey_category10"
    linestyle: str = "-"
    linewidth: float | int = 2
    linealpha: AlphaRange = 1.0


@dataclass
class Marker(Element):
    """A marker placed at each (x, y) point per group (scatter/jitter)."""

    type: str = "marker"
    marker: str | dict = "o"
    markercolor: ColorParameters = "glasbey_category10"
    edgecolor: ColorParameters = "white"
    markeredgewidth: float = 1.0
    markersize: float | str | tuple = 5.0
    alpha: AlphaRange = 1.0
    edge_alpha: AlphaRange = 1.0


@dataclass
class Bar(Element):
    """A rectangle outline (histogram bar / categorical bar) per group.

    Fill styling (facecolor, fillalpha, hatch) comes from a separate
    :class:`Fill` element; ``Bar`` only controls the outline and geometry.
    """

    type: str = "bar"
    edgecolor: ColorParameters = "glasbey_category10"
    barwidth: float = 0.9
    linewidth: float = 1
    edge_alpha: AlphaRange = 1.0


@dataclass
class Fill(Element):
    """Filled region - the single source of fill styling.

    Used for any filled geometry: density/hist fill-under curves and bar
    faces. Pair with :class:`Bar` for outlines and with :class:`Line` for density
    outlines. Hatching is also fill styling.
    """

    type: str = "fill"
    fillcolor: ColorParameters = "glasbey_category10"
    fillalpha: AlphaRange = 0.5
    hatch: str | None = None
    edgecolor: ColorParameters = "none"
    edgealpha: AlphaRange = 1.0


@dataclass
class ErrorBand(Element):
    """Shaded band between a transform's ``center +/- error`` bounds.

    Not used for general fills - those are :class:`Fill`.
    """

    type: str = "errorband"
    fillcolor: ColorParameters = "glasbey_category10"
    fillalpha: AlphaRange = 0.5
    edgecolor: ColorParameters = "none"
    edgealpha: AlphaRange = 1.0
    linewidth: float = 1.0


@dataclass
class ErrorBar(Element):
    """Capped error bars around an aggregate center per group."""

    type: str = "errorbar"
    linecolor: ColorParameters = "glasbey_category10"
    linealpha: AlphaRange = 1.0
    linewidth: float = 2.0
    capsize: float = 5.0
    capstyle: CapStyle = "butt"


@dataclass
class Annotation(Element):
    """Free-floating text placed on the axes."""

    type: str = "annotation"
    text: str = ""
    x: float | None = None
    y: float | None = None
    fontsize: float = 12
    color: str = "black"
    ha: str = "center"
    va: str = "center"
    rotation: float = 0


@dataclass
class Significance(Element):
    """Statistical significance bracket (GraphPad asterisks style)."""

    type: str = "significance"
    text: str = "*"
    x1: float | None = None
    x2: float | None = None
    y: float | None = None
    linecolor: str = "black"
    linewidth: float = 1.5
    fontsize: float = 12
    capsize: float = 5
