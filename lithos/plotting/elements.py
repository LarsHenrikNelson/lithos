"""Element dataclasses for the new plot API.

Elements describe *how* processed geometry should be rendered on the final
plot. They carry no computation logic and no knowledge of the backend; each
element is a plain-formatting spec that can be serialized to a dict (and thus
into metadata or passed straight to a non-matplotlib renderer such as a JS
frontend).

Aggregation and error computation live in the transforms (see
``lithos.plotting.transforms``), never in elements. Elements such as
``ErrorBand``, and ``ErrorBar`` merely render the error/quantile geometry
produced by a transform.

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
from typing import ClassVar

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
    "SummaryLine",
]


@dataclass
class Element:
    """Base class for all formatting elements.

        Subclasses add their own fields; all fields must keep a default so that
        element specs can be built positionally and stay JSON-serializable.
        ``type`` is a class variable, not a field: it identifies the element in
        serialized specs but is not part of ``__init__`` (the class name already
    carries that information).

        Attributes:
            zorder (int | float | None): Explicit draw order for the rendered artists.
                ``None`` falls back to the plotter default (layers stack in ``.add()``
                order).
    """

    type: ClassVar[str] = "element"
    zorder: int | float | None = None

    def to_spec(self) -> dict:
        """Return a plain nested dict suitable for metadata/JSON export.

        Returns:
            dict: The element's serialized spec - the ``type`` class variable
            plus every field from ``asdict(self)``.
        """
        return {"type": self.type, **asdict(self)}


@dataclass
class Line(Element):
    """A line joining a sequence of (x, y) points per group.

    Also serves as the connector for paired plots; no separate connector
    element exists.

    Attributes:
        linecolor (ColorParameters): Line color spec - a color/palette name, a
            dict keyed by group, or a tuple of colors cycled per group.
        linestyle (str): Matplotlib line style (e.g. ``"-"``, ``"--"``).
        linewidth (float | int): Line width in points.
        linealpha (AlphaRange): Line alpha (transparency), between 0 and 1.
    """

    type: ClassVar[str] = "line"
    linecolor: ColorParameters = "glasbey_category10"
    linestyle: str = "-"
    linewidth: float | int = 2
    linealpha: AlphaRange = 1.0


@dataclass
class Marker(Element):
    """A marker placed at each (x, y) point per group (scatter/jitter).

    ``marker`` may be a single symbol, a list cycled over the layer's
    ``unique_id`` values (legacy ``jitteru`` subject markers), or a dict
    keyed by the unique_id value (nested layers) or the group key (flat
    layers).

    Attributes:
        marker (str | dict | list): Marker symbol(s); see above for the
            accepted list/dict forms.
        markercolor (ColorParameters): Marker face color spec.
        edgecolor (ColorParameters): Marker edge color spec.
        markeredgewidth (float): Marker edge width in points.
        markersize (float | str | tuple): Marker size, passed to matplotlib as
            the scatter ``s`` value (squared when numeric).
        alpha (AlphaRange): Marker face alpha (transparency).
        edge_alpha (AlphaRange): Marker edge alpha (transparency).
    """

    type: ClassVar[str] = "marker"
    marker: str | dict | list = "o"
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

    Attributes:
        edgecolor (ColorParameters): Bar outline color spec.
        barwidth (float): Bar width as a fraction of the resolved slot (flat
            layers) or of the subject column (``unique_id``-nested layers).
        linewidth (float): Bar outline width in points.
        edge_alpha (AlphaRange): Bar outline alpha (transparency).
    """

    type: ClassVar[str] = "bar"
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

    Attributes:
        fillcolor (ColorParameters): Fill color spec.
        fillalpha (AlphaRange): Fill alpha (transparency).
        hatch (str | None): Matplotlib hatch pattern; ``None`` for a plain fill.
        edgecolor (ColorParameters): Fill boundary color spec (``"none"`` for no
            boundary).
        edgealpha (AlphaRange): Fill boundary alpha (transparency).
    """

    type: ClassVar[str] = "fill"
    fillcolor: ColorParameters = "glasbey_category10"
    fillalpha: AlphaRange = 0.5
    hatch: str | None = None
    edgecolor: ColorParameters = "none"
    edgealpha: AlphaRange = 1.0


@dataclass
class ErrorBand(Element):
    """Shaded band between a transform's ``center +/- error`` bounds.

    Not used for general fills - those are :class:`Fill`.

    Attributes:
        fillcolor (ColorParameters): Band fill color spec.
        fillalpha (AlphaRange): Band fill alpha (transparency).
        edgecolor (ColorParameters): Band boundary color spec (``"none"`` for no
            boundary).
        edgealpha (AlphaRange): Band boundary alpha (transparency).
        linewidth (float): Band boundary width in points.
    """

    type: ClassVar[str] = "errorband"
    fillcolor: ColorParameters = "glasbey_category10"
    fillalpha: AlphaRange = 0.5
    edgecolor: ColorParameters = "none"
    edgealpha: AlphaRange = 1.0
    linewidth: float = 1.0


@dataclass
class ErrorBar(Element):
    """Capped error bars around an aggregate center per group.

    Attributes:
        linecolor (ColorParameters): Error bar (line and caps) color spec.
        linealpha (AlphaRange): Error bar alpha (transparency).
        linewidth (float): Error bar line width in points.
        capsize (float): Error bar cap width in points.
        capstyle (CapStyle): Cap end shape (``"butt"``, ``"round"``, or
            ``"projecting"``).
    """

    type: ClassVar[str] = "errorbar"
    linecolor: ColorParameters = "glasbey_category10"
    linealpha: AlphaRange = 1.0
    linewidth: float = 2.0
    capsize: float = 5.0
    capstyle: CapStyle = "butt"


@dataclass
class SummaryLine(Element):
    """A short line drawn across an aggregate center.

    The GraphPad-style summary line (legacy ``summary``/``summaryu``). The
    line length is controlled by the layer ``width`` given to ``.add()``: a
    fraction of the resolved slot for flat layers (legacy ``barwidth``), or
    the full subject column for ``unique_id``-nested geometry — so a summary
    line can be drawn wider or narrower than a jitter layer on the same slot
    by giving the layers different ``width`` values. ``capstyle`` selects the
    shape of the line ends (``"round"``/``"butt"``/``"projecting"``).
    Errors render through a separate :class:`ErrorBar` from the same
    transform geometry.

    Attributes:
        linecolor (ColorParameters): Summary line color spec.
        linewidth (float | int): Summary line width in points.
        linealpha (AlphaRange): Summary line alpha (transparency).
        capstyle (CapStyle): Line end shape (``"round"``, ``"butt"``, or
            ``"projecting"``).
    """

    type: ClassVar[str] = "summaryline"
    linecolor: ColorParameters = "glasbey_category10"
    linewidth: float | int = 2
    linealpha: AlphaRange = 1.0
    capstyle: CapStyle = "round"


@dataclass
class Annotation(Element):
    """Free-floating text placed on the axes.

    Attributes:
        text (str): Text to draw.
        x (float | None): X position in axis coordinates.
        y (float | None): Y position in axis coordinates.
        fontsize (float): Text font size in points.
        color (str): Text color.
        ha (str): Horizontal alignment (``"center"``, ``"left"``, ``"right"``).
        va (str): Vertical alignment (``"center"``, ``"top"``, ``"bottom"``).
        rotation (float): Text rotation in degrees.
    """

    type: ClassVar[str] = "annotation"
    text: str = ""
    x: float | None = None
    y: float | None = None
    fontsize: float = 12
    color: str = "black"
    ha: str = "center"
    va: str = "center"
    rotation: float = 0
