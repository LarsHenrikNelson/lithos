"""High-level verb methods for the spec plot API (the Phase 2.5 sugar layer).

The legacy API's named methods (``scatter``, ``jitter``, ``kde``, ``hist``,
...) grew into flat signatures merging statistical, styling and layout
concerns. The verbs here restore that one-call convenience on the spec
classes while keeping the architecture's separation of concerns: every verb
is a *thin factory* that builds one transform plus role-default elements and
calls ``.add()`` - nothing else. Layers stay the only plot state and
metadata stays pure layer specs, so verbs inherit the pipeline's geometry,
parity and metadata replay unchanged (no metadata version bump, no renderer
change; see ``doc/migration_plan.md``).

Argument contract (three tiers, in precedence order):

1. *Element role overrides win completely.* Each verb exposes its default
   elements as optional role arguments (``marker=``, ``line=``, ``fill=``,
   ``bar=``...). A passed instance replaces that role's default outright -
   intent arguments are not applied to it - and ``False`` drops the role
   from the composition. Dropping every role surfaces ``.add()``'s
   no-element error.
2. *Intent arguments shape only the defaults:* ``color`` fans out to the
   role defaults' primary color fields (``linecolor``/``fillcolor``/
   ``markercolor``; never edge fields), ``alpha`` to their primary alpha
   fields (``Marker.alpha``, ``Fill.fillalpha``, ``Line.linealpha``...).
   ``None`` (the default) keeps the element dataclass defaults untouched.
3. *Statistical and layer keyword arguments are verbatim:* transform field
   names (``bw``, ``kernel``, ``unique_id``, ``agg_func``, ``err_func``,
   ...) and ``.add()`` layer names (``position``, ``width``, ``jitter_type``,
   ``seed``, ``style``). Verb defaults adopt the new-API defaults (e.g.
   ``KDE bw="ISJ"``), deliberately diverging from the legacy method defaults.

Rendering defaults are layout-aware where the resolver is: ``kde`` draws a
line + fill-under curve on continuous layouts but only the ``Fill`` violin
body on categorical ones. Verbs land family by family with their Phase 2
parity ports; ``line``/``summary``/``bar`` (aggregate family), ``box`` and
the categorical ``hist``/``ecdf`` forms (the ``percent`` analog) are still
pending.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

from typing_extensions import Self

from ...types.basic_types import BW, JitterType, KDEType, Kernels, LayerStyle
from ...types.plot_input import Agg, AlphaRange, ColorParameters, Error
from ..elements import Bar, Element, ErrorBand, ErrorBar, Fill, Line, Marker, SummaryLine
from ..transforms import KDE, Identity, Transform

_ElementT = TypeVar("_ElementT", bound=Element)

_INTENT_FIELDS: dict[type[Element], tuple[str, str]] = {
    Line: ("linecolor", "linealpha"),
    Marker: ("markercolor", "alpha"),
    Fill: ("fillcolor", "fillalpha"),
    Bar: ("edgecolor", "edge_alpha"),
    ErrorBar: ("linecolor", "linealpha"),
    ErrorBand: ("fillcolor", "fillalpha"),
    SummaryLine: ("linecolor", "linealpha"),
}
"""Primary (color, alpha) field per element - intent kwargs never touch edge fields."""


def _with_intent(
    element: _ElementT,
    color: ColorParameters | None,
    alpha: AlphaRange | None,
) -> _ElementT:
    """Apply the intent ``color``/``alpha`` to an element's primary fields.

    ``None`` intent returns ``element`` untouched, so the dataclass defaults
    stay in charge (including the ``glasbey_category10`` palette cycles).

    Args:
        element (Element): The element to style (usually a fresh default).
        color (ColorParameters | None): Primary color field override.
        alpha (AlphaRange | None): Primary alpha field override.

    Returns:
        Element: The styled element - ``element`` itself when neither intent is given.
    """
    color_field, alpha_field = _INTENT_FIELDS[type(element)]
    updates: dict[str, object] = {}
    if color is not None:
        updates[color_field] = color
    if alpha is not None:
        updates[alpha_field] = alpha
    return replace(element, **updates) if updates else element


def _resolve_role(
    role: _ElementT | Literal[False] | None,
    build: Callable[[], _ElementT],
    default_on: bool = True,
) -> _ElementT | None:
    """Resolve an element role: the explicit instance wins, ``False`` drops, ``None`` builds.

    Args:
        role (Element | False | None): An explicit element instance (wins
            completely), ``False`` (drop the role) or ``None`` (default).
        build (Callable[[], Element]): Default-element constructor.
        default_on (bool): Whether a ``None`` role builds the default.

    Returns:
        Element | None: The resolved element, or ``None`` when the role is off.
    """
    if role is False:
        return None
    if role is not None:
        return role
    return build() if default_on else None


class SugarMethods:
    """Verbs shared by every layout: points (``scatter``) and densities (``kde``).

    Mixed into :class:`~lithos.plotting.spec.base.Plot`, which supplies
    ``add`` and the ``layout`` name; layout-specific verbs
    (``hist``/``ecdf``/``fit`` on :class:`~lithos.plotting.spec.line.LinePlot`)
    live on the layout classes.
    """

    layout: ClassVar[str] = "base"

    if TYPE_CHECKING:
        # Provided by Plot at runtime; declared so the layer calls inside
        # the mixin type-check.
        def add(
            self,
            transform: Transform,
            *elements: Element,
            position: str | None = None,
            width: float = 0.9,
            jitter_type: JitterType = "fill",
            seed: int = 42,
            style: LayerStyle | None = None,
        ) -> Self: ...

    def scatter(
        self,
        *,
        unique_id: str | None = None,
        connect: bool = False,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        line: Line | Literal[False] | None = None,
        marker: Marker | Literal[False] | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a points layer: every raw value drawn as a marker.

        One verb for legacy ``scatter`` (continuous), ``jitter``
        (categorical slot spread - ``width``/``jitter_type``/``seed``) and
        ``jitteru`` (``unique_id``: one even column per subject).
        ``connect=True`` joins each subject's ordered series with a line -
        the legacy ``paired`` plot; the series order is data-driven by the
        plot's pairing column, so the legacy ``index``/``order`` arguments
        do not exist.

        Args:
            unique_id (str | None): Subject column; nests the layer so each
                subject's points draw as their own series/column.
            connect (bool): Join the (per-subject) points with lines; adds
                the ``line`` role by default.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            line (Line | False | None): Connector role - an explicit ``Line``
                wins completely, ``False`` drops it even with ``connect=True``,
                ``None`` builds the default (only when ``connect``).
            marker (Marker | False | None): Point role - explicit wins,
                ``False`` drops it, ``None`` builds the default.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of the random point spread.
            seed (int): Seed for the reproducible random spread.

        Returns:
            Self: The plot, for chaining.

        Raises:
            ValueError: When every role is dropped (``.add()`` requires at
                least one element).
        """
        line_element = _resolve_role(line, lambda: _with_intent(Line(), color, alpha), default_on=connect)
        marker_element = _resolve_role(marker, lambda: _with_intent(Marker(), color, alpha))
        elements = [element for element in (line_element, marker_element) if element is not None]
        return self.add(
            Identity(unique_id=unique_id),
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
        )

    def kde(
        self,
        *,
        kernel: Kernels = "gaussian",
        bw: BW = "ISJ",
        tol: float | int | tuple = 1e-3,
        kde_length: int | None = None,
        KDEType: KDEType = "fft",
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        fill: Fill | Literal[False] | None = None,
        line: Line | Literal[False] | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
        style: LayerStyle | None = None,
    ) -> Self:
        """Add a kernel-density layer - the legacy ``kde`` curve and ``violin`` in one verb.

        On a continuous layout the density renders as a fill-under curve
        with a line on top; on a categorical layout the same ``KDE`` geometry
        resolves to a violin body, which only
        :class:`~lithos.plotting.elements.Fill` renders, so the ``line``
        role is off by default there (passing a ``Line`` explicitly surfaces
        the renderer's violin error). Split/one-sided/overlap violins come
        from ``style`` (``"left"``/``"right"``/``"alternate"``/
        ``"overlap"``). With ``unique_id`` every subject gets its own
        density; ``agg_func`` re-aggregates the per-subject densities on a
        shared grid, ``err_func`` giving per-grid-point errors.

        Deliberate divergences from the legacy methods: the verb adopts the
        transform defaults (``bw="ISJ"`` - legacy ``"silverman"``;
        ``kde_length=None`` - legacy violin ``128``) and fills by default
        (legacy ``fill_under=False``; pass ``fill=False`` for the bare curve).

        Args:
            kernel (Kernels): Kernel shape for the density estimate.
            bw (BW): Bandwidth rule or explicit width.
            tol (float | int | tuple): Tolerance passed to the KDE fit.
            kde_length (int | None): Number of evaluation points
                (``None``: data-driven).
            KDEType (KDEType): Estimator engine, e.g. ``"fft"``.
            unique_id (str | None): Subject column; one density per subject.
            agg_func (Agg | None): Re-aggregate the per-subject densities
                across subjects.
            err_func (Error | None): Per-grid-point error for the aggregated
                densities.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            fill (Fill | False | None): Density-body role - an explicit
                ``Fill`` wins completely, ``False`` draws the outline-only
                curve, ``None`` builds the default.
            line (Line | False | None): Curve-outline role - off by default on
                categorical layouts (violin bodies render through ``Fill``).
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.
            style (LayerStyle | None): Violin form (``"left"``/``"right"``/
                ``"alternate"``/``"overlap"``); ``None`` keeps the default
                geometry.

        Returns:
            Self: The plot, for chaining.
        """
        transform = KDE(
            kernel=kernel,
            bw=bw,
            tol=tol,
            kde_length=kde_length,
            KDEType=KDEType,
            unique_id=unique_id,
            agg_func=agg_func,
            err_func=err_func,
        )
        fill_element = _resolve_role(fill, lambda: _with_intent(Fill(), color, alpha))
        line_element = _resolve_role(
            line, lambda: _with_intent(Line(), color, alpha), default_on=self.layout != "categorical"
        )
        elements = [element for element in (fill_element, line_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
            style=style,
        )
