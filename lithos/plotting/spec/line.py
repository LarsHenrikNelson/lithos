"""Continuous-layout spec plot (new API).

``LinePlot`` renders layers on continuous numeric axes; layout-specific
settings are *faceting*, kept separate from grouping (``.facet()``). The
Phase 2.5 sugar verbs for continuous-only statistics live here: ``line()``,
``hist()``, ``ecdf()`` and ``fit()`` (the shared ``scatter()``/``kde()``
verbs come mixed in from :mod:`lithos.plotting.spec.methods`).
"""

from typing import ClassVar, Literal

from typing_extensions import Self

from ...types.basic_types import CIFunc, FitFunc, HistBinLimits, HistStat, JitterType, LayerStyle, NBins
from ...types.plot_input import Agg, AlphaRange, ColorParameters, Error
from ...utils import DataHolder
from ..elements import Bar, ErrorBand, Fill, Line, Marker
from ..plot_utils import _create_groupings, create_dict
from ..transforms import ECDF, Aggregate, Fit, Histogram, Identity
from .base import Plot
from .methods import _resolve_role, _with_intent


class LinePlot(Plot):
    """Continuous x-axis layout with optional faceting (new spec API)."""

    layout: ClassVar[str] = "continuous"
    default_position: ClassVar[str] = "passthrough"
    positions: ClassVar[tuple[str, ...]] = ("passthrough",)

    def __init__(self):
        super().__init__()
        self._layout_options = {"facet": False, "facet_title": False}

    def facet(self, facet: bool = True, facet_title: bool = False) -> Self:
        """Facet by group (one axes per group), optionally titled by group name.

        Args:
            facet (bool): One axes per group. Defaults to True.
            facet_title (bool): Title each axes with its group name.

        Returns:
            Self: The plot, for chaining.
        """
        self._layout_options["facet"] = facet
        self._layout_options["facet_title"] = facet_title
        return self

    # -- Phase 2.5 sugar verbs (continuous-layout statistics) ------------------

    def hist(
        self,
        *,
        bins: NBins = 50,
        bin_limits: HistBinLimits = None,
        stat: HistStat = "density",
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        fill: Fill | bool | None = None,
        bar: Bar | bool | None = None,
        step: Line | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
        style: LayerStyle | None = None,
    ) -> Self:
        """Add a histogram layer: per-group (stackable) bins over the value axis.

        The default composition draws bar faces (:class:`Fill`) with
        outlines (:class:`Bar`) - legacy ``hist_type="bar"``; the legacy
        ``hist_type="step"`` outline is role composition instead:
        ``hist(fill=False, bar=False, step=Line())``. Stacking comes from
        ``style``: ``"stack"`` (cumulative) and ``"fill"`` (100% stacked),
        which require shared bin edges (``bin_limits="common"``) and cannot
        carry error geometry. With ``unique_id`` every subject gets its own
        histogram; ``agg_func`` re-aggregates them, ``err_func`` giving
        per-bin errors. The verb adopts the transform default
        ``stat="density"`` (legacy: ``"count"``).

        Args:
            bins (NBins): Number of bins (or explicit bin edges).
            bin_limits (HistBinLimits): Bin range policy - per-group auto,
                ``"common"`` global shared edges, or an explicit ``(low, high)``.
            stat (HistStat): Height statistic (``"density"``, ``"count"``, ...).
            unique_id (str | None): Subject column; one histogram per subject.
            agg_func (Agg | None): Re-aggregate the per-subject heights per group.
            err_func (Error | None): Per-bin error for the aggregated heights.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            fill (Fill | bool | None): Bar-face role - an explicit ``Fill``
                wins completely, ``False`` drops it, ``None`` builds the default.
            bar (Bar | bool | None): Bar-outline role - an explicit ``Bar``
                wins completely, ``False`` drops it, ``None`` builds the default.
            step (Line | bool | None): Step outline role - off by default;
                ``True`` accepts the intent-styled default; an explicit
                ``Line`` wins completely. Either draws the legacy
                ``hist_type="step"`` outline over the bins.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.
            style (LayerStyle | None): Stacking form (``"stack"``/``"fill"``);
                ``None`` keeps the default geometry.

        Returns:
            Self: The plot, for chaining.
        """
        transform = Histogram(
            bins=bins,
            bin_limits=bin_limits,
            stat=stat,
            unique_id=unique_id,
            agg_func=agg_func,
            err_func=err_func,
        )
        fill_element = _resolve_role(fill, lambda: _with_intent(Fill(), color, alpha))
        bar_element = _resolve_role(bar, lambda: _with_intent(Bar(), color, alpha))
        step_element = _resolve_role(step, lambda: _with_intent(Line(), color, alpha), default_on=False)
        elements = [element for element in (fill_element, bar_element, step_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
            style=style,
        )

    def ecdf(
        self,
        *,
        ecdf_type: Literal["bootstrap", "spline", "none"] = "none",
        ecdf_args: dict | None = None,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        fill: Fill | bool | None = None,
        line: Line | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
        style: LayerStyle | None = None,
    ) -> Self:
        """Add an empirical-CDF layer (legacy ``LinePlot.ecdf``).

        Unlike the legacy method the verb does not pin axis limits: the
        probability axis follows the value column's axis (values rendered on
        x put the probabilities on y and vice versa), which is only known at
        ``.plot()`` - pin it explicitly with ``.axis(ylim=(0.0, 1.0))`` or
        ``.axis(xlim=(0.0, 1.0))``. ``ecdf_type`` selects the variant:
        ``"none"`` (raw staircase, default), ``"spline"`` or ``"bootstrap"``
        (extra arguments via ``ecdf_args``); the aggregated two-level branch
        requires ``ecdf_type="spline"``. With ``unique_id`` every subject
        gets its own ECDF; ``agg_func`` re-aggregates the per-subject
        quantile functions on a shared probability grid, ``err_func`` giving
        per-grid-point errors.

        Args:
            ecdf_type ("bootstrap" | "spline" | "none"): ECDF variant.
            ecdf_args (dict | None): Extra arguments for the chosen variant
                (e.g. ``size`` for ``"spline"``/``"bootstrap"``).
            unique_id (str | None): Subject column; one ECDF per subject.
            agg_func (Agg | None): Re-aggregate the per-subject ECDFs
                across subjects.
            err_func (Error | None): Per-grid-point error for the
                aggregated ECDFs.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            fill (Fill | bool | None): Fill-under role - an explicit ``Fill``
                wins completely, ``False`` draws the bare staircase,
                ``None`` builds the default.
            line (Line | bool | None): Staircase-outline role - an explicit
                ``Line`` wins completely, ``False`` drops it, ``None`` builds
                the default.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.
            style (LayerStyle | None): Reserved - ``ECDF`` has no layer form
                styles yet (``.add()`` rejects one).

        Returns:
            Self: The plot, for chaining.
        """
        transform = ECDF(
            ecdf_type=ecdf_type,
            ecdf_args=ecdf_args if ecdf_args is not None else {},
            unique_id=unique_id,
            agg_func=agg_func,
            err_func=err_func,
        )
        fill_element = _resolve_role(fill, lambda: _with_intent(Fill(), color, alpha))
        line_element = _resolve_role(line, lambda: _with_intent(Line(), color, alpha))
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

    def fit(
        self,
        *,
        fit_func: FitFunc = "linear",
        ci_func: CIFunc | None = "ci",
        fit_args: dict | None = None,
        unique_id: str | None = None,
        agg_func: Agg | None = None,
        err_func: Error = None,
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        band: ErrorBand | bool | None = None,
        line: Line | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
        style: LayerStyle | None = None,
    ) -> Self:
        """Add a curve-fit layer: a fitted line with an interval band (legacy ``LinePlot.fit``).

        ``fit_func`` selects a built-in fit (``"linear"``, ``"polynomial"``
        with ``fit_args={"degree": ...}``, ``"sine"``) or any callable, and
        ``ci_func`` the interval around the fit (``"ci"`` confidence /
        ``"pi"`` prediction / ``"bootstrap_ci"``); ``None`` draws the bare
        fit and drops the ``band`` role by default (pass ``band=ErrorBand()``
        to override and let the renderer explain). With ``unique_id`` each
        subject is fitted separately; ``agg_func`` re-aggregates the
        per-subject fits on a shared grid, ``err_func`` giving per-grid-point
        errors. The verb adopts the transform defaults (``agg_func=None``);
        the legacy method defaulted ``agg_func="mean"``/``err_func="sem"`` -
        pass them explicitly for that behavior.

        Args:
            fit_func (FitFunc): Built-in fit name or callable.
            ci_func (CIFunc | None): Interval type around the fit, or ``None``
                for the bare fit line.
            fit_args (dict | None): Extra keyword arguments for the fit
                function (e.g. ``degree`` for polynomial fits).
            unique_id (str | None): Subject column; one fit per subject.
            agg_func (Agg | None): Re-aggregate the per-subject fits across
                subjects.
            err_func (Error | None): Per-grid-point error for the aggregated
                fits.
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            band (ErrorBand | bool | None): Interval-band role - an explicit
                ``ErrorBand`` wins completely, ``False`` drops it, ``None``
                builds the default (off when ``ci_func is None``).
            line (Line | bool | None): Fit-line role - an explicit ``Line``
                wins completely, ``False`` drops it, ``None`` builds the
                default.
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.
            style (LayerStyle | None): Reserved - ``Fit`` has no layer form
                styles yet (``.add()`` rejects one).

        Returns:
            Self: The plot, for chaining.
        """
        transform = Fit(
            fit_func=fit_func,
            ci_func=ci_func,
            fit_args=fit_args if fit_args is not None else {},
            unique_id=unique_id,
            agg_func=agg_func,
            err_func=err_func,
        )
        band_element = _resolve_role(
            band, lambda: _with_intent(ErrorBand(), color, alpha), default_on=ci_func is not None
        )
        line_element = _resolve_role(line, lambda: _with_intent(Line(), color, alpha))
        elements = [element for element in (band_element, line_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
            style=style,
        )

    def line(
        self,
        *,
        func: Agg | None = None,
        err_func: Error | None = None,
        unique_id: str | None = None,
        how: Literal["auto", "groupby", "matrix"] = "auto",
        color: ColorParameters | None = None,
        alpha: AlphaRange | None = None,
        line: Line | bool | None = None,
        marker: Marker | bool | None = None,
        band: ErrorBand | bool | None = None,
        position: str | None = None,
        width: float = 0.9,
        jitter_type: JitterType = "fill",
        seed: int = 42,
    ) -> Self:
        """Add a line layer: raw series, or an aggregated line (legacy ``line``/``aggline``).

        ``func=None`` (default) draws the raw series through every point -
        ``Identity(unique_id=...)``: one line per group, or one per subject
        with ``unique_id``. With ``func`` the verb switches to
        ``Aggregate``: y is aggregated at each unique value of the plot's
        ``x`` column (per-x semantics; ``err_func``/``how`` apply only to
        this branch - ragged data aggregates via ``how="groupby"``, aligned
        ``unique_id`` data via the fast ``"matrix"`` pivot, and ``"auto"``
        picks). With both ``func`` and ``err_func`` the interval band
        defaults on (legacy ``line`` filled between its errors; the legacy
        ``aggline`` fixed ``fill_between=False`` - pass ``band=False``
        there). Markers are a role (legacy default ``marker="none"``).

        The legacy ``index=``/``sort=`` arguments are gone: pairing is the
        plot's ``x`` column (data-driven), and per-x aggregation always
        evaluates on the sorted grid.

        Args:
            func (Agg | None): Aggregate y at each x with this function;
                ``None`` draws the raw series (``Identity``).
            err_func (Error | None): Per-x error around the aggregated line
                (aggregated branch only).
            unique_id (str | None): Subject column; per-subject raw series
                (``func=None``) or per-x matrix aggregation (``func`` given).
            how ("auto" | "groupby" | "matrix"): Per-x aggregation strategy
                (see :class:`~lithos.plotting.transforms.Aggregate`).
            color (ColorParameters | None): Intent color for the role
                defaults' primary color fields; ``None`` keeps element defaults.
            alpha (AlphaRange | None): Intent alpha for the role defaults'
                primary alpha fields.
            line (Line | bool | None): Series role - an explicit ``Line``
                wins completely, ``False`` drops it, ``None`` builds the default.
            marker (Marker | bool | None): Point role - absent by default;
                ``True`` accepts the intent-styled default, an explicit
                ``Marker`` wins completely.
            band (ErrorBand | bool | None): Interval-band role - an explicit
                ``ErrorBand`` wins completely, ``False`` drops it, ``None``
                builds the default (on only when ``func`` and ``err_func``
                are both given).
            position (str | None): Resolver position mode; ``None`` uses the
                layout default.
            width (float): Slot footprint fraction (categorical layouts).
            jitter_type (JitterType): Shape of any random point spread.
            seed (int): Seed for the reproducible spread.

        Returns:
            Self: The plot, for chaining.
        """
        transform = (
            Identity(unique_id=unique_id)
            if func is None
            else Aggregate(func=func, err_func=err_func, unique_id=unique_id, how=how)
        )
        band_element = _resolve_role(
            band, lambda: _with_intent(ErrorBand(), color, alpha), default_on=func is not None and err_func is not None
        )
        line_element = _resolve_role(line, lambda: _with_intent(Line(), color, alpha))
        marker_element = _resolve_role(marker, lambda: _with_intent(Marker(), color, alpha), default_on=False)
        elements = [element for element in (band_element, line_element, marker_element) if element is not None]
        return self.add(
            transform,
            *elements,
            position=position,
            width=width,
            jitter_type=jitter_type,
            seed=seed,
        )

    def _layout_context(self, data: DataHolder) -> dict:
        """Build the continuous layout context (axes index per group key)."""
        group_order, subgroup_order, unique_groups, levels = _create_groupings(
            data,
            self._grouping["group"],
            self._grouping["subgroup"],
            self._grouping["group_order"],
            self._grouping["subgroup_order"],
        )
        if self._layout_options["facet"]:
            # one axes index per group
            loc_dict = create_dict(group_order, unique_groups)
        else:
            loc_dict = create_dict(0, unique_groups)

        return {
            "layout": self.layout,
            "group_order": list(group_order),
            "subgroup_order": list(subgroup_order) if subgroup_order is not None else None,
            "unique_groups": [tuple(group) for group in unique_groups],
            "levels": tuple(levels),
            "loc_dict": loc_dict,
            "facet": self._layout_options["facet"],
            "facet_title": self._layout_options["facet_title"],
        }
