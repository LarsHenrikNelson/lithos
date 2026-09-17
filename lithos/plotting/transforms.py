"""Transform dataclasses for the new plot API.

Transforms are pure statistical operations: grouped data in, *geometry* out.
They have no knowledge of colors, markers or backends, and they own all
aggregation and error computation (see ``doc/migration_plan.md``).

Each transform ``__call__`` returns ``dict[group_key, geometry_dict]`` where
the geometry dict is the only source of numbers that elements render.

Geometry contract per transform:

- ``Identity``: ``{n, x?, y?}`` — at least one of x/y; the omitted axis is
  supplied by the position resolver (jitter/dodge; horizontal layouts use
  ``x`` only). With ``unique_id``: per-subject ordered series (legacy
  ``paired`` connector geometry), sorted by ``x`` when given
- ``Aggregate``: arrays — without ``x``: ``{center, error_low, error_high,
  n}`` (one value per group); with ``x``: ``{x, center, error_low,
  error_high, n}`` (one value per unique x; per-x aggregation, legacy
  ``aggline``/``line`` parity)
- ``KDE``/``ECDF``: ``{x, y, n}``; ``Histogram``: ``{edges, height,
  binwidth, centers, stat, n}``. With ``unique_id``: one geometry dict per
  subject (nested group keys; per-group shared bin edges for ``Histogram``).
  With ``unique_id`` + ``agg_func``: per-subject densities on a shared grid,
  re-aggregated per group — ``KDE``: ``{x, y, error_low, error_high, n}``;
  ``Histogram``: ``{edges, height, error_low, error_high, binwidth,
  centers, stat, n}``; ``ECDF``: ``{x, y, error_low, error_high, n}``
  (``x`` = aggregated quantile values, ``y`` = shared probability grid);
  ``n`` = subjects, errors per grid point/bin
- ``Summary``: ``{center, mean, median, q1, q3, whisker_low, whisker_high,
  error_low, error_high, notch_low, notch_high, n}``
- ``Fit``: ``{x, y, ci, n}``

Error normalization contract: an error function must return a scalar
(symmetric) or a pair ``(low, high)``; transforms normalize both to
``error_low`` / ``error_high``.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..stats import ecdf, fit, hist, kde
from ..types.basic_types import (
    BW,
    CIFunc,
    FitFunc,
    HistBinLimits,
    HistStat,
    KDEType,
    Kernels,
    Levels,
    NBins,
    Transform,
)
from ..types.plot_input import Agg, Error
from ..utils import DataHolder, get_transform

__all__ = [
    "Aggregate",
    "as_error_pair",
    "ECDF",
    "Fit",
    "Histogram",
    "Identity",
    "KDE",
    "Summary",
    "Transform",
]


def as_error_pair(e) -> tuple[float | None, float | None]:
    """Normalize an error function output to a ``(low, high)`` pair.

    Scalars become a symmetric pair; a length-2 sequence is treated as
    ``(low, high)`` already.
    """
    if e is None:
        return None, None
    arr = np.asarray(e).ravel()
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    elif arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError(f"Error functions must return a scalar or a (low, high) pair, got {arr.size} values.")


def _get_column_values(data: DataHolder, indexes: np.ndarray, column: str, tr: Transform | None = None) -> np.ndarray:
    """Extract and optionally transform a column for a set of row indexes."""
    vals = np.asarray(data[indexes, column], dtype=float)
    if tr is not None:
        vals = np.asarray(get_transform(tr)(vals), dtype=float)
    return vals


@dataclass
class Transform:
    """Base class for statistical transforms.

    Subclasses implement ``__call__`` returning a per-group geometry dict.
    """

    name: str = "transform"

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        *args,
        **kwargs,
    ) -> dict[tuple, dict]:
        """Compute per-group geometry.

        Args:
            data: the input data.
            y: value column.
            x: independent column (required by ``Fit``; optional for
              ``Identity`` and per-x ``Aggregate`` — the omitted axis is
              supplied by the position resolver).
            levels: grouping columns (group, subgroup, ...).
            ytransform: transform applied to y values (log10 etc).
            xtransform: transform applied to x values.
        """
        raise NotImplementedError("Transforms must implement __call__.")

    def _groups(self, data: DataHolder, levels: Levels) -> dict[tuple, np.ndarray]:
        """Return ``{group_key: row_indexes}`` for the given grouping levels."""
        return data.groups(tuple(levels) if levels is not None else ())


@dataclass
class Identity(Transform):
    """Pass raw values through as per-group point geometry.

    Requires at least one of ``x``/``y``. The omitted axis is left out of
    the geometry dict so the position resolver (jitter/dodge) can supply it;
    this mirrors the legacy ``jitter`` processor and enables horizontal
    layouts (``x`` only). Both given -> 2-D scatter.

    ``unique_id`` optionally nests the group key so each geometry dict is
    a single subject's ordered series — the paired-connector geometry
    (legacy ``paired``). The series is sorted by ``x`` (the order/pairing
    column) when given, otherwise kept in row order. When set, the data is
    validated for pairing: every subject within a group must hold the same
    complete set of order values, each appearing exactly once.
    """

    name: str = "identity"
    unique_id: str | None = None

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        **kwargs,
    ) -> dict[tuple, dict]:
        if x is None and y is None:
            raise ValueError("Identity requires an x or y column.")
        levels = tuple(levels)
        if self.unique_id is None:
            groups = self._groups(data, levels)
        else:
            groups = self._groups(data, levels + (self.unique_id,))
            self._validate_pairs(data, x, groups, levels)
        output = {}
        for group_key, indexes in groups.items():
            geometry = {"n": int(indexes.size)}
            if x is not None:
                geometry["x"] = _get_column_values(data, indexes, x, xtransform)
            if y is not None:
                geometry["y"] = _get_column_values(data, indexes, y, ytransform)
            if self.unique_id is not None and x is not None:
                # paired series: order the connection points by the order column
                order = np.argsort(geometry["x"], kind="stable")
                geometry["x"] = geometry["x"][order]
                if y is not None:
                    geometry["y"] = geometry["y"][order]
            output[group_key] = geometry
        return output

    def _validate_pairs(self, data: DataHolder, x: str | None, groups: dict[tuple, np.ndarray], levels: tuple) -> None:
        """Validate paired alignment within each group.

        Every ``unique_id`` within a group must have the same number of rows
        (legacy "missing or extra values"), each order value at most once,
        and every subject must share the same complete set of order values
        (legacy "unique_ids missing values").
        """
        n_levels = len(levels)
        per_group: dict[tuple, list] = defaultdict(list)
        for key, indexes in groups.items():
            per_group[key[:n_levels] if n_levels > 0 else ()].append(indexes)
        for index_lists in per_group.values():
            sizes = [idx.size for idx in index_lists]
            if len(set(sizes)) != 1 or sizes[0] == 0:
                raise AttributeError("Some pairs may have missing or extra values.")
            if x is None:
                continue
            per_uid = [set(np.asarray(data[idx, x]).tolist()) for idx in index_lists]
            for order_vals, idx in zip(per_uid, index_lists):
                if len(order_vals) != idx.size:
                    raise AttributeError("Some pairs may have missing or extra values.")
            if not all(order_vals == per_uid[0] for order_vals in per_uid):
                raise ValueError("Some unique_ids are missing values. N rows divide number of pairings must equal 0.")


@dataclass
class Aggregate(Transform):
    """Aggregate y per group — optionally along x, optionally nested by ``unique_id``.

    Args:
        func: aggregation function applied to y within each group (or within
            each unique_id first).
        err_func: error function applied to the same values; the result is
            normalized to ``error_low``/``error_high``.
        agg_func: second-level aggregation applied across unique_id samples
            when ``unique_id`` is given (defaults to ``func``).
        unique_id: column whose unique values are first aggregated with
            ``func``, then re-aggregated per group with ``agg_func``.
        how: per-x aggregation strategy when ``x`` is given (``x`` must be
            numeric and sortable):

            - ``"groupby"``: aggregate per (levels, x) — handles ragged data
              (unequal y counts or missing x values per uid).
            - ``"matrix"``: pivot to a dense (uid, x) matrix and aggregate
              ``axis=0`` — the fast path for aligned data; requires
              ``unique_id`` and every uid to share the same x grid.
            - ``"auto"`` (default): ``matrix`` when the data is aligned,
              ``groupby`` otherwise.

    Geometry per group (always arrays):

    - without ``x``: ``{center, error_low, error_high, n}`` — length-1
      arrays (one aggregated value per group).
    - with ``x``: ``{x, center, error_low, error_high, n}`` — one entry per
      unique x value (per-x aggregation, legacy ``aggline``/``line`` parity).

    ``error_low``/``error_high`` are ``None`` when ``err_func`` is not given.
    With ``unique_id`` the error is computed on the first-level (per-uid)
    aggregates, otherwise on the raw values.
    """

    name: str = "aggregate"
    func: Agg = "mean"
    err_func: Error = None
    agg_func: Agg | None = None
    unique_id: str | None = None
    how: Literal["auto", "groupby", "matrix"] = "auto"

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        **kwargs,
    ) -> dict[tuple, dict]:
        if y is None:
            raise ValueError("Aggregate requires a y column.")
        levels = tuple(levels)
        if x is None:
            return self._aggregate_per_group(data, y, levels, ytransform)
        return self._aggregate_per_x(data, y, x, levels, ytransform, xtransform)

    def _error_pair(self, vals: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Apply ``err_func`` to one sample of values, returning length-1 error arrays."""
        if self.err_func is None:
            return None, None
        low, high = as_error_pair(get_transform(self.err_func)(vals))
        low = None if low is None else np.array([low])
        high = None if high is None else np.array([high])
        return low, high

    def _aggregate_per_group(
        self, data: DataHolder, y: str, levels: tuple, ytransform: Transform | None
    ) -> dict[tuple, dict]:
        """Aggregate y to a single value per group (bars / summary points)."""
        first = get_transform(self.func)
        output = {}
        if self.unique_id is None:
            for group_key, indexes in self._groups(data, levels).items():
                vals = _get_column_values(data, indexes, y, ytransform)
                low, high = self._error_pair(vals)
                output[group_key] = {
                    "center": np.array([float(first(vals))]),
                    "error_low": low,
                    "error_high": high,
                    "n": np.array([vals.size]),
                }
            return output

        # Nested aggregation: first over (levels + unique_id), then over levels.
        sub_levels = levels + (self.unique_id,)
        n_levels = len(levels)
        second = get_transform(self.agg_func if self.agg_func is not None else self.func)
        per_group: dict[tuple, list] = defaultdict(list)
        for sub_key, sub_indexes in self._groups(data, sub_levels).items():
            vals = _get_column_values(data, sub_indexes, y, ytransform)
            per_group[sub_key[:n_levels] if n_levels > 0 else ("",)].append(first(vals))
        for group_key, centers in per_group.items():
            centers = np.asarray(centers, dtype=float)
            low, high = self._error_pair(centers)
            output[group_key] = {
                "center": np.array([float(second(centers))]),
                "error_low": low,
                "error_high": high,
                "n": np.array([centers.size]),
            }
        return output

    def _aggregate_per_x(
        self,
        data: DataHolder,
        y: str,
        x: str,
        levels: tuple,
        ytransform: Transform | None,
        xtransform: Transform | None,
    ) -> dict[tuple, dict]:
        """Aggregate y at each unique x value per group (legacy ``aggline``/``line``)."""
        groups = self._groups(data, levels)
        second = get_transform(self.agg_func if self.agg_func is not None else self.func)

        if self.how == "auto":
            use_matrix = self.unique_id is not None and self._is_aligned(data, x, groups)
        else:
            use_matrix = self.how == "matrix"
        if use_matrix:
            if self.unique_id is None:
                raise ValueError("Aggregate how='matrix' requires a unique_id column.")
            return {
                group_key: self._matrix_geometry(data, indexes, y, x, ytransform, xtransform, second)
                for group_key, indexes in groups.items()
            }

        # groupby path: aggregate per (levels, x) — handles ragged data.
        first = get_transform(self.func)
        n_levels = len(levels)
        sub_levels = levels + (x,) if self.unique_id is None else levels + (x, self.unique_id)
        per_group_x: dict[tuple, dict] = defaultdict(dict)
        for sub_key, sub_indexes in self._groups(data, sub_levels).items():
            gkey = sub_key[:n_levels] if n_levels > 0 else ("",)
            xv = sub_key[n_levels]
            vals = _get_column_values(data, sub_indexes, y, ytransform)
            if self.unique_id is None:
                per_group_x[gkey][xv] = vals
            else:
                per_group_x[gkey].setdefault(xv, []).append(first(vals))

        output = {}
        for group_key, per_x in per_group_x.items():
            xs = sorted(per_x.keys())
            centers, lows, highs, ns = [], [], [], []
            for xv in xs:
                values = np.asarray(per_x[xv], dtype=float)
                if self.unique_id is None:
                    centers.append(float(first(values)))
                else:
                    centers.append(float(second(values)))
                low, high = self._error_pair(values)
                lows.append(low[0] if low is not None else None)
                highs.append(high[0] if high is not None else None)
                ns.append(int(values.size))
            if self.err_func is None:
                error_low = error_high = None
            else:
                error_low = np.asarray(lows, dtype=float)
                error_high = np.asarray(highs, dtype=float)
            output[group_key] = {
                "x": np.asarray(get_transform(xtransform)(np.asarray(xs, dtype=float)), dtype=float),
                "center": np.asarray(centers, dtype=float),
                "error_low": error_low,
                "error_high": error_high,
                "n": np.asarray(ns, dtype=int),
            }
        return output

    def _is_aligned(self, data: DataHolder, x: str, groups: dict[tuple, np.ndarray]) -> bool:
        """True when every unique_id shares the same complete x grid within each group."""
        for indexes in groups.values():
            uid_vals = np.asarray(data[indexes, self.unique_id])
            x_vals = np.asarray(data[indexes, x])
            if indexes.size != np.unique(uid_vals).size * np.unique(x_vals).size:
                return False
            pairs = np.unique(np.stack([x_vals, uid_vals], axis=1), axis=0)
            if pairs.shape[0] != indexes.size:
                return False
        return True

    def _matrix_geometry(
        self,
        data: DataHolder,
        indexes: np.ndarray,
        y: str,
        x: str,
        ytransform: Transform | None,
        xtransform: Transform | None,
        second,
    ) -> dict:
        """Fast path: pivot the group to a dense (uid, x) matrix and aggregate axis=0."""
        uid_vals = np.asarray(data[indexes, self.unique_id])
        x_vals = np.asarray(data[indexes, x])
        yvals = _get_column_values(data, indexes, y, ytransform)
        xs = np.unique(x_vals)
        uids = np.unique(uid_vals)
        pairs = np.unique(np.stack([x_vals, uid_vals], axis=1), axis=0)
        if indexes.size != uids.size * xs.size or pairs.shape[0] != indexes.size:
            raise ValueError(
                "Aggregate how='matrix' requires every unique_id to share the same x grid "
                "(each (unique_id, x) pair appearing at most once); use how='groupby' or 'auto'."
            )
        matrix = np.full((uids.size, xs.size), np.nan)
        matrix[np.searchsorted(uids, uid_vals), np.searchsorted(xs, x_vals)] = yvals
        if self.err_func is not None:
            error = np.asarray(get_transform(self.err_func)(matrix, axis=0), dtype=float)
            error_low = error_high = error
        else:
            error_low = error_high = None
        return {
            "x": np.asarray(get_transform(xtransform)(xs), dtype=float),
            "center": np.asarray(second(matrix, axis=0), dtype=float),
            "error_low": error_low,
            "error_high": error_high,
            "n": np.full(xs.size, uids.size, dtype=int),
        }


@dataclass
class _DensityTransform(Transform):
    """Shared machinery for the density transforms (``KDE``/``Histogram``/``ECDF``).

    ``unique_id`` nests the group key so each geometry dict is a single
    subject's density. With ``unique_id`` + ``agg_func`` the densities are
    first computed per subject on a *shared grid* and then re-aggregated per
    group (legacy ``kde``/``hist``/``ecdf`` two-level semantics):
    ``agg_func``/``err_func`` are applied across the subject curves, with
    errors normalized per grid point to ``error_low``/``error_high`` (scalar
    errors are symmetric). In the two-level geometry ``n`` is the subject
    count; otherwise ``n`` is the value count.

    Subclasses implement ``_density_geometry(vals, edges=None)`` (one group
    or one subject) and ``_aggregate_geometry(...)`` (two-level per group),
    and may override ``_common_edges``/``_group_edges`` (shared bin edges)
    and ``_validate``.
    """

    name: str = "density"
    unique_id: str | None = None
    agg_func: Agg | None = None
    err_func: Error = None

    def _validate(self) -> None:
        """Per-transform validation; subclasses extend and call ``super()``."""
        if self.agg_func is not None and self.unique_id is None:
            raise ValueError(f"{type(self).__name__} agg_func requires a unique_id column.")

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        **kwargs,
    ) -> dict[tuple, dict]:
        if y is None:
            raise ValueError(f"{type(self).__name__} requires a y column.")
        self._validate()

        levels = tuple(levels)
        groups = self._groups(data, levels)
        unique_groups = self._groups(data, levels + (self.unique_id,)) if self.unique_id is not None else None
        common_edges = self._common_edges(data, y, ytransform)

        output = {}
        for group_key, indexes in groups.items():
            vals = _get_column_values(data, indexes, y, ytransform)
            edges = self._group_edges(vals) if common_edges is None else common_edges
            if self.unique_id is None:
                output[group_key] = self._density_geometry(vals, edges=edges)
                continue

            uids = np.unique(np.asarray(data[indexes, self.unique_id]))
            if self.agg_func is None:
                # nesting: one geometry dict per (group, subject)
                for uid in uids:
                    sub_indexes = unique_groups[group_key + (uid,)]
                    sub_vals = _get_column_values(data, sub_indexes, y, ytransform)
                    output[group_key + (uid,)] = self._density_geometry(sub_vals, edges=edges)
            else:
                # two-level: per-subject densities on the shared grid, re-aggregated per group
                output[group_key] = self._aggregate_geometry(
                    vals, uids, group_key, unique_groups, data, y, ytransform, edges
                )
        return output

    def _subject_values(
        self,
        group_key: tuple,
        uids: np.ndarray,
        unique_groups: dict[tuple, np.ndarray],
        data: DataHolder,
        y: str,
        ytransform: Transform | None,
    ) -> list[np.ndarray]:
        """Per-subject value arrays within a group, in uid order."""
        return [_get_column_values(data, unique_groups[group_key + (uid,)], y, ytransform) for uid in uids]

    def _density_geometry(self, vals: np.ndarray, edges: np.ndarray | None = None) -> dict:
        """Density geometry for one set of values (one group or one subject)."""
        raise NotImplementedError("Density transforms must implement _density_geometry.")

    def _aggregate_geometry(
        self,
        vals: np.ndarray,
        uids: np.ndarray,
        group_key: tuple,
        unique_groups: dict[tuple, np.ndarray],
        data: DataHolder,
        y: str,
        ytransform: Transform | None,
        edges: np.ndarray | None,
    ) -> dict:
        """Two-level geometry: per-subject densities on a shared grid, aggregated per group."""
        raise NotImplementedError("Density transforms must implement _aggregate_geometry.")

    def _common_edges(self, data: DataHolder, y: str, ytransform: Transform | None) -> np.ndarray | None:
        """Global shared bin edges (``bin_limits="common"``); ``None`` otherwise."""
        return None

    def _group_edges(self, vals: np.ndarray) -> np.ndarray | None:
        """Per-group shared bin edges; ``None`` when the transform has no bins."""
        return None

    def _pair_error(self, hold: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Normalize ``err_func`` output across subjects to per-grid-point (low, high).

        Per-grid-point extension of ``as_error_pair``: a ``(2, n)`` array is
        treated as ``(low, high)`` rows; anything else is symmetric per point.
        """
        if self.err_func is None:
            return None, None
        error = np.asarray(get_transform(self.err_func)(hold, axis=0), dtype=float)
        if error.ndim == 2 and error.shape[0] == 2:
            return error[0], error[1]
        return error, error.copy()


@dataclass
class KDE(_DensityTransform):
    """Smooth per-group density curves via ``KDEpy``.

    ``unique_id`` nests the group key so each geometry dict is one subject's
    KDE (per-subject grids). With ``unique_id`` + ``agg_func`` each subject
    KDE is evaluated on a shared grid built from the pooled group values
    (``tol`` pads the range; a ``(low, high)`` tuple fixes it; grid
    evaluation is FFT) and ``agg_func``/``err_func`` are applied across the
    subject curves.

    Geometry per group/subject: ``{x, y, n}``; two-level per group:
    ``{x, y, error_low, error_high, n}`` (``n`` = subjects, errors per grid
    point).
    """

    name: str = "kde"
    kernel: Kernels = "gaussian"
    bw: BW = "ISJ"
    tol: float | int | tuple = 1e-3
    kde_length: int | None = None
    KDEType: KDEType = "fft"

    def _density_geometry(self, vals: np.ndarray, edges: np.ndarray | None = None) -> dict:
        if vals.size < 2:
            xv, yv = vals, np.zeros_like(vals, dtype=float)
        else:
            xv, yv = kde(
                vals,
                kernel=self.kernel,
                bw=self.bw,
                tol=self.tol,
                kde_length=self.kde_length,
                KDEType=self.KDEType,
            )
        return {"x": xv, "y": yv, "n": int(vals.size)}

    def _aggregate_geometry(
        self,
        vals: np.ndarray,
        uids: np.ndarray,
        group_key: tuple,
        unique_groups: dict[tuple, np.ndarray],
        data: DataHolder,
        y: str,
        ytransform: Transform | None,
        edges: np.ndarray | None,
    ) -> dict:
        grid = self._shared_grid(vals)
        sub_vals = self._subject_values(group_key, uids, unique_groups, data, y, ytransform)
        # grid evaluation is FFT-only (legacy behavior, made explicit)
        hold = np.vstack(
            [kde(v, kernel=self.kernel, bw=self.bw, tol=self.tol, x=grid, KDEType="fft")[1] for v in sub_vals]
        )
        center = np.asarray(get_transform(self.agg_func)(hold, axis=0), dtype=float)
        low, high = self._pair_error(hold)
        return {"x": grid, "y": center, "error_low": low, "error_high": high, "n": int(uids.size)}

    def _shared_grid(self, vals: np.ndarray) -> np.ndarray:
        """Shared evaluation grid from the pooled group values (legacy semantics)."""
        if isinstance(self.tol, tuple):
            lo, hi = float(self.tol[0]), float(self.tol[1])
        else:
            lo, hi = float(vals.min()), float(vals.max())
            # relative padding, guarding a zero bound (legacy behavior)
            lo = lo - abs(lo * self.tol) if lo != 0 else -1e-10
            hi = hi + abs(hi * self.tol) if hi != 0 else 1e-10
        kde_length = self.kde_length if self.kde_length is not None else 1 << int(np.ceil(np.log2(max(vals.size, 2))))
        return np.linspace(lo, hi, num=kde_length)


@dataclass
class Histogram(_DensityTransform):
    """Per-group histogram geometry.

    ``bin_limits`` unifies range control:

    - ``None`` (default): per-group auto range (group min/max);
    - ``"common"``: global pooled range — every group/subject shares the
      same edges;
    - ``(low, high)``: explicit fixed range.

    ``unique_id`` nests the group key with per-group shared edges so subject
    hists are comparable. With ``unique_id`` + ``agg_func`` (integer ``bins``
    required) the per-subject heights are aggregated per group with per-bin
    errors.

    Geometry per group/subject: ``{edges, height, binwidth, centers, stat,
    n}``; two-level per group: ``{edges, height, error_low, error_high,
    binwidth, centers, stat, n}`` (``n`` = subjects, errors per bin).
    """

    name: str = "histogram"
    bins: NBins = 50
    bin_limits: HistBinLimits = None
    stat: HistStat = "density"

    def _validate(self) -> None:
        super()._validate()
        if self.agg_func is not None and isinstance(self.bins, str):
            raise ValueError("bins must be an integer when agg_func is given.")

    def _edges(self, vals: np.ndarray) -> np.ndarray:
        """Bin edges for one set of values under the current ``bin_limits``."""
        limits = self.bin_limits if isinstance(self.bin_limits, tuple) else None
        return np.histogram_bin_edges(vals, bins=self.bins, range=limits)

    def _common_edges(self, data: DataHolder, y: str, ytransform: Transform | None) -> np.ndarray | None:
        if self.bin_limits == "common":
            all_vals = _get_column_values(data, np.arange(data.shape[0]), y, ytransform)
            return self._edges(all_vals)
        return None

    def _group_edges(self, vals: np.ndarray) -> np.ndarray | None:
        # per-group shared bin edges so subject hists are comparable
        return self._edges(vals)

    def _density_geometry(self, vals: np.ndarray, edges: np.ndarray | None = None) -> dict:
        edges = self._edges(vals) if edges is None else edges
        height = hist(vals, edges, self.stat)
        return {
            "edges": edges,
            "height": height,
            "binwidth": np.diff(edges),
            "centers": edges[:-1] + np.diff(edges) / 2,
            "stat": self.stat,
            "n": int(vals.size),
        }

    def _aggregate_geometry(
        self,
        vals: np.ndarray,
        uids: np.ndarray,
        group_key: tuple,
        unique_groups: dict[tuple, np.ndarray],
        data: DataHolder,
        y: str,
        ytransform: Transform | None,
        edges: np.ndarray | None,
    ) -> dict:
        edges = self._edges(vals) if edges is None else edges
        sub_vals = self._subject_values(group_key, uids, unique_groups, data, y, ytransform)
        hold = np.vstack([hist(v, edges, self.stat) for v in sub_vals])
        center = np.asarray(get_transform(self.agg_func)(hold, axis=0), dtype=float)
        low, high = self._pair_error(hold)
        return {
            "edges": edges,
            "height": center,
            "error_low": low,
            "error_high": high,
            "binwidth": np.diff(edges),
            "centers": edges[:-1] + np.diff(edges) / 2,
            "stat": self.stat,
            "n": int(uids.size),
        }


@dataclass
class ECDF(_DensityTransform):
    """Per-group empirical cumulative density (raw, spline or bootstrap).

    ``unique_id`` nests the group key so each geometry dict is one subject's
    ECDF. With ``unique_id`` + ``agg_func`` (``ecdf_type="spline"`` required)
    every subject is evaluated on a shared probability grid (``size``
    defaults to the largest subject) so the aggregated geometry is the mean
    quantile function — the legacy axis swap: ``x`` holds quantile values,
    ``y`` the shared probabilities.

    Geometry per group/subject: ``{x, y, n}``; two-level per group:
    ``{x, y, error_low, error_high, n}`` (``n`` = subjects, errors per grid
    point).
    """

    name: str = "ecdf"
    ecdf_type: Literal["bootstrap", "spline", "none"] = "none"
    ecdf_args: dict = field(default_factory=dict)

    def _validate(self) -> None:
        super()._validate()
        if self.agg_func is not None and self.ecdf_type != "spline":
            raise ValueError("ecdf_type must be 'spline' when agg_func is given (shared probability grid).")

    def _density_geometry(self, vals: np.ndarray, edges: np.ndarray | None = None) -> dict:
        xv, yv = ecdf(vals, self.ecdf_type, **self.ecdf_args)
        return {"x": xv, "y": yv, "n": int(vals.size)}

    def _aggregate_geometry(
        self,
        vals: np.ndarray,
        uids: np.ndarray,
        group_key: tuple,
        unique_groups: dict[tuple, np.ndarray],
        data: DataHolder,
        y: str,
        ytransform: Transform | None,
        edges: np.ndarray | None,
    ) -> dict:
        # shared probability grid; x holds the aggregated quantile values (legacy axis swap)
        args = dict(self.ecdf_args)
        sub_vals = self._subject_values(group_key, uids, unique_groups, data, y, ytransform)
        size = args.pop("size", max(v.size for v in sub_vals))
        grid_y = np.arange(size) / size
        hold = np.vstack([ecdf(v, "spline", size=size, **args)[1] for v in sub_vals])
        center = np.asarray(get_transform(self.agg_func)(hold, axis=0), dtype=float)
        low, high = self._pair_error(hold)
        return {"x": center, "y": grid_y, "error_low": low, "error_high": high, "n": int(uids.size)}


@dataclass
class Summary(Transform):
    """Compute summary/box-whisker geometry per group.

    Args:
        func: center value of each group (default median).
        err_func: error function applied to the same values as ``func``.
        whisker: what the whisker extent should represent - ``"quantiles"``
            (default), ``"minmax"`` or ``"none"``.
        whisker_quantiles: (low, high) percentiles used by
            ``whisker="quantiles"``.
        notch: whether to compute (approx Gaussian) notch bounds around the
            median (standard boxplot formula).
    """

    name: str = "summary"
    func: Agg = "median"
    err_func: Error = None
    whisker: Literal["quantiles", "minmax", "none"] = "quantiles"
    whisker_quantiles: tuple[float, float] = (5, 95)
    notch: bool = False

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        **kwargs,
    ) -> dict[tuple, dict]:
        if y is None:
            raise ValueError("Summary requires a y column.")
        output = {}
        for group_key, indexes in self._groups(data, levels).items():
            vals = _get_column_values(data, indexes, y, ytransform)
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            median = float(np.percentile(vals, 50))
            mean = float(np.mean(vals))
            center = float(get_transform(self.func)(vals))
            if self.err_func is not None:
                low, high = as_error_pair(get_transform(self.err_func)(vals))
            else:
                low, high = None, None

            if self.whisker == "quantiles":
                wlow_p, whigh_p = self.whisker_quantiles
                whisker_low = float(np.percentile(vals, wlow_p))
                whisker_high = float(np.percentile(vals, whigh_p))
            elif self.whisker == "minmax":
                whisker_low, whisker_high = float(vals.min()), float(vals.max())
            else:
                whisker_low, whisker_high = None, None

            notch_low = notch_high = None
            if self.notch and vals.size > 1:
                iqr = q3 - q1
                notch = 1.58 * iqr / np.sqrt(vals.size)
                notch_low, notch_high = median - notch, median + notch

            output[group_key] = {
                "center": center,
                "mean": mean,
                "median": median,
                "q1": q1,
                "q3": q3,
                "whisker_low": whisker_low,
                "whisker_high": whisker_high,
                "error_low": low,
                "error_high": high,
                "notch_low": notch_low,
                "notch_high": notch_high,
                "n": int(vals.size),
            }
        return output


@dataclass
class Fit(Transform):
    """Fit a function to x/y per group and return the fit line + CI band."""

    name: str = "fit"
    fit_func: FitFunc = "linear"
    ci_func: CIFunc = "ci"
    fit_args: dict = field(default_factory=dict)

    def __call__(
        self,
        data: DataHolder,
        y: str | None = None,
        x: str | None = None,
        levels: Levels = (),
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        **kwargs,
    ) -> dict[tuple, dict]:
        if x is None or y is None:
            raise ValueError("Fit requires both x and y columns.")
        output = {}
        for group_key, indexes in self._groups(data, levels).items():
            xv = _get_column_values(data, indexes, x, xtransform)
            yv = _get_column_values(data, indexes, y, ytransform)
            fit_output = fit(
                self.fit_func,
                x=xv,
                y=yv,
                ci_func=self.ci_func,
                **self.fit_args,
            )
            output[group_key] = {
                "x": fit_output[2],
                "y": fit_output[1],
                "ci": fit_output[3],
                "n": int(xv.size),
            }
        return output
