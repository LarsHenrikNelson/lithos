"""Transform dataclasses for the new plot API.

Transforms are pure statistical operations: grouped data in, *geometry* out.
They have no knowledge of colors, markers or backends, and they own all
aggregation and error computation (see ``doc/migration_plan.md``).

Each transform ``__call__`` returns ``dict[group_key, geometry_dict]`` where
the geometry dict is the only source of numbers that elements render.

Geometry contract per transform:

- ``Identity``: ``{x, y, n}``
- ``Aggregate``: ``{center, error_low, error_high, n}``
- ``Density``: kde/ecdf -> ``{x, y, n}``; hist -> ``{edges, height, binwidth,
  centers, stat, n}``
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
    "Density",
    "Fit",
    "Identity",
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
        **kwargs,
    ) -> dict[tuple, dict]:
        """Compute per-group geometry.

        Args:
            data: the input data.
            y: value column.
            x: independent column (required by ``Identity``/``Fit``).
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
    """Pass raw values through as per-group point geometry."""

    name: str = "identity"

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
            raise ValueError("Identity requires both x and y columns.")
        output = {}
        for group_key, indexes in self._groups(data, levels).items():
            output[group_key] = {
                "x": _get_column_values(data, indexes, x, xtransform),
                "y": _get_column_values(data, indexes, y, ytransform),
                "n": int(indexes.size),
            }
        return output


@dataclass
class Aggregate(Transform):
    """Aggregate y per group (optionally nesting via ``unique_id``).

    Args:
        func: aggregation function applied to y within each group (or within
            each unique_id group first).
        err_func: error function applied to the same values; the result is
            normalized to ``error_low``/``error_high``.
        agg_func: second-level aggregation applied across unique_id samples
            when ``unique_id`` is given (defaults to ``func``).
        unique_id: column whose unique values are first aggregated with
            ``func``, then re-aggregated per group with ``agg_func``.
    """

    name: str = "aggregate"
    func: Agg = "mean"
    err_func: Error = None
    agg_func: Agg | None = None
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
        if y is None:
            raise ValueError("Aggregate requires a y column.")
        output = {}

        if self.unique_id is None:
            for group_key, indexes in self._groups(data, levels).items():
                vals = _get_column_values(data, indexes, y, ytransform)
                center = float(get_transform(self.func)(vals))
                if self.err_func is not None:
                    low, high = as_error_pair(get_transform(self.err_func)(vals))
                else:
                    low, high = None, None
                output[group_key] = {
                    "center": center,
                    "error_low": low,
                    "error_high": high,
                    "n": int(vals.size),
                }
            return output

        # Nested aggregation: first over (levels + unique_id), then over levels.
        sub_levels = tuple(levels) + (self.unique_id,)
        n_levels = len(tuple(levels))
        per_group: dict[tuple, list[np.ndarray]] = defaultdict(list)
        first = get_transform(self.func)
        for sub_key, sub_indexes in self._groups(data, sub_levels).items():
            vals = _get_column_values(data, sub_indexes, y, ytransform)
            per_group[sub_key[:n_levels] if n_levels > 0 else ("",)].append(first(vals))
        second = get_transform(self.agg_func if self.agg_func is not None else self.func)
        for group_key, centers in per_group.items():
            centers = np.asarray(centers, dtype=float)
            center = float(second(centers))
            if self.err_func is not None:
                low, high = as_error_pair(get_transform(self.err_func)(centers))
            else:
                low, high = None, None
            output[group_key] = {
                "center": center,
                "error_low": low,
                "error_high": high,
                "n": int(centers.size),
            }
        return output


@dataclass
class Density(Transform):
    """Compute a 1-D density geometry per group.

    ``kind`` selects the estimator:

    - ``"kde"`` -> smooth density curves via ``KDEpy``;
    - ``"hist"`` -> histogram (edges, heights);
    - ``"ecdf"`` -> empirical cumulative density (raw, spline or bootstrap).
    """

    name: str = "density"
    kind: Literal["kde", "hist", "ecdf"] = "kde"
    # kde
    kernel: Kernels = "gaussian"
    bw: BW = "ISJ"
    tol: float | int | tuple = 1e-3
    kde_length: int | None = None
    KDEType: KDEType = "fft"
    # hist
    bins: NBins = 50
    bin_range: tuple[float, float] | None = None
    stat: HistStat = "density"
    # ecdf
    ecdf_type: Literal["bootstrap", "spline", "none"] = "none"
    ecdf_args: dict = field(default_factory=dict)

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
            raise ValueError("Density requires a y column.")
        output = {}
        for group_key, indexes in self._groups(data, levels).items():
            vals = _get_column_values(data, indexes, y, ytransform)
            if self.kind == "kde":
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
                output[group_key] = {"x": xv, "y": yv, "n": int(vals.size)}
            elif self.kind == "hist":
                edges = np.histogram_bin_edges(vals, bins=self.bins, range=self.bin_range)
                height = hist(vals, edges, self.stat)
                output[group_key] = {
                    "edges": edges,
                    "height": height,
                    "binwidth": np.diff(edges),
                    "centers": edges[:-1] + np.diff(edges) / 2,
                    "stat": self.stat,
                    "n": int(vals.size),
                }
            elif self.kind == "ecdf":
                xv, yv = ecdf(vals, self.ecdf_type, **self.ecdf_args)
                output[group_key] = {"x": xv, "y": yv, "n": int(vals.size)}
            else:
                raise ValueError(f"kind must be 'kde', 'hist' or 'ecdf', got {self.kind!r}.")
        return output


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
