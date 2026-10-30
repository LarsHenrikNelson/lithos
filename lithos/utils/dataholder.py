"""Normalized, backend-agnostic tabular data container.

``DataHolder`` is the data-access contract consumed by the processors and
transforms: it accepts a ``dict`` of columns, a 1-D/2-D ``numpy.ndarray``
(columns are addressed positionally as integers ``0..n-1``), or a pandas
``Series``/``DataFrame`` and normalizes everything into a single internal
``DataFrame`` once at construction. All lookups (``__getitem__``,
``__contains__``, ``groups``, ``groupby``, ``min``/``max``, ``size``/``shape``)
operate on that frame, making pandas an implementation detail that a future
backend could replace by reimplementing these methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..types.basic_types import InputData


class DataHolder:
    """Container that normalizes dict/numpy/pandas input into one table.

    Indexing semantics:

    - ``holder[column]`` -> column by label (``str`` or ``int``).
    - ``holder[rows, column]`` -> positional rows (int/bool arrays, lists or
      slices) paired with a single column label or a sequence of labels (a
      sequence returns a ``DataFrame``). Rows are always positional because
      ``groups()`` returns positional row indexes.

    Notes:
        - The input is never mutated; a caller's dict/DataFrame is left
          untouched and a pandas ``DataFrame`` input is reused without copying.
        - ``groups(())`` returns the sentinel key ``("",)`` mapping to every
          row (the "no grouping" case).
        - NaN group keys are dropped, mirroring the pandas ``groupby`` default.
        - ``to_pd()`` returns the internal frame without copying; callers must
          not mutate it.
    """

    def __init__(self, data: InputData | pd.Series | DataHolder) -> None:
        self._df = self._normalize(data)
        self._groups_cache: dict[tuple, dict[tuple, np.ndarray]] = {}

    @staticmethod
    def _normalize(data: InputData | pd.Series | DataHolder) -> pd.DataFrame:
        """Convert any accepted input into one canonical ``DataFrame``.

        Raises:
            TypeError: If ``data`` is not a dict, numpy array, pandas
                Series/DataFrame, or another ``DataHolder``.
            ValueError: If a dict is empty or has columns of unequal length,
                or if a numpy array is not 1-D/2-D.
        """
        if isinstance(data, DataHolder):
            return data._df
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.ndim != 2:
                raise ValueError(f"numpy input must be 1-D or 2-D, got {data.ndim}-D.")
            return pd.DataFrame(data)  # single-dtype 2-D arrays are wrapped without copying
        if isinstance(data, dict):
            if not data:
                raise ValueError("dict input must contain at least one column.")
            try:
                return pd.DataFrame(data)
            except ValueError as err:
                raise ValueError(f"dict columns must all have the same length: {err}") from err
        raise TypeError("Only numpy arrays, dict, or pandas DataFrames/Series are accepted.")

    def __contains__(self, item: object) -> bool:
        """Return whether ``item`` is a column label of the data."""
        try:
            return item in self._df.columns
        except TypeError:  # unhashable item
            return False

    def _column_locs(self, columns: Sequence | str | int) -> int | np.ndarray:
        """Resolve column label(s) to positional column indexes."""
        is_sequence = isinstance(columns, (list, tuple, np.ndarray, pd.Index))
        labels = columns if is_sequence else [columns]
        locs = self._df.columns.get_indexer(labels)
        missing = [label for label, loc in zip(labels, locs) if loc < 0]
        if missing:
            raise KeyError(f"Columns {missing} not found in data columns {list(self._df.columns)}.")
        if not is_sequence:
            return locs[0]
        return locs

    def __getitem__(self, index: str | int | tuple | np.integer) -> Any:
        """Index the data by column label, or by ``(rows, column(s))`` positionally."""
        if isinstance(index, tuple):
            if len(index) != 2:
                raise TypeError(f"tuple indexing must be (rows, column), got {len(index)} items.")
            rows, columns = index
            return self._df.iloc[rows, self._column_locs(columns)]
        if isinstance(index, (str, int, np.integer)):
            if index not in self:
                raise KeyError(f"Column {index!r} not found in data columns {list(self._df.columns)}.")
            return self._df[index]
        raise TypeError(f"Invalid index type {type(index).__name__}; use a column label or (rows, column).")

    def min(self, column: str | int) -> float | int:
        """Return the minimum value of ``column``."""
        return np.min(self[column])

    def max(self, column: str | int) -> float | int:
        """Return the maximum value of ``column``."""
        return np.max(self[column])

    @property
    def size(self) -> int:
        """Total number of cells (rows * columns)."""
        return self._df.size

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the underlying table."""
        return self._df.shape

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._df.shape[0]

    def groups(self, levels: tuple) -> dict[tuple, np.ndarray]:
        """Return ``{group_key: positional row indexes}`` for the grouping ``levels``.

        Cached per ``levels``. Empty ``levels`` returns the sentinel key
        ``("",)`` mapping to every row. NaN group keys are dropped (pandas
        ``groupby`` default).
        """
        if levels in self._groups_cache:
            return self._groups_cache[levels]
        if len(levels) == 0:
            new_groups = {("",): np.arange(self._df.shape[0])}
        else:
            new_groups = {}
            for key, indexes in self._df.groupby(list(levels)).indices.items():
                new_groups[key if isinstance(key, tuple) else (key,)] = indexes
        self._groups_cache[levels] = new_groups
        return new_groups

    def groupby(self, values: str | int | tuple, by: tuple | Sequence, sort: bool = True) -> Any:
        """Return a pandas ``GroupBy`` of the ``by`` columns, keeping ``by + values`` columns.

        Args:
            values (str | int | tuple): Column(s) to aggregate later.
            by (tuple | Sequence): Columns to group on.
            sort (bool): Sort the group keys.

        Returns:
            Any: A pandas ``DataFrameGroupBy`` (``as_index=False``). Built
            fresh on each call — pandas ``GroupBy`` objects are stateful after
            ``.agg()`` so they are not cached.
        """
        if not isinstance(values, tuple):
            values = (values,)
        by = tuple(by)
        frame = self._df[list(by) + list(values)]
        return frame.groupby(list(by), sort=sort, as_index=False)

    def to_pd(self) -> pd.DataFrame:
        """Return the internal canonical ``DataFrame`` (not a copy — do not mutate)."""
        return self._df
