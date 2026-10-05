"""Significance brackets: layout decorations resolved at plot time.

Unlike elements - which only format the per-group geometry a transform
computes - a significance bracket consumes no transform geometry. Its span is
either absolute (``x1``/``x2``) or resolved from the layout's group slot
positions at plot time (``groups``), and its height is computed from the data
so it sits just above whatever is plotted. It is therefore a *layout
decoration* added with :meth:`Plot.add_significance` (the ``add_axline``
family), not an element passed to ``.add()``.

Two bracket styles:

- ``"bracket"`` - the GraphPad mustache: a horizontal line whose caps descend
  to the top of the plotted data under the span (or to a nested bracket's
  line below it).
- ``"line"`` - a plain horizontal line without caps.

Automatic height: brackets with ``y=None`` sit one ``gap`` (a fraction of the
y-range) above the maximum y value under their span, and brackets whose
x-spans overlap stack one ``step`` apart so stacked comparisons never collide.
An explicit ``y`` disables both behaviors for that bracket.

Brackets render inside ``SpecPlotter._plot()``, i.e. *before*
``format_plot()`` autoscales the axes, so the axis limits expand to include
them (unless an explicit ``ylim`` is set on the plot).
"""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from ...types.basic_types import Transform
from ...utils import DataHolder, get_transform

BRACKET_STYLES = ("bracket", "line")


@dataclass
class Significance:
    """One significance bracket: text, span, height and styling.

    ``groups`` entries are plain group values (e.g. ``0`` or ``"ctrl"``) when
    the plot groups only, and ``(group, subgroup)`` tuples when it also
    subgroups; a plain group value on a subgrouped plot spans the whole
    cluster. Two entries bracket the pair; three or more span from the
    leftmost to the rightmost slot (a main-effect bracket). On continuous
    layouts a single group key selects the facet, and ``x1``/``x2`` (optional)
    restrict the span to that x-window.
    """

    text: str = "*"
    groups: list[str | int | tuple] | None = None
    x1: float | None = None
    x2: float | None = None
    y: float | None = None
    style: Literal["bracket", "line"] = "bracket"
    gap: float = 0.02
    step: float = 0.05
    linecolor: str = "black"
    linewidth: float = 1.5
    fontsize: float = 12
    zorder: int | float | None = None

    def to_spec(self) -> dict:
        """Return a plain dict suitable for metadata/JSON export."""
        return asdict(self)


def _group_key(key) -> tuple:
    """Normalize one ``groups`` entry to a loc-dict-style key tuple."""
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)


def _column_values(data: DataHolder, indexes: np.ndarray, column: str, transform: Transform | None) -> np.ndarray:
    """Extract (and optionally transform) a column for a set of row indexes."""
    vals = np.asarray(data[indexes, column], dtype=float)
    return np.asarray(get_transform(transform)(vals), dtype=float)


def _slot_positions(key: tuple, loc_dict: dict) -> list[float]:
    """Axis positions for one group key - exact, or all slots it prefixes."""
    if key in loc_dict:
        return [loc_dict[key]]
    positions = [pos for slot, pos in loc_dict.items() if slot[: len(key)] == key]
    if positions:
        return positions
    raise ValueError(f"No slot found for group key {key!r}; available keys: {sorted(loc_dict, key=str)!r}.")


def _key_rows(key: tuple, groups_index: dict) -> np.ndarray:
    """Row indexes for one group key - exact, or all groups it prefixes."""
    if key in groups_index:
        return np.asarray(groups_index[key])
    rows = [np.asarray(index) for gkey, index in groups_index.items() if gkey[: len(key)] == key]
    if rows:
        return np.concatenate(rows)
    raise ValueError(f"No data rows found for group key {key!r}.")


def _union_rows(keys: list[tuple], groups_index: dict) -> np.ndarray:
    """Row indexes covered by a set of group keys."""
    rows = [_key_rows(key, groups_index) for key in keys]
    return rows[0] if len(rows) == 1 else np.concatenate(rows)


def _resolve_span(
    sig: Significance,
    data: DataHolder,
    x: str | None,
    context: dict,
    groups_index: dict,
) -> tuple[float, float, int, np.ndarray]:
    """Resolve one bracket's span into (x1, x2, facet, rows under the span)."""
    keys = [_group_key(key) for key in sig.groups] if sig.groups is not None else []

    if context["layout"] == "categorical":
        if keys:
            positions = [pos for key in keys for pos in _slot_positions(key, context["loc_dict"])]
            return min(positions), max(positions), 0, _union_rows(keys, groups_index)
        # absolute positions: cover the slots that fall inside the window
        in_window = [slot for slot, pos in context["loc_dict"].items() if sig.x1 <= pos <= sig.x2]
        rows = _union_rows(in_window, groups_index) if in_window else np.arange(data.shape[0])
        return sig.x1, sig.x2, 0, rows

    # continuous layout: loc_dict maps group keys to facet (axes) indices
    if keys:
        if len(keys) != 1:
            raise ValueError("Significance groups on a continuous layout select one facet; pass a single group key.")
        facet = int(_slot_positions(keys[0], context["loc_dict"])[0])
        rows = _key_rows(keys[0], groups_index)
    else:
        facet, rows = 0, np.arange(data.shape[0])

    if sig.x1 is not None:
        if x is not None:
            xvals = np.asarray(data[rows, x], dtype=float)
            rows = rows[(xvals >= sig.x1) & (xvals <= sig.x2)]
        return sig.x1, sig.x2, facet, rows

    # no explicit window: span the full x-range of the rows under the bracket
    if x is None:
        raise ValueError("Significance on a continuous layout needs x1/x2 or the x column for the full span.")
    xvals = np.asarray(data[rows, x], dtype=float)
    return float(np.nanmin(xvals)), float(np.nanmax(xvals)), facet, rows


def _resolve_bracket(
    sig: Significance,
    data: DataHolder,
    y: str,
    x: str | None,
    context: dict,
    groups_index: dict,
    ytransform: Transform | None,
    y_span: float,
    earlier: list[dict],
) -> dict:
    """Resolve one bracket into a concrete, renderer-ready dict."""
    x1, x2, facet, rows = _resolve_span(sig, data, x, context, groups_index)
    same_facet = [bracket for bracket in earlier if bracket["facet"] == facet]
    top = float(np.nanmax(_column_values(data, rows, y, ytransform)))

    if sig.y is None:
        overlapping = [
            bracket["level"] + 1
            for bracket in same_facet
            if bracket["auto"] and bracket["x1"] <= x2 and bracket["x2"] >= x1
        ]
        level = max(overlapping, default=0)
        y_value = top + sig.gap * y_span + level * sig.step * y_span
    else:
        level, y_value = None, float(sig.y)

    cap_bottoms = None
    if sig.style == "bracket":
        # caps descend to the top of what is plotted: the data under the
        # span, or a nested lower bracket's line covering the cap position.
        cap_bottoms = []
        for cap_x in (x1, x2):
            bottom = top
            for bracket in same_facet:
                if bracket["y"] <= y_value and bracket["x1"] <= cap_x <= bracket["x2"]:
                    bottom = max(bottom, bracket["y"])
            cap_bottoms.append(min(bottom, y_value))

    return {
        "text": sig.text,
        "x1": float(x1),
        "x2": float(x2),
        "y": float(y_value),
        "level": level,
        "auto": sig.y is None,
        "cap_bottoms": cap_bottoms,
        "style": sig.style,
        "facet": facet,
        "linecolor": sig.linecolor,
        "linewidth": sig.linewidth,
        "fontsize": sig.fontsize,
        "zorder": sig.zorder,
    }


def resolve_significance(
    significances: list[Significance],
    data: DataHolder,
    y: str | None,
    x: str | None,
    context: dict,
    ytransform: Transform | None = None,
) -> list[dict]:
    """Resolve every bracket against the data and the layout context.

    This is the significance counterpart of the position resolver: it runs at
    plot time (so ``.grouping()`` calls made after ``add_significance()`` are
    honored) and turns pure bracket specs into concrete dicts the plotter
    renders. No bracket geometry is stored in the metadata - like layer
    geometry, it is recomputed on load.
    """
    if not significances:
        return []

    if any(sig.y is None or sig.style == "bracket" for sig in significances) and y is None:
        raise ValueError(
            "Significance auto-y and bracket caps need the y column; pass y to plot() or set the bracket y explicitly "
            "with style='line'."
        )
    groups_index = data.groups(context["levels"])

    # gap/step are fractions of the full (transformed) y-range
    all_vals = _column_values(data, np.arange(data.shape[0]), y, ytransform)
    y_span = float(np.nanmax(all_vals) - np.nanmin(all_vals))
    if y_span == 0:
        y_span = 1.0

    resolved: list[dict] = []
    for sig in significances:
        resolved.append(_resolve_bracket(sig, data, y, x, context, groups_index, ytransform, y_span, resolved))
    return resolved
