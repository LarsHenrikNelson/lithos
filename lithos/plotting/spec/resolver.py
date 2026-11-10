"""Position resolution for the spec plot API.

The resolver interprets the transform geometry computed by ``Plot._process_data()``
onto axis positions using the layout context produced by the plot classes.
Transforms stay layout-agnostic: the same ``dict[group_key, geometry]``
renders differently per layout.

Position modes (per ``.add()`` layer):

- ``passthrough``: the transform must supply the coordinate itself
  (continuous default).
- ``dodge``: categorical slot positions from the layout ``loc_dict``. The
  layer's ``width`` (a fraction of the slot, default 0.9 from ``.add()``)
  sets the spread within the slot:

  - ``width=0``: every point at the slot center (summary/bar positions).
  - ``width > 0`` with flat geometry: random jitter within ``width``, shaped
    by ``jitter_type`` and reproducible via ``seed`` (legacy ``jitter``).
    Geometry holding a single value per group key (aggregate centers)
    stays anchored at the slot center instead — legacy ``summary`` lines
    and error bars are never jittered.
  - ``width > 0`` with ``unique_id``-nested geometry: one even column per
    subject across the width — deterministic, no randomness (legacy
    ``jitteru``/``summaryu``).

Density orientation: ``KDE``/``Histogram`` layers carry a resolved
``value_axis`` (the axis of the layer's value column — ``y`` or ``x``);
renderers place the values on that axis so horizontal and vertical density
plots work like ``Identity``/``Summary`` horizontal layouts. On categorical
layouts a ``KDE`` layer resolves to *violin* geometry instead: values on the
value axis, the density normalized to the layer ``width`` around the slot
position (``unique_id``-nested keys: around their even column).

Layer form styles (``.add(..., style=...)``): forms that need cross-group
information are resolved here - never computed by transforms (per-group by
contract) or elements (pure per-artist formatting). ``left``/``right``/
``alternate`` turn a violin's symmetric ``spread`` into one-sided
``spread_low``/``spread_high`` bounds (the flat edge on the slot boundary,
legacy ``violin`` styles; ``alternate`` picks the side by slot order);
``overlap`` places ``unique_id``-nested keys on the slot center at full
extent (legacy violin ``unique_style="overlap"``); ``stack``/``fill`` give
histogram geometry cumulative ``bottom`` offsets in geometry order per facet
(``fill`` first normalizes heights by the per-bin total - legacy
``hist_type`` stack modes).
"""

from collections import defaultdict

import numpy as np

from ..plot_utils import process_jitter


def locate_key(key: tuple, positions: dict) -> float:
    """Return the position for a (possibly nested) group key via longest-prefix match.

    Geometry keys may be longer than the grouping levels (e.g. nested by
    ``unique_id``); the resolver matches the group prefix. Keys that match no
    prefix fall back to the single ``("",)`` slot of ungrouped layouts.

    Args:
        key (tuple): Geometry group key (possibly nested beyond the grouping
            levels).
        positions (dict): The layout ``loc_dict`` mapping slot keys to axis
            positions.

    Returns:
        float: The resolved axis position.

    Raises:
        KeyError: If no slot matches the key.
    """
    if key in positions:
        return positions[key]
    for size in range(len(key) - 1, 0, -1):
        if key[:size] in positions:
            return positions[key[:size]]
    if ("",) in positions:
        return positions[("",)]
    raise KeyError(f"No position found for group key {key!r}.")


def slot_key(key: tuple, positions: dict) -> tuple:
    """The layout slot a (possibly nested) geometry key resolves to.

    Args:
        key (tuple): Geometry group key (possibly nested beyond the grouping
            levels).
        positions (dict): The layout ``loc_dict`` mapping slot keys to axis
            positions.

    Returns:
        tuple: The slot key the geometry key resolves to.

    Raises:
        KeyError: If no slot matches the key.
    """
    if key in positions:
        return key
    for size in range(len(key) - 1, 0, -1):
        if key[:size] in positions:
            return key[:size]
    if ("",) in positions:
        return ("",)
    raise KeyError(f"No position found for group key {key!r}.")


def _as_values(values) -> np.ndarray:
    """Coerce values to a flat float array."""
    return np.atleast_1d(np.asarray(values, dtype=float))


def _layer_unique_id(layer: dict):
    """The ``unique_id`` column a layer's transform nests its geometry by."""
    transform = layer.get("transform")
    if isinstance(transform, dict):
        return transform.get("unique_id")
    return getattr(transform, "unique_id", None)


def _layer_transform_name(layer: dict) -> str | None:
    """The transform name of a layer spec (``"kde"``, ``"histogram"``, ...)."""
    transform = layer.get("transform")
    if isinstance(transform, dict):
        return transform.get("name")
    return getattr(transform, "name", None)


def _layer_value_axis(layer: dict) -> str | None:
    """The value axis of a density layer: ``"y"`` or ``"x"``; ``None`` otherwise.

    Density geometry is canonical (``x`` holds the evaluated values, ``y``
    the density); the value column's axis decides the render orientation.
    ``ECDF`` stays canonical (values on x, probabilities on y) and therefore
    has no value axis.
    """
    if _layer_transform_name(layer) not in ("kde", "histogram"):
        return None
    return "y" if layer.get("y") is not None else "x"


def _even_columns(keys, context: dict, spread: float) -> dict:
    """Even column position and per-column extent for ``unique_id``-nested keys.

    Keys nested beyond the grouping levels share one layout slot; they are
    placed on ``n`` evenly spaced columns across ``spread`` axis units
    (legacy ``jitteru``/``summaryu``), with ``spread / n`` per column for
    element widths. A single subject sits at the slot center spanning the
    full spread.
    """
    by_slot: dict[tuple, list] = defaultdict(list)
    for key in keys:
        by_slot[slot_key(key, context["loc_dict"])].append(key)

    output = {}
    for keys_in_slot in by_slot.values():
        center = context["loc_dict"][slot_key(keys_in_slot[0], context["loc_dict"])]
        ordered = sorted(keys_in_slot)
        n = len(ordered)
        if n > 1:
            columns = np.linspace(center - spread / 2, center + spread / 2, num=n * 2 + 1)[1::2]
            extent = spread / n
        else:
            columns = np.array([center])
            extent = spread
        for uid_index, (key, column) in enumerate(zip(ordered, columns)):
            output[key] = (float(column), float(extent), uid_index)
    return output


def _overlap_columns(keys, context: dict) -> dict:
    """Slot-center placement for ``unique_id``-nested keys (legacy violin ``overlap``).

    Every nested key in a slot resolves to the slot center with the full slot
    extent, so per-subject densities overlay at full width (legacy
    ``unique_style="overlap"`` violins); the index keeps ``uid_index``-based
    element cycling (marker lists) working.

    Args:
        keys (list): Geometry keys nested beyond the grouping levels.
        context (dict): The layout context.

    Returns:
        dict: ``{key: (center, extent, uid_index)}`` for every nested key.
    """
    by_slot: dict[tuple, list] = defaultdict(list)
    for key in keys:
        by_slot[slot_key(key, context["loc_dict"])].append(key)

    output = {}
    for keys_in_slot in by_slot.values():
        center = context["loc_dict"][slot_key(keys_in_slot[0], context["loc_dict"])]
        extent = float(context["width"])
        for uid_index, key in enumerate(sorted(keys_in_slot)):
            output[key] = (float(center), extent, uid_index)
    return output


def _resolved_coordinates(
    layer: dict, context: dict, position: float, values, column: float | None = None
) -> np.ndarray:
    """Positions for the missing axis, given the layer's position mode.

    Args:
        layer (dict): The processed layer spec (``position``, ``width``,
            ``jitter_type``, ``seed``).
        context (dict): The layout context.
        position (float): The group's resolved slot position.
        values: The geometry values along the other axis.
        column (float | None): Precomputed even-column position for
            ``unique_id``-nested geometry.

    Returns:
        np.ndarray: Positions for the missing axis.

    Raises:
        ValueError: If the position mode cannot supply a missing coordinate.
    """
    if layer["position"] == "dodge":
        values = _as_values(values)
        if column is not None:
            # unique_id-nested geometry: the precomputed even column
            return np.full(values.size, column)
        spread = layer.get("width", 0.0) * context["width"]
        # width=0, or single-value-per-key geometry (aggregate centers),
        # sits at the slot center — legacy summary lines/error bars are
        # never jittered, whatever the layer width.
        if spread == 0 or values.size == 1:
            return np.full(values.size, position)
        return process_jitter(
            values,
            position,
            spread,
            seed=layer.get("seed", 42),
            jitter_type=layer.get("jitter_type", "fill"),
        )
    raise ValueError(
        f"position={layer['position']!r} cannot supply a missing coordinate; "
        "use a transform that provides it or the dodge position."
    )


def _violin_geometry(resolved: dict, value_axis: str, width: float) -> dict:
    """Slot-relative violin geometry for one categorical ``KDE`` group.

    The canonical geometry (``x`` = values, ``y`` = density) becomes values
    on the value axis plus a normalized ``spread`` around the slot position
    on the other axis (legacy ``violin`` semantics): the density is scaled
    so its maximum reaches ``width`` x the resolved extent (half-spread
    ``width * extent / 2``).
    """
    values = _as_values(resolved.pop("x"))
    density = _as_values(resolved.pop("y"))
    dmax = float(density.max()) if density.size else 0.0
    half = width * resolved.get("extent", 1.0) / 2
    resolved["spread"] = density / dmax * half if dmax > 0 else np.zeros_like(density)
    resolved[value_axis] = values
    return resolved


def _violin_side_bounds(spread: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) offsets from the slot position for a one-sided violin.

    ``"left"``/``"right"`` draw one half spanning the same total extent as
    the full violin, with the flat edge on the opposite slot boundary
    (legacy ``violin`` styles): ``left`` -> ``[h - 2*spread, h]`` and
    ``right`` -> ``[-h, -h + 2*spread]`` with ``h`` the maximum spread.

    Args:
        spread (np.ndarray): The symmetric violin spread (max ``h``).
        side (str): ``"left"`` or ``"right"``.

    Returns:
        tuple[np.ndarray, np.ndarray]: The ``(spread_low, spread_high)``
        offsets from the slot position.
    """
    spread = _as_values(spread)
    h = float(spread.max()) if spread.size else 0.0
    if side == "left":
        return h - 2.0 * spread, np.full_like(spread, h)
    return np.full_like(spread, -h), 2.0 * spread - h


def _stack_density(geometry: dict, style: str) -> None:
    """Add cumulative stacking to resolved histogram geometry (in place).

    Groups stack in geometry order within each facet - the legacy
    ``_post_process_density`` row order. ``style="fill"`` first normalizes
    every height by the per-bin total across the facet (100% stacked, legacy
    ``hist_type="fill"``); ``style="stack"`` keeps the heights and only adds
    the offsets (legacy ``hist_type="stack"``). Each resolved geometry gains
    a ``bottom`` array (the exclusive cumulative sum before it, zeroed where
    its own height is zero, matching the legacy quirks exactly).

    Args:
        geometry (dict): The resolved geometry dict of one layer.
        style (str): ``"stack"`` or ``"fill"``.

    Raises:
        ValueError: If the layer carries error geometry (stacked errors are
            ill-defined) or the bins do not share edges within a facet
            (per-group auto ranges; pass ``bin_limits="common"``).
    """
    if any(resolved.get("error_low") is not None for resolved in geometry.values()):
        raise ValueError("style='stack' cannot render error geometry; drop err_func from the Histogram transform.")

    proportion = style == "fill"
    by_facet: dict[int, list] = defaultdict(list)
    for key, resolved in geometry.items():
        by_facet[resolved.get("facet", 0)].append(key)

    for keys in by_facet.values():
        raw = np.vstack([np.atleast_1d(np.asarray(geometry[key]["height"], dtype=float)) for key in keys])
        edges = [np.asarray(geometry[key]["edges"], dtype=float) for key in keys]
        if any(edge.shape != edges[0].shape or not np.array_equal(edge, edges[0]) for edge in edges[1:]):
            raise ValueError("style='stack' requires shared bin edges; pass bin_limits='common' to Histogram.")
        if proportion:
            totals = raw.sum(axis=0)
            heights = np.divide(raw, totals, where=totals > 0)
            heights[raw == 0] = 0.0
        else:
            heights = raw
        bottoms = np.zeros_like(heights)
        bottoms[1:, :] = heights.cumsum(axis=0)[:-1, :]
        bottoms[heights == 0] = 0.0
        for key, height, bottom in zip(keys, heights, bottoms):
            geometry[key]["height"] = height
            geometry[key]["bottom"] = bottom


def resolve_layer(layer: dict, context: dict) -> dict:
    """Resolve one layer's processed geometry against the layout context.

    Args:
        layer (dict): A processed layer (transform spec, element specs and
            fresh geometry).
        context (dict): The layout context produced by the plot class.

    Returns:
        dict: The layer with per-key positions, facets, extents and any
        resolver-supplied missing-axis coordinates. Density layers
        (``KDE``/``Histogram``) additionally carry ``value_axis``; a
        categorical ``KDE`` layer is resolved to violin geometry (values on
        the value axis, ``spread`` around the slot) and a categorical
        ``Summary`` (box) layer carries ``span`` (the box body width: the
        layer ``width`` x the key's extent) plus ``value_axis``. Layer
        styles add form fields: one-sided violins carry
        ``spread_low``/``spread_high``, stacked histograms carry ``bottom``
        (``"fill"`` also normalizes ``height``).
    """
    geometry = {}
    categorical = context["layout"] == "categorical"
    unique_id = _layer_unique_id(layer)
    value_axis = _layer_value_axis(layer)
    violin = categorical and _layer_transform_name(layer) == "kde"
    summary = categorical and _layer_transform_name(layer) == "summary"
    if summary and layer.get("y") is None:
        raise NotImplementedError("Summary (box) layers need the y value column; horizontal boxes are pending.")
    style = layer.get("style")
    # even columns apply only to keys nested beyond the grouping levels
    # (per-subject geometry); aggregated group-level keys resolve like flat keys
    levels_size = len(context.get("levels", ()))
    nested_keys = (
        [key for key in layer["geometry"] if len(key) > levels_size] if categorical and unique_id is not None else []
    )
    if nested_keys and style == "overlap":
        # legacy violin unique_style="overlap": subjects overlay on the slot center
        columns = _overlap_columns(nested_keys, context)
    elif nested_keys:
        columns = _even_columns(nested_keys, context, layer.get("width", 0.0) * context.get("width", 1.0))
    else:
        columns = {}
    violin_sides = {}
    if violin and style == "alternate":
        # the side alternates by slot order (the legacy draw-order flip)
        ordered = sorted(layer["geometry"], key=lambda key: (locate_key(key, context["loc_dict"]), key))
        violin_sides = {key: ("left" if index % 2 == 0 else "right") for index, key in enumerate(ordered)}
    for key, group_geometry in layer["geometry"].items():
        resolved = dict(group_geometry)
        position = locate_key(key, context["loc_dict"])
        resolved["position"] = position
        # continuous layouts use loc_dict for facet (axes) indices
        resolved["facet"] = int(position) if context["layout"] == "continuous" else 0
        if value_axis is not None:
            # density layers: the axis holding the values (render orientation)
            resolved["value_axis"] = value_axis

        if categorical:
            if key in columns:
                # unique_id-nested geometry: even columns + per-column extent
                column, extent, uid_index = columns[key]
                resolved["extent"] = extent
                resolved["uid"] = key[-1]
                resolved["uid_index"] = uid_index
            else:
                resolved["extent"] = context["width"]
                column = None
            if summary:
                # box layers carry the body span (layer width x extent) and a
                # value axis like density layers: the values sit on y
                resolved["span"] = layer.get("width", 0.9) * resolved["extent"]
                resolved["value_axis"] = "y"
            if violin:
                if value_axis is None:
                    raise ValueError("KDE layers must carry a value axis to render as violins.")
                # KDE on a categorical layout renders as a violin: values on
                # the value axis, the density normalized to the layer width
                # around the slot (unique_id-nested keys: their column)
                resolved = _violin_geometry(resolved, value_axis, layer.get("width", 0.9))
                side = violin_sides.get(key, style)
                if side in ("left", "right"):
                    # one-sided violin (legacy style="left"/"right"/"alternate")
                    resolved["spread_low"], resolved["spread_high"] = _violin_side_bounds(resolved["spread"], side)
            elif "x" not in resolved and ("y" in resolved or "center" in resolved):
                # vertical layouts: supply x positions from the group location
                resolved["x"] = _resolved_coordinates(
                    layer, context, position, resolved.get("y", resolved.get("center")), column=column
                )
            elif "y" not in resolved and "x" in resolved:
                # horizontal layouts: supply y positions from the group location
                resolved["y"] = _resolved_coordinates(layer, context, position, resolved["x"], column=column)
        geometry[key] = resolved
    if style in ("stack", "fill"):
        # histogram stacking: cumulative bottoms in geometry order per facet
        _stack_density(geometry, style)
    return {**layer, "geometry": geometry}


def resolve_layers(layers: list[dict], context: dict) -> list[dict]:
    """Resolve every layer of a plot against the layout context.

    Args:
        layers (list[dict]): The processed layers.
        context (dict): The layout context produced by the plot class.

    Returns:
        list[dict]: The resolved layers, ready for the plotter.
    """
    return [resolve_layer(layer, context) for layer in layers]
