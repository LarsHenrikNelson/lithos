"""Position resolution for the spec plot API.

The resolver interprets the transform geometry computed by ``Plot._process_data()``
onto axis positions using the layout context produced by the plot classes.
Transforms stay layout-agnostic: the same ``dict[group_key, geometry]``
renders differently per layout.

Position modes (per ``.add()`` layer):

- ``passthrough``: the transform must supply the coordinate itself
  (continuous default).
- ``dodge``: categorical slot positions from the layout ``loc_dict``. The
  layer's ``width`` (a fraction of the slot) sets the spread within the slot:

  - ``width=0``: every point at the slot center (summary/bar positions).
  - ``width > 0`` with flat geometry: random jitter within ``width``, shaped
    by ``jitter_type`` and reproducible via ``seed`` (legacy ``jitter``).
  - ``width > 0`` with ``unique_id``-nested geometry: one even column per
    subject across the width — deterministic, no randomness (legacy
    ``jitteru``/``summaryu``).
"""

from collections import defaultdict

import numpy as np

from ..plot_utils import process_jitter


def locate_key(key: tuple, positions: dict) -> float:
    """Return the position for a (possibly nested) group key via longest-prefix match.

    Geometry keys may be longer than the grouping levels (e.g. nested by
    ``unique_id``); the resolver matches the group prefix. Keys that match no
    prefix fall back to the single ``("",)`` slot of ungrouped layouts.
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
    """The layout slot a (possibly nested) geometry key resolves to."""
    if key in positions:
        return key
    for size in range(len(key) - 1, 0, -1):
        if key[:size] in positions:
            return key[:size]
    if ("",) in positions:
        return ("",)
    raise KeyError(f"No position found for group key {key!r}.")


def _as_values(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def _layer_unique_id(layer: dict):
    """The ``unique_id`` column a layer's transform nests its geometry by."""
    transform = layer.get("transform")
    if isinstance(transform, dict):
        return transform.get("unique_id")
    return getattr(transform, "unique_id", None)


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


def _resolved_coordinates(
    layer: dict, context: dict, position: float, values, column: float | None = None
) -> np.ndarray:
    """Positions for the missing axis, given the layer's position mode."""
    if layer["position"] == "dodge":
        if column is not None:
            # unique_id-nested geometry: the precomputed even column
            return np.full(_as_values(values).size, column)
        spread = layer.get("width", 0.0) * context["width"]
        if spread == 0:
            return np.full(_as_values(values).size, position)
        return process_jitter(
            _as_values(values),
            position,
            spread,
            seed=layer.get("seed", 42),
            jitter_type=layer.get("jitter_type", "fill"),
        )
    raise ValueError(
        f"position={layer['position']!r} cannot supply a missing coordinate; "
        "use a transform that provides it or the dodge position."
    )


def resolve_layer(layer: dict, context: dict) -> dict:
    """Resolve one layer's processed geometry against the layout context."""
    geometry = {}
    categorical = context["layout"] == "categorical"
    unique_id = _layer_unique_id(layer)
    columns = (
        _even_columns(layer["geometry"], context, layer.get("width", 0.0) * context.get("width", 1.0))
        if categorical and unique_id is not None
        else {}
    )
    for key, group_geometry in layer["geometry"].items():
        resolved = dict(group_geometry)
        position = locate_key(key, context["loc_dict"])
        resolved["position"] = position
        # continuous layouts use loc_dict for facet (axes) indices
        resolved["facet"] = int(position) if context["layout"] == "continuous" else 0

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
            if "x" not in resolved and ("y" in resolved or "center" in resolved):
                # vertical layouts: supply x positions from the group location
                resolved["x"] = _resolved_coordinates(
                    layer, context, position, resolved.get("y", resolved.get("center")), column=column
                )
            elif "y" not in resolved and "x" in resolved:
                # horizontal layouts: supply y positions from the group location
                resolved["y"] = _resolved_coordinates(layer, context, position, resolved["x"], column=column)
        geometry[key] = resolved
    return {**layer, "geometry": geometry}


def resolve_layers(layers: list[dict], context: dict) -> list[dict]:
    """Resolve every layer of a plot against the layout context."""
    return [resolve_layer(layer, context) for layer in layers]
