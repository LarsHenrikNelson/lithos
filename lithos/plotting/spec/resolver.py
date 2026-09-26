"""Position resolution for the spec plot API.

The resolver interprets the transform geometry computed by ``Plot._process_data()``
onto axis positions using the layout context produced by the plot classes.
Transforms stay layout-agnostic: the same ``dict[group_key, geometry]``
renders differently per layout.

Position modes (per ``.add()`` layer):

- ``passthrough``: the transform must supply the coordinate itself
  (continuous default).
- ``dodge``: one axis position per group key from the layout ``loc_dict``
  (categorical default).
- ``jitter``: legacy jitter within the group's layout width.
"""

import numpy as np

from ..plot_utils import process_jitter


def locate_key(key: tuple, positions: dict) -> float:
    """Return the position for a (possibly nested) group key via longest-prefix match.

    Geometry keys may be longer than the grouping levels (e.g. nested by
    ``unique_id``); the resolver matches the group prefix.
    """
    if key in positions:
        return positions[key]
    for size in range(len(key) - 1, 0, -1):
        if key[:size] in positions:
            return positions[key[:size]]
    raise KeyError(f"No position found for group key {key!r}.")


def _as_values(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def _resolved_coordinates(layer: dict, context: dict, position: float, values) -> np.ndarray:
    """Positions for the missing axis, given the layer's position mode."""
    if layer["position"] == "jitter":
        return process_jitter(_as_values(values), position, context["width"], seed=layer.get("seed", 42))
    if layer["position"] == "dodge":
        return np.full(_as_values(values).size, position)
    raise ValueError(
        f"position={layer['position']!r} cannot supply a missing coordinate; "
        "use a transform that provides it or a dodge/jitter position."
    )


def resolve_layer(layer: dict, context: dict) -> dict:
    """Resolve one layer's processed geometry against the layout context."""
    geometry = {}
    for key, group_geometry in layer["geometry"].items():
        resolved = dict(group_geometry)
        position = locate_key(key, context["loc_dict"])
        resolved["position"] = position
        # continuous layouts use loc_dict for facet (axes) indices
        resolved["facet"] = int(position) if context["layout"] == "continuous" else 0

        if context["layout"] == "categorical":
            if "x" not in resolved and ("y" in resolved or "center" in resolved):
                # vertical layouts: supply x positions from the group location
                resolved["x"] = _resolved_coordinates(
                    layer, context, position, resolved.get("y", resolved.get("center"))
                )
            elif "y" not in resolved and "x" in resolved:
                # horizontal layouts: supply y positions from the group location
                resolved["y"] = _resolved_coordinates(layer, context, position, resolved["x"])
        geometry[key] = resolved
    return {**layer, "geometry": geometry}


def resolve_layers(layers: list[dict], context: dict) -> list[dict]:
    """Resolve every layer of a plot against the layout context."""
    return [resolve_layer(layer, context) for layer in layers]
