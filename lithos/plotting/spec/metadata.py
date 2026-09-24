"""Serialization helpers for the spec plot API (metadata version 2).

Spec metadata is plain JSON (standard, robust, and still human-readable via
indentation). The legacy custom txt/literal_eval format stays with the legacy
plot classes and is untouched.

JSON-friendliness rules:

- numpy arrays/scalars are converted to lists/numbers and callables are
  serialized to their name (shared limitation with the legacy format).
- Tuple-keyed group geometry (``dict[tuple, dict]``) cannot be a JSON
  object, so serialized layers store geometry as a ``"groups"`` list of
  ``{"key": [...], "geometry": {...}}`` entries; ``layer_from_json``
  rebuilds the tuple-keyed geometry dict on load.

On load, transforms and elements are rebuilt from their specs and ``.add()``
is replayed so geometry is recomputed against the current data.
"""

import json
from pathlib import Path

import numpy as np

from ...utils import metadata_utils
from ..elements import (
    Annotation,
    Bar,
    Element,
    ErrorBand,
    ErrorBar,
    Fill,
    Line,
    Marker,
    Significance,
)
from ..transforms import (
    ECDF,
    KDE,
    Aggregate,
    Fit,
    Histogram,
    Identity,
    Summary,
    Transform,
)

ELEMENT_TYPES = {cls().type: cls for cls in (Annotation, Bar, ErrorBand, ErrorBar, Fill, Line, Marker, Significance)}
TRANSFORM_NAMES = {cls().name: cls for cls in (Aggregate, ECDF, Fit, Histogram, Identity, KDE, Summary)}


def to_jsonable(obj):
    """Recursively convert numpy values and callables to plain JSON-safe types."""
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if callable(obj):
        return getattr(obj, "__name__", "callable")
    return obj


def layer_to_json(layer: dict) -> dict:
    """JSON-friendly layer: tuple-keyed geometry becomes a ``"groups"`` list."""
    output = {}
    for key, value in layer.items():
        if key == "geometry":
            output["groups"] = [
                {"key": to_jsonable(list(group_key)), "geometry": to_jsonable(geometry)}
                for group_key, geometry in value.items()
            ]
        else:
            output[key] = to_jsonable(value)
    return output


def layer_from_json(layer: dict) -> dict:
    """Rebuild an in-memory layer (tuple-keyed geometry) from its JSON form."""
    output = {key: value for key, value in layer.items() if key != "groups"}
    output["geometry"] = {tuple(entry["key"]): entry["geometry"] for entry in layer["groups"]}
    return output


def _metadata_file(file_path: str | Path) -> Path:
    """Resolve a metadata path to a ``.json`` file.

    A bare string (no suffix) keeps the legacy UX: it is stored in the
    configured metadata directory as ``<name>.json``.
    """
    if isinstance(file_path, str):
        if len(Path(file_path).suffix) == 0:
            file_path = metadata_utils.metadata_dir() / file_path
    file_path = Path(file_path)
    if file_path.suffix in ("", ".txt"):
        file_path = file_path.with_suffix(".json")
    return file_path


def save_spec_metadata(metadata: dict, file_path: str | Path) -> None:
    """Save version-2 spec metadata as JSON."""
    file_path = _metadata_file(file_path)
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_spec_metadata(file_path: str | dict | Path) -> dict:
    """Load version-2 spec metadata from JSON (a dict passes through untouched)."""
    if isinstance(file_path, dict):
        return file_path
    file_path = _metadata_file(file_path)
    with open(file_path) as f:
        return json.load(f)


def build_transform(spec: dict) -> Transform:
    """Rebuild a transform instance from its serialized spec."""
    name = spec.get("name", "transform")
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"Unknown transform {name!r} in metadata.")
    return TRANSFORM_NAMES[name](**spec)


def build_element(spec: dict) -> Element:
    """Rebuild an element instance from its serialized spec."""
    element_type = spec.get("type", "element")
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type {element_type!r} in metadata.")
    return ELEMENT_TYPES[element_type](**spec)
