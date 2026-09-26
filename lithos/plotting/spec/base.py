"""Spec-based plot API: the ``Plot`` base class (metadata version 2).

The new user-facing plot API, decoupled from the legacy classes in
:mod:`lithos.plotting.plot_class` (which remain untouched and serve as the
parity reference). The API decomposes the concerns the legacy
``grouping()`` methods mixed together:

- ``.grouping()`` — pure grouping (group/subgroup columns + ordering), shared.
- ``.plot_data()`` — plot-level default y/x columns (data only, no label text).
- ``.labels()`` — label text (``None`` = no label, ``""`` = empty label; unset axis
  labels default to the y/x column names).
- ``.label_format()`` — pure label/tick formatting (sizes, fonts, rotations).
- ``.add(transform, *elements)`` — the layer API. The transform and elements
  are held *as-is*; no data is processed until ``_process_data()`` runs (at
  plot time), so grouping/columns set after ``.add()`` are honored. All
  layers are processed in one pass, so stacked layers (e.g. subject-level
  ``Identity`` + aggregate-level ``Aggregate``) never disagree.
- Layout-specific settings (faceting for :class:`~lithos.plotting.spec.line.LinePlot`,
  spacing/labels for :class:`~lithos.plotting.spec.categorical.CategoricalPlot`)
  live on the subclasses, not on ``grouping()``.
"""

from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Literal

from typing_extensions import Self

from ...types.basic_types import InputData, SavePath, Transform
from ...types.plot_input import Grouping, Subgrouping
from ...utils import DataHolder
from ..elements import Element
from ..transforms import Transform as StatTransform
from .metadata import (
    build_element,
    build_transform,
    layer_to_json,
    load_spec_metadata,
    save_spec_metadata,
)


class Unset:
    """Sentinel for a label that has not been set (resolved at plot time)."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = Unset()


class Plot:
    """Base class for the spec plot API.

    Subclasses declare a ``layout`` name, the default per-layer ``position``
    mode, and the allowed ``positions``; they implement ``_layout_context()``
    (the serializable positioning/labeling context handed to the plotter).
    """

    layout: ClassVar[str] = "base"
    default_position: ClassVar[str] = "passthrough"
    positions: ClassVar[tuple[str, ...]] = ("passthrough",)

    def __init__(self, data: InputData):
        self.data = DataHolder(data)
        self.layers: list[dict] = []
        self.plotter = None

        self._grouping = {"group": None, "subgroup": None, "group_order": None, "subgroup_order": None}
        self._layout_options: dict = {}
        self._plot_data = {"y": None, "x": None}
        self._labels = {"ylabel": UNSET, "xlabel": UNSET, "title": UNSET, "figure_title": UNSET}
        self.plot_format: dict = {}
        self._plot_transforms: dict = {}

        self.label_format()
        self.axis()
        self.axis_format()
        self.figure()
        self.grid()
        self.transform()

    def grouping(
        self,
        group: str | int | None = None,
        subgroup: str | int | None = None,
        group_order: Grouping = None,
        subgroup_order: Subgrouping = None,
    ) -> Self:
        """Set the grouping columns and ordering (pure grouping — no layout settings)."""
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
        }

        return self

    def plot_data(self, y: str | None = None, x: str | None = None) -> Self:
        """Set the plot-level default y/x columns (pure data — no label text).

        The defaults are used by ``.add()`` layers that do not pass their own
        ``y``/``x``; stacked layers can still override them per layer. Label
        text lives in ``.labels()``; formatting lives in ``.label_format()``.
        """
        self._plot_data = {"y": y, "x": x}

        return self

    def labels(
        self,
        ylabel: str | None | Unset = UNSET,
        xlabel: str | None | Unset = UNSET,
        title: str | None | Unset = UNSET,
        figure_title: str | None | Unset = UNSET,
    ) -> Self:
        """Set the label text, kept separate from the data columns.

        Each label distinguishes three states:

        - unset (default) — the axis labels fall back to the ``y``/``x``
          column names from ``.plot_data()``; the titles fall back to none.
        - ``None`` — no label.
        - ``""`` — an explicitly empty label.
        """
        self._labels = {
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
            "figure_title": figure_title,
        }

        return self

    def _resolved_labels(self) -> dict:
        """Resolve the label states into concrete label text for the plotter.

        Unset axis labels fall back to the plot-level ``y``/``x`` column names
        (blank when the column is not set) and unset titles to no title. The
        ``None`` (no label) and ``""`` (empty label) states are preserved so
        they round-trip through the metadata; the plotter renders both blank.
        """
        labels = dict(self._labels)
        if isinstance(labels["ylabel"], Unset):
            labels["ylabel"] = self._plot_data["y"] if self._plot_data["y"] is not None else ""
        if isinstance(labels["xlabel"], Unset):
            labels["xlabel"] = self._plot_data["x"] if self._plot_data["x"] is not None else ""
        if isinstance(labels["title"], Unset):
            labels["title"] = ""
        if isinstance(labels["figure_title"], Unset):
            labels["figure_title"] = ""
        return labels

    def add(
        self,
        transform: StatTransform,
        *elements: Element,
        y: str | None = None,
        x: str | None = None,
        ytransform: Transform | None = None,
        xtransform: Transform | None = None,
        position: str | None = None,
        seed: int = 42,
    ) -> Self:
        """Add a layer: one transform plus the elements that render its geometry.

        The transform and elements are held *as-is* — no data is processed.
        Geometry is computed later by ``_process_data()`` (at plot time)
        against the grouping/columns in effect then, so ``.grouping()`` and
        ``.plot_data()`` calls made after ``.add()`` are honored.
        ``position`` selects how the resolver maps the layer onto the layout
        when the transform does not supply a coordinate itself.
        """
        if not isinstance(transform, StatTransform):
            raise TypeError(f"add() expects a Transform instance, got {type(transform).__name__!r}.")
        if not elements:
            raise ValueError("add() requires at least one element to render.")
        for element in elements:
            if not isinstance(element, Element):
                raise TypeError(f"add() expects Element instances, got {type(element).__name__!r}.")
        if position is None:
            position = self.default_position
        if position not in self.positions:
            raise ValueError(f"position must be one of {self.positions} for {type(self).__name__}, got {position!r}.")

        self.layers.append(
            {
                "transform": transform,
                "elements": list(elements),
                "y": y,
                "x": x,
                "ytransform": ytransform,
                "xtransform": xtransform,
                "position": position,
                "seed": seed,
            }
        )

        return self

    def _process_data(self) -> list[dict]:
        """Compute every layer's geometry against the current plot state.

        This is the only place transforms are invoked: each held transform is
        called with the grouping and columns in effect right now, so
        ``.grouping()``/``.plot_data()`` calls made after ``.add()`` are
        honored. The returned layers are plain serializable dicts (asdict
        transform spec, element specs, fresh geometry) — the backend-agnostic
        artifact that can be handed to any renderer.
        """
        processed = []
        for layer in self.layers:
            y = layer["y"] if layer["y"] is not None else self._plot_data["y"]
            x = layer["x"] if layer["x"] is not None else self._plot_data["x"]
            geometry = layer["transform"](
                self.data,
                y=y,
                x=x,
                levels=self._levels(),
                ytransform=layer["ytransform"],
                xtransform=layer["xtransform"],
            )
            processed.append(
                {
                    "transform": asdict(layer["transform"]),
                    "elements": [element.to_spec() for element in layer["elements"]],
                    "y": y,
                    "x": x,
                    "ytransform": layer["ytransform"],
                    "xtransform": layer["xtransform"],
                    "position": layer["position"],
                    "seed": layer["seed"],
                    "geometry": geometry,
                }
            )
        return processed

    def _levels(self) -> tuple:
        """Grouping columns (group, subgroup) with ``None`` entries dropped."""
        return tuple(level for level in (self._grouping["group"], self._grouping["subgroup"]) if level is not None)

    def label_format(
        self,
        labelsize: float = 20,
        titlesize: float = 22,
        xticklabel_size: int = 12,
        yticklabel_size: int = 12,
        font: str = "DejaVu Sans",
        fontweight: None | str | float = None,
        title_fontweight: str | float = "regular",
        label_fontweight: str | float = "regular",
        tick_fontweight: str | float = "regular",
        xlabel_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ylabel_rotation: Literal["horizontal", "vertical"] | float = "vertical",
        xtick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ytick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
    ) -> Self:
        """Set the label/tick formatting: sizes, fonts, weights and rotations."""
        if fontweight is not None:
            title_fontweight = fontweight
            label_fontweight = fontweight
            tick_fontweight = fontweight

        label_props = {
            "labelsize": labelsize,
            "titlesize": titlesize,
            "font": font,
            "xticklabel_size": xticklabel_size,
            "yticklabel_size": yticklabel_size,
            "title_fontweight": title_fontweight,
            "label_fontweight": label_fontweight,
            "tick_fontweight": tick_fontweight,
            "xlabel_rotation": xlabel_rotation,
            "ylabel_rotation": ylabel_rotation,
            "xtick_rotation": xtick_rotation,
            "ytick_rotation": ytick_rotation,
        }
        self.plot_format["labels"] = label_props
        return self

    def axis(
        self,
        ylim: list | tuple | None = None,
        xlim: list | tuple | None = None,
        yaxis_lim: list | tuple | None = None,
        xaxis_lim: list | tuple | None = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        ydecimals: int | None = None,
        xdecimals: int | None = None,
        xformat: Literal["f", "e"] = "f",
        yformat: Literal["f", "e"] = "f",
        yunits: Literal["degree", "radian", "wradian"] | None = None,
        xunits: Literal["degree", "radian", "wradian"] | None = None,
    ) -> Self:
        if ylim is None:
            ylim = (None, None)
        if xlim is None:
            xlim = (None, None)

        axis_settings = {
            "yscale": yscale,
            "xscale": xscale,
            "ylim": ylim,
            "xlim": xlim,
            "yaxis_lim": yaxis_lim,
            "xaxis_lim": xaxis_lim,
            "ydecimals": ydecimals,
            "xdecimals": xdecimals,
            "xunits": xunits,
            "yunits": yunits,
            "xformat": xformat,
            "yformat": yformat,
        }
        self.plot_format["axis"] = axis_settings

        return self

    def axis_format(
        self,
        linewidth: float | dict[str, float] = 2,
        tickwidth: float = 2,
        ticklength: float = 5.0,
        minor_tickwidth: float = 1.5,
        minor_ticklength: float = 2.5,
        yminorticks: int = 0,
        xminorticks: int = 0,
        ysteps: int | tuple[int, int, int] = 5,
        xsteps: int | tuple[int, int, int] = 5,
        truncate_xaxis: bool = False,
        truncate_yaxis: bool = False,
        style: Literal["default", "lithos"] = "lithos",
    ) -> Self:
        if isinstance(ysteps, int):
            ysteps = (ysteps, 0, ysteps)
        if isinstance(xsteps, int):
            xsteps = (xsteps, 0, xsteps)
        if isinstance(linewidth, (int, float)):
            linewidth = {"left": linewidth, "bottom": linewidth, "top": 0, "right": 0}
        elif isinstance(linewidth, dict):
            temp_lw = {"left": 0, "bottom": 0, "top": 0, "right": 0}
            for key, value in linewidth:
                temp_lw[key] = value
            linewidth = temp_lw

        axis_format = {
            "tickwidth": tickwidth,
            "ticklength": ticklength,
            "linewidth": linewidth,
            "minor_tickwidth": minor_tickwidth,
            "minor_ticklength": minor_ticklength,
            "yminorticks": yminorticks,
            "xminorticks": xminorticks,
            "xsteps": xsteps,
            "ysteps": ysteps,
            "style": style,
            "truncate_xaxis": truncate_xaxis,
            "truncate_yaxis": truncate_yaxis,
        }

        self.plot_format["axis_format"] = axis_format

        return self

    def figure(
        self,
        margins=0.05,
        aspect: int | float | None = None,
        figsize: None | tuple[int, int] = None,
        gridspec_kw: dict[str, str | int | float] | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        projection: Literal["rectilinear", "polar"] = "rectilinear",
    ) -> Self:
        figure = {
            "gridspec_kw": gridspec_kw,
            "margins": margins,
            "aspect": aspect if projection == "rectilinear" else None,
            "figsize": figsize,
            "nrows": nrows,
            "ncols": ncols,
            "projection": projection,
        }

        self.plot_format["figure"] = figure

        return self

    def grid(
        self,
        ygrid: int | float = 0,
        xgrid: int | float = 0,
        yminor_grid: int | float = 0,
        xminor_grid: int | float = 0,
        linestyle: str | tuple = "solid",
        minor_linestyle: str | tuple = "solid",
    ) -> Self:
        grid = {
            "ygrid": ygrid,
            "xgrid": xgrid,
            "yminor_grid": yminor_grid,
            "xminor_grid": xminor_grid,
            "linestyle": linestyle,
            "minor_linestyle": minor_linestyle,
        }
        self.plot_format["grid"] = grid

        return self

    def transform(
        self,
        ytransform: Transform | None = None,
        back_transform_yticks: bool = False,
        xtransform: Transform | None = None,
        back_transform_xticks: bool = False,
    ) -> Self:
        self._plot_transforms = {}
        self._plot_transforms["ytransform"] = ytransform
        if callable(ytransform):
            self._plot_transforms["back_transform_yticks"] = False
        else:
            self._plot_transforms["back_transform_yticks"] = back_transform_yticks

        self._plot_transforms["xtransform"] = xtransform
        if callable(xtransform):
            self._plot_transforms["back_transform_xticks"] = False
        else:
            self._plot_transforms["back_transform_xticks"] = back_transform_xticks

        return self

    def add_axline(
        self,
        linetype: Literal["hline", "vline"],
        lines: list,
        linestyle="solid",
        linealpha=1,
        linecolor="black",
        linewidth=1.5,
        zorder=1,
    ) -> Self:
        if linetype not in ["hline", "vline"]:
            raise AttributeError("linetype must by hline or vline")
        if isinstance(lines, (float, int)):
            lines = [lines]
        self.plot_format[linetype] = {
            "linetype": linetype,
            "lines": lines,
            "linestyle": linestyle,
            "linealpha": linealpha,
            "linecolor": linecolor,
            "linewidth": linewidth,
            "zorder": zorder,
        }

        return self

    def get_format(self) -> dict:
        return self.plot_format

    # -- layout hooks ------------------------------------------------------
    def _layout_context(self) -> dict:
        """Serializable positioning/labeling context consumed by the plotter."""
        raise NotImplementedError("Subclasses must implement _layout_context().")

    def layout_options(self) -> dict:
        return dict(self._layout_options)

    def _set_layout_options(self, options: dict):
        self._layout_options = dict(options)

    # -- metadata (version 2, JSON) ------------------------------------------
    def metadata(self) -> dict:
        """Version-2 metadata: grouping, layout, format, labels and layer specs.

        The output is plain JSON. Layers are serialized lazily: each held
        transform becomes an ``asdict`` spec and each element its ``to_spec()``
        dict — no geometry is stored (``load_metadata`` recomputes it by
        replaying ``.add()`` against the current data).
        """
        return {
            "version": 2,
            "layout": self.layout,
            "grouping": dict(self._grouping),
            "layout_options": self._layout_options,
            "data": dict(self._plot_data),
            "labels": self._resolved_labels(),
            "format": self.plot_format,
            "transforms": self._plot_transforms,
            "layers": [layer_to_json(layer) for layer in self.layers],
        }

    def save_metadata(self, file_path: str | Path):
        save_spec_metadata(self.metadata(), file_path)

    def load_metadata(self, metadata_path: str | dict | Path) -> Self:
        """Load version-2 JSON metadata onto this plot, replaying the ``.add()`` layers."""
        metadata = load_spec_metadata(metadata_path)
        if metadata.get("version") != 2:
            raise ValueError("Not a spec (version 2) metadata file.")
        if metadata.get("layout") != self.layout:
            raise ValueError(
                f"Metadata layout {metadata.get('layout')!r} does not match {type(self).__name__} layout {self.layout!r}."
            )

        self._grouping = dict(metadata["grouping"])
        self._set_layout_options(metadata.get("layout_options", {}))
        self._plot_data = {key: metadata["data"].get(key) for key in ("y", "x")}
        labels = metadata.get("labels")
        if labels is None:
            # earlier metadata versions carried the label text inside "data"
            labels = {key: metadata["data"].get(key, "") for key in ("ylabel", "xlabel", "title", "figure_title")}
        self._labels = {key: labels.get(key, "") for key in ("ylabel", "xlabel", "title", "figure_title")}
        for key, value in metadata["format"].items():
            self.plot_format[key] = value
        self._plot_transforms = dict(metadata["transforms"])

        self.layers = []
        for layer in metadata["layers"]:
            self.add(
                build_transform(layer["transform"]),
                *[build_element(spec) for spec in layer["elements"]],
                y=layer["y"],
                x=layer["x"],
                ytransform=layer.get("ytransform"),
                xtransform=layer.get("xtransform"),
                position=layer["position"],
                seed=layer.get("seed", 42),
            )

        return self

    # -- rendering -----------------------------------------------------------
    def plot(
        self,
        savefig: bool = False,
        path: SavePath = "",
        filename: str = "",
        filetype: str = "svg",
        save_metadata: bool = False,
        **kwargs,
    ) -> Self:
        """Resolve every layer against the layout and render through the spec plotter."""
        if path == "" or path is None:
            path = Path().cwd()
        elif isinstance(path, str):
            path = Path(path)
        filename_output = filename if filename != "" else (self._plot_data["y"] or "")

        context = self._layout_context()

        from .plotter import get_spec_plotter
        from .resolver import resolve_layers

        resolved = resolve_layers(self._process_data(), context)
        self.plotter = get_spec_plotter(context["layout"])(
            layers=resolved,
            plot_dict=context,
            metadata=self.metadata(),
            savefig=savefig,
            path=path,
            filetype=filetype,
            filename=filename_output,
            **kwargs,
        )
        self.plotter.plot()

        if save_metadata and isinstance(path, Path):
            self.save_metadata(path / f"{filename_output}.txt")

        return self
