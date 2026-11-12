# Architecture

Overview of how the lithos package is layered, for contributors and coding agents.
The plan for the new API lives in [migration_plan.md](./migration_plan.md); read that
for design decisions about the element/transform API. This document describes the
structure of the code as it exists.

## Layers (current, legacy API)

```
User API                 lithos.plotting.plot_class (LinePlot / CategoricalPlot, BasePlot)
                             |
Input types              lithos.types.plot_input (Group, Subgroup, UniqueGroups)
Data handling           lithos.utils.dataholder (DataHolder), metadata_utils (save/load)
                             |
Processing              lithos.plotting.processing
                        base_processor -> line_processor (continuous x-axis)
                                       -> categorical_processor (dodge/jitter/stack)
                             |
Rendering               lithos.plotting.matplotlib_plotter
                        Plotter -> LinePlotter (rectilinear + polar)
                                -> CategoricalPlotter (categorical axes)
                             |
Formatting              lithos.plotting.formatting
                        FigureFormatter -> ContinuousFormatter (rectilinear + polar)
                                          -> CategoricalFormatter (categorical ticks)
                             |
Stats                   lithos.stats (pure functions: kde, ecdf, histogram, curve_fit,
                        circular stats) — no plotting imports allowed here
```

Dependency direction is strictly downward: plot classes -> processors -> stats/transforms.
`lithos/stats` must stay free of `lithos.plotting` imports; the formatting
module must not import renderer or plot-class code. Do not introduce cycles.

## New element/transform API (Phase 2, in progress)

- `lithos/plotting/elements.py` — formatting-only dataclasses (`Line`, `Marker`, `Bar`,
  `Fill`, `ErrorBand`, `ErrorBar`, `Annotation`) that
  serialize via `to_spec()`/`asdict()`.
- `lithos/plotting/transforms.py` — pure statistical dataclasses (`Identity`, `Aggregate`,
  `Density`, `Summary`, `Fit`) built on `lithos.stats`. Each transform returns
  `dict[group_key, geometry_dict]`.
- `lithos/plotting/spec/` — the new plot classes (`Plot` base with `LinePlot` /
  `CategoricalPlot`), the position resolver and the `SpecPlotter`.
  `.add(transform, *elements)` holds raw transform/element objects; `_process_data()`
  is the only place transforms are invoked (at plot time, against the grouping/columns
  in effect then — so grouping may be set after `.add()`). It emits serializable layer
  specs (asdict transform + element specs + geometry) that the resolver/plotter consume
  and that can be handed to alternative renderers. Metadata stores pure layer specs —
  no geometry — and `load_metadata` recomputes it by replaying `.add()` against the data.
- `lithos/plotting/spec/methods.py` — the Phase 2.5 sugar verbs (`.scatter()`, `.kde()`
  on every layout; `.line()`/`.hist()`/`.ecdf()`/`.fit()` on `LinePlot`; `.summary()`/
  `.bar()`/`.box()` on `CategoricalPlot`): thin factories that build a transform plus role-default
  elements and call `.add()`, so they inherit the pipeline's geometry, parity and metadata
  replay unchanged (see `doc/migration_plan.md` for the verb catalog and argument contract).
- `lithos/plotting/spec/significance.py` — the `Significance` bracket spec and its
  plot-time resolver. Brackets are layout decorations (`add_significance()`, like
  `add_axline`), not `.add()` elements: they consume no transform geometry, render once
  per bracket, resolve group names to slot positions, compute their height from the
  data (auto-y + stacking), and serialize into metadata with no geometry like layers.
- All are exported from `lithos/__init__.py` (see the migration plan for remaining phases).

## Formatting/rendering split (Phase 2.6, in progress)

Plot formatting (axes, ticks, grids, labels, legends) is generic matplotlib
machinery and gets its own component, separate from lithos-specific rendering
(decision pinned in [migration_plan.md](./migration_plan.md)):

- `lithos/plotting/formatting.py` - `FigureFormatter` (base) with
  `ContinuousFormatter` (rectilinear + polar) and `CategoricalFormatter`
  (slot ticks, two-tier group/subgroup labels). The formatter holds the
  figure, the axes and *references* to the format/label/transform dicts
  plus the layout context, and styles grids, minor ticks, tick
  decimals/back-transforms/tick window, spines, axis/title labels,
  reference lines (`hline`/`vline`), legend patches and polar radian
  ticks. It consumes only `(fig, axes)` plus dicts - no plot data - so the
  same machinery can style any matplotlib figure; a public entry point is
  deliberately deferred.
- `lithos/plotting/matplotlib_plotter.py` - keeps figure lifecycle
  (`create_figure`, `plot()`, `savefig`) and all rendering (legacy plot
  methods plus the spec element dispatch). `format_plot()` picks the
  layout's formatter via a `formatter_cls` ClassVar and delegates;
  `plot_legend()` delegates too.
- The spec plotters are unchanged: they inherit the delegating
  `format_plot()` from their legacy plotter bases, and `_LayerView` still
  feeds plot-type names to `format(plot_types=...)` (the kde/hist
  default-limits special case).

## Data flow (legacy)

1. `DataHolder` (or dict/DataFrame/2D array) is passed to a plot class.
2. `preprocess_args` normalizes styling kwargs (any key containing `color`, plus
   `marker`/`linestyle`/`hatch`/`barwidth`).
3. Processors group data (`Group`/`Subgroup`/`UniqueGroups`), compute stats via
   `lithos.stats`, attach metadata, and resolve positions
   (passthrough/dodge/jitter/stack for categorical layouts).
4. Plotters consume processed geometry dicts and render with matplotlib; axis/legend/tick
   formatting (`format_plot`, `set_axis`, `plot_legend`) is delegated to the
   formatting layer, `lithos.plotting.formatting` (`FigureFormatter` subclasses
   consume the format dict and a layout context but never plot data; the
   plotter passes plot-type names into `format(plot_types=...)`).

## Tests

`tests/` mirrors the package: `dataholder/`, `processing/` (line + categorical processor
parity), plus top-level files for transforms, elements, metadata, plot utils,
formatting and colors.
All stats functions are exercised indirectly through transform geometry contracts.
Run with `uv run pytest` (see root `AGENTS.md` for the full command table).

## Metadata save/load

`lithos.utils.metadata_utils` + `DataHolder` provide GraphPad-"magic"-style plot metadata
persistence: design a plot, save its metadata, reload it for other plots. The element spec
extends this serialization in the new API.
