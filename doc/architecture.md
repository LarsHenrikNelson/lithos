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
                                -> CategoricalPlotter (categorical axes, formatting)
                             |
Stats                   lithos.stats (pure functions: kde, ecdf, histogram, curve_fit,
                        circular stats) — no plotting imports allowed here
```

Dependency direction is strictly downward: plot classes -> processors -> stats/transforms.
`lithos/stats` must stay free of `lithos.plotting` imports. Do not introduce cycles.

## New element/transform API (Phase 0/1)

- `lithos/plotting/elements.py` — formatting-only dataclasses (`Line`, `Marker`, `Bar`,
  `Fill`, `ErrorBand`, `ErrorBar`, `Whisker`, `Box`, `Annotation`, `Significance`) that
  serialize via `to_spec()`/`asdict()`.
- `lithos/plotting/transforms.py` — pure statistical dataclasses (`Identity`, `Aggregate`,
  `Density`, `Summary`, `Fit`) built on `lithos.stats`. Each transform returns
  `dict[group_key, geometry_dict]`.
- Both are exported from `lithos/__init__.py` today; the `Plot` class, position resolver
  and `SpecPlotter` arrive in Phase 1 (see the migration plan).

## Data flow (legacy)

1. `DataHolder` (or dict/DataFrame/2D array) is passed to a plot class.
2. `preprocess_args` normalizes styling kwargs (any key containing `color`, plus
   `marker`/`linestyle`/`hatch`/`barwidth`).
3. Processors group data (`Group`/`Subgroup`/`UniqueGroups`), compute stats via
   `lithos.stats`, attach metadata, and resolve positions
   (passthrough/dodge/jitter/stack for categorical layouts).
4. Plotters consume processed geometry dicts and render with matplotlib; axis/legend/tick
   formatting (`format_plot`, `set_axis`, `plot_legend`) is centralized in `Plotter`.

## Tests

`tests/` mirrors the package: `dataholder/`, `processing/` (line + categorical processor
parity), plus top-level files for transforms, elements, metadata, plot utils and colors.
All stats functions are exercised indirectly through transform geometry contracts.
Run with `uv run pytest` (see root `AGENTS.md` for the full command table).

## Metadata save/load

`lithos.utils.metadata_utils` + `DataHolder` provide GraphPad-"magic"-style plot metadata
persistence: design a plot, save its metadata, reload it for other plots. The element spec
extends this serialization in the new API.
