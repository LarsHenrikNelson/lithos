# Migration Plan: Element/Transform Plot API

This document captures the architecture target and the step-by-step migration
away from the verbose per-method plotting API (`LinePlot.line()`,
`CategoricalPlot.jitter()`, ...) toward an element/transform API. It exists to
keep the refactor reviewable and to pin down design decisions *before* large
amounts of code are written.

## Motivation

The current design is soundly layered (plot classes collect input -> processors
compute + attach metadata -> plotters render), but the user-facing methods have
grown into flat bags of 10-20 kwargs, and the method surface encodes a hidden
matrix of *statistical transform x rendering element*. That causes the
duplication we have observed:

- `scatter` == `jitter` == `paired` with different axis positioning
- `kde` == `violin` == `hist` == `ecdf` == `percent` as density/ecdf on the y-axis
- `bar` == `summary` == `aggline` == `fit` as aggregation + a rendered element

## Target architecture

```
User API (plot classes)
  .add(transform, *elements)          + high-level sugar methods
        |
        v
Transforms  (pure stats: Identity, Aggregate, Density, Summary, Fit)
        |
        v
Position resolver (shared; replaces line/categorical processor split)
        |
        v
Spec: backend-agnostic element dicts (serializable, JSON-ready)
        |
        v
Plotters (matplotlib now; JS/other later)
```

### Design decision (agreed): transform owns aggregation AND error

Aggregation and error computation live once in the **transform**. Elements are
pure formatting and never compute statistics. Consequence:

```python
.add(Aggregate(func="mean", error="sem"), Marker(), ErrorBar(), ErrorBand())
```

draws the marker, error bar, and error band from the *single* error geometry
produced by the transform - stacked error elements can never disagree with each
other.

## Element catalog (pure formatting, dataclasses -> `asdict()`)

| Element | Fields (defaults) | Renders |
| --- | --- | --- |
| `Line` | `linecolor, linestyle, linewidth, linealpha` | joined lines (incl. paired connectors) |
| `Marker` | `marker, markercolor, edgecolor, markeredgewidth, markersize, alpha, edge_alpha` | points at geometry |
| `Bar` | `edgecolor, barwidth, linewidth, edge_alpha` | rectangle outlines (hist bars, categorical bars) |

| `Fill` | `fillcolor, fillalpha, hatch, edgecolor, edgealpha` | filled regions (density fill-under, bar faces); the single source of fill styling |
| `ErrorBand` | `fillcolor, fillalpha, edgecolor, edgealpha, linewidth` | shaded band between center +/- error bounds |

## Transform catalog and geometry contract

Each transform `__call__(data, y=None, x=None, levels=(), ytransform=None,
xtransform=None, **kwargs)` returns a `dict[group_key, geometry_dict]`. The
geometry dict is the **only** source of numbers that elements render - no
element computes values.

| Transform | Parameters | Geometry per group |
| --- | --- | --- |
| `Identity` | --- | `{x, y, n}` |
| `Aggregate` | `func, err_func, agg_func, unique_id` | `{center, error_low, error_high, n}` |
| `Density(kind)` | `kde`: `kernel, bw, tol, kde_length, KDEType`<br/>`hist`: `bins, bin_range, stat`<br/>`ecdf`: `ecdf_type, ecdf_args` | kde/ecdf: `{x, y, n}`<br/>hist: `{edges, height, binwidth, centers, stat, n}` |
| `Summary` | `func, err_func, whisker, whisker_quantiles, notch` | `{center, mean, median, q1, q3, whisker_low, whisker_high, error_low, error_high, notch_low, notch_high, n}` |
| `Fit` | `fit_func, ci_func, fit_args` | `{x, y, ci, n}` |

Error normalization contract: error funcs must return a scalar or a (low, high)
pair; `Aggregate`/`Summary` normalize both to `error_low`/`error_high`.

## Migration phases

### Phase 0 - Spec foundation (current)
- `lithos/plotting/elements.py` (element dataclasses, `to_spec()`)
- `lithos/plotting/transforms.py` (transform dataclasses over `lithos/stats`)
- Unit tests: element serialization; transform geometry contracts on synthetic
  grouped data.
- Exit: both modules import cleanly, tests green, zero behavior change.

### Phase 1 - `Plot` prototype
- Single `Plot(data, layout="continuous"|"categorical")` class with
  `.grouping()`, `.add(transform, *elements)`, `.plot()`.
- Position resolution in `BaseProcessor` (layouts: passthrough / dodge /
  jitter / stack) used only by the new API.
- `SpecPlotter` (matplotlib) mapping element dicts -> existing `_plot_*`
  renderers. `metadata()`/save/load extended with the element spec.
- Exit: prototype renders a categorical and a continuous example.

### Phase 2 - Method family ports (dual API; legacy untouched)
Port one family at a time, each time asserting parity between the element-spec
pipeline and the legacy processor output:

1. scatter/jitter/paired -> `Identity` + `Marker` (+ `Line`)
2. summary/bar/aggline/summaryu -> `Aggregate` + elements
3. kde/hist/ecdf/violin/percent -> `Density` + elements
4. box -> `Summary` + `Bar`/`Fill`/`Line`/`Marker`
5. fit -> `Fit` + `Line` + `ErrorBand`

Parity tests live in `tests/processing/` and compare `Plot.add(...)` processed
geometry to the legacy `LineProcessor`/`CategoricalProcessor` output for the
same inputs and defaults.

### Phase 3 - Collapse
- Remove legacy methods (or keep as thin aliases if still in use).
- Merge `LineProcessor`/`CategoricalProcessor` into the shared resolver.
- Decide `LinePlot`/`CategoricalPlot` fate: either thin facades over
  `Plot(layout=...)` or removed entirely (open question, revisit after Phase 1).

## Open questions (parked until their phase)

- Facet / polar figure handling stays renderer-side (`create_figure`,
  `format_polar`) - not part of the spec.
- `bin_limits="common"`-style cross-group binning inside `Density` (Phase 2).
- Whether `LinePlot`/`CategoricalPlot` survive as facades (Phase 3).

| `ErrorBar` | `linecolor, linealpha, linewidth, capsize, capstyle` | caps error bars |
| `Whisker` | `linecolor, linealpha, linewidth, capsize, capstyle` | whisker lines from quantile geometry |
| `Annotation` | `text, x, y, fontsize, color, ha, va, rotation` | free text |
| `Significance` | `text, x1, x2, y, linecolor, linewidth, fontsize, capsize` | GraphPad-style brackets+asterisks |

Notes:

- `Fill` is the single source of fill styling (including hatch) for density
  fill-under curves and bar faces. `ErrorBand` is strictly the center +/- error
  band. No `Connector`: paired connectors are just `Line`
  elements over the connection geometry.
- Field names intentionally match the legacy processor argument names so that
  the serialized spec can reuse `preprocess_args` (any key containing `color`,
  plus `marker`/`linestyle`/`hatch`/`barwidth` special-casing) unchanged.
