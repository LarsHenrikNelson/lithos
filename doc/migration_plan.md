# Migration Plan: Element/Transform Plot API

This document captures the architecture target and the step-by-step migration
away from the verbose per-method plotting API (`LinePlot.line()`,
`CategoricalPlot.jitter()`, ...) toward an element/transform API. It exists to
keep the refactor reviewable and to pin down design decisions *before* large
amounts of code are written. Phases below are marked **done** / **(current)** /
pending as the migration proceeds; the catalogs describe the API as built.

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
Transforms  (pure stats: Identity, Aggregate, KDE, Histogram, ECDF, Summary, Fit)
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
.add(Aggregate(func="mean", err_func="sem"), Marker(), ErrorBar(), ErrorBand())
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
| `ErrorBar` | `linecolor, linealpha, linewidth, capsize, capstyle` | capped error bars around a center (also box-plot whiskers over `Summary` geometry) |
| `SummaryLine` | `linecolor, linewidth, linealpha, capstyle` | short line across an aggregate center (legacy `summary`/`summaryu` line); the length comes from the layer `width` in `.add()` (slot fraction, default 0.9; `unique_id`-nested lines fill their subject column) |
| `Annotation` | `text, x, y, fontsize, color, ha, va, rotation` | free text |

Every element also carries a `zorder` field (`int | float | None`; `None` = the
plotter default, with layers stacking in `.add()` order), and a `type` ClassVar
that identifies the element in serialized specs but is not an `__init__` field.
There is deliberately no `Whisker` element: box-plot whiskers render through
`ErrorBar` over the `whisker_low`/`whisker_high` geometry the `Summary`
transform already computes.

## Transform catalog and geometry contract

Each transform `__call__(data, y=None, x=None, levels=(), ytransform=None,
xtransform=None, **kwargs)` returns a `dict[group_key, geometry_dict]`. The
geometry dict is the **only** source of numbers that elements render - no
element computes values.

Density transforms (`KDE`/`Histogram`/`ECDF`) take exactly one value column -
`y` or `x`, never both (legacy silently preferred `x` when both were given) -
each paired with its own axis transform. Legacy `kde`/`hist`/`ecdf` crossed
the transforms (`transform = ytransform if xtransform is None else
xtransform`, plus a `xtransfrom` typo in `_hist` and hardcoded `ytransform`
in `_violin`/`_bar`), so the wrong-axis transform was applied whenever both
were set; the new API pairs column and transform unconditionally. Density
geometry is canonical and orientation-free - `x`/`edges` hold the evaluated
values, `y`/`height` the density - and orientation is decided at render time
from the value column's axis (values land on their named axis, like
`Identity`/`Summary` horizontal layouts), never baked into the geometry.

Density orientation is resolved, not swapped in transforms: the resolver
marks every `KDE`/`Histogram` geometry with `value_axis` (the value column's
axis) and renderers place the values on that axis. Vertical (`y`-column) KDE
curves and histogram bars render through `fill_betweenx`/`barh`; horizontal
(`x`-column) layers render as-is; `ECDF` stays canonical (values on x,
probabilities on y, either column). On categorical layouts a `KDE` layer
resolves to violin geometry - values on the value axis, the density scaled so
its maximum spans `width` x the slot extent around the position
(`unique_id`-nested keys: their even column) - rendered through `Fill`; other
elements over violin geometry raise until their Phase 2 port, and the legacy
split/alternate violin styles remain pending.

Density parity (`tests/processing/density_parity_test.py`) pins the shared
computation between `Plot.add(KDE()/Histogram(), ...)` and the legacy
`_kde`/`_hist`/`_violin` processors, with explicit parameters where defaults
deliberately differ: legacy `kde` defaults `bw="silverman"` (transform:
`"ISJ"`), legacy `hist` defaults `stat="count"` (transform: `"density"`),
legacy `violin` defaults `kde_length=128` (transform: `None`). Parity only
sets the matching axis transform (the legacy transform-crossing bug would
otherwise compare against wrongly-transformed geometry). Known positional
divergences, documented in the test: subgrouped violins (legacy width is
absolute axis units; the resolver scales by the slot extent) and nested
violins (legacy `overlap` stacks subjects on the slot center; the resolver
places even columns). Legacy `_hist` also computes the auto bin range from
the first group and reuses it for every later group, while the transform
computes true per-group ranges — histogram parity therefore pins
`bin_limits="common"` (global edges, identical on both sides). The legacy
per-subject hist branch additionally double-truncates the bin edges (it
appends left edges where every other branch appends full edges), silently
dropping the last bin; the transform keeps every bin.

| Transform | Parameters | Geometry per group |
| --- | --- | --- |
| `Identity` | `unique_id` | `{n, x?, y?}` — at least one of x/y; the omitted axis is supplied by the position resolver (jitter/dodge; horizontal layouts use `x` only). With `unique_id`: one geometry dict per subject's ordered series (paired connectors), sorted by `x` when given, with pair-completeness validation |
| `Aggregate` | `func, err_func, agg_func, unique_id, how` | arrays — without `x`: `{center, error_low, error_high, n}` (one value per group; with `unique_id` and `agg_func=None`: one per `(group..., uid)` — legacy `summaryu` per-subject lines); with `x`: `{x, center, error_low, error_high, n}` (one value per unique x; per-x aggregation, legacy `aggline`/`line` parity). `how=auto\|groupby\|matrix` picks the ragged-vs-aligned per-x strategy |
| `KDE` | `kernel, bw, tol, kde_length, KDEType` + `unique_id, agg_func, err_func` | `{x, y, n}` per group/subject (per-subject grids). With `unique_id` + `agg_func`: pooled-group eval grid (FFT-evaluated), `agg_func`/`err_func` across subject curves: `{x, y, error_low, error_high, n}` (`n` = subjects, errors per grid point) |
| `Histogram` | `bins, bin_limits, stat` + `unique_id, agg_func, err_func` | `{edges, height, binwidth, centers, stat, n}`. `bin_limits`: `None` = per-group auto range, `"common"` = global shared edges, `(low, high)` = explicit range. With `unique_id`: per-group shared edges per subject; with `agg_func`: per-subject heights aggregated per group with per-bin errors (integer `bins`) |
| `ECDF` | `ecdf_type, ecdf_args` + `unique_id, agg_func, err_func` | `{x, y, n}` per group/subject. With `unique_id` + `agg_func` (spline): shared probability grid, `x` = aggregated quantile values (legacy axis swap): `{x, y, error_low, error_high, n}` |
| `Summary` | `func, err_func, whisker, whisker_quantiles, notch` | `{center, mean, median, q1, q3, whisker_low, whisker_high, error_low, error_high, notch_low, notch_high, n}` |
| `Fit` | `fit_func, ci_func, fit_args, unique_id, agg_func, err_func` | `{x, y, error_low, error_high, n}` — fit grid `x`, fitted `y`, and CI half-widths as symmetric `error_low`/`error_high` (`None` without a `ci_func`; replaces the orphan `ci` key so `ErrorBand` renders fit bands without special-casing). With `unique_id` and `agg_func=None`: one geometry per subject (nested `(group..., uid)` keys) — per-subject fits with per-subject CI. With `unique_id` + `agg_func`: shared `linspace(min(group x), max(group x), 100)` grid, `agg_func` across subject curves, `err_func` per-grid-point errors — legacy `fit(unique_id=...)` parity. `n` = fitted points, or subjects in the aggregated branch |

Error normalization contract: error funcs must return a scalar or a (low, high)
pair; `Aggregate`/`Summary` normalize both to `error_low`/`error_high`.

## Migration phases

### Phase 0 - Spec foundation (done)
- `lithos/plotting/elements.py` (element dataclasses, `to_spec()`)
- `lithos/plotting/transforms.py` (transform dataclasses over `lithos.stats`)
- Unit tests: element serialization (`tests/elements_test.py`); transform
  geometry contracts on synthetic grouped data (`tests/transforms_test.py`).
- Exit: both modules import cleanly, tests green, zero behavior change.

### Phase 1 - `Plot` prototype (done)
- `lithos/plotting/spec/` package: `base.py` (the `Plot` base class with
  `.grouping()`, `.labels()`, `.add()`, `.transform()`, `process_data()`,
  `.plot()` and metadata save/load), `line.py` / `categorical.py` (the layout
  subclasses), `resolver.py`, `plotter.py`, `metadata.py` and
  `significance.py`.
- Landed as two layout classes rather than a single `Plot(layout=...)`:
  `LinePlot` (continuous axes; default position `passthrough`, plus
  `.facet()`) and `CategoricalPlot` (default position `dodge`, positions
  `passthrough`/`dodge`, plus `.spacing()` / `.categorical_labels()`).
- `.add(transform, *elements, position=, width=, jitter_type=, seed=)` holds
  raw transform/element objects; all geometry is computed in a single
  `Plot._process_data()` pass at plot time (also exposed as the public
  `process_data()`), so `.grouping()`/`.transform()` calls made after `.add()`
  are honored. A `unique_id` that collides with a grouping column raises
  `ValueError` (nesting keys would collide with the group slots).
- Point spread within a categorical slot is a per-layer concern: `width`
  (fraction of the slot; 0 = slot center), `jitter_type`, `seed` in `.add()`.
  Transforms nesting by `unique_id` place one even column per subject across
  the width (legacy `jitteru`/`summaryu` positions) — decided, no
  `dodge_group`/`dodge_subgroup` layout knobs. Single-value-per-group geometry
  (aggregate centers) stays anchored at the slot center.
- Categorical layout: group clusters span a fixed 1 unit; `.spacing(pitch=)`
  is the distance between group centers (unlike legacy `group_spacing`, which
  was really the cluster width). The layout context still passes
  `group_spacing=1.0` for the legacy axis-margin math.
- `SpecPlotter` (matplotlib) maps element dicts -> the existing `_plot_*`
  renderers. Metadata is version-3 JSON saved/loaded via
  `save_metadata()`/`load_metadata()`; it stores pure layer specs (no
  geometry) and `load_metadata` recomputes geometry by replaying `.add()`
  against the data.
- Exit: prototype renders a categorical and a continuous example.

### Phase 2 - Method family ports (current; dual API, legacy untouched)
Port one family at a time, each time asserting parity between the element-spec
pipeline and the legacy processor output:

1. done - scatter/jitter/paired -> `Identity` + `Marker` (+ `Line`);
   parity in `tests/processing/paired_parity_test.py`.
2. in progress - summary/bar/aggline/summaryu -> `Aggregate` +
   `SummaryLine`/`ErrorBar`/`Bar`/`Fill`; the `SummaryLine` element landed,
   processor parity tests pending.
3. pending - kde/hist/ecdf/violin/percent -> `KDE`/`Histogram`/`ECDF` +
   elements (the transforms themselves are landed and tested in
   `tests/transforms_test.py`; the parity port is pending).
4. pending - box -> `Summary` + `Bar`/`Fill`/`Line`/`Marker`/`ErrorBar`
   (no `Whisker` element: whiskers render via `ErrorBar` over the
   `whisker_low`/`whisker_high` geometry `Summary` already computes).
5. done - fit -> `Fit` + `Line` + `ErrorBand`; parity in
   `tests/processing/fit_parity_test.py`.

Parity tests live in `tests/processing/` and compare `Plot.add(...)` processed
geometry to the legacy `LineProcessor`/`CategoricalProcessor` output for the
same inputs and defaults.

### Phase 3 - Collapse
- Remove legacy methods (or keep as thin aliases if still in use).
- Merge `LineProcessor`/`CategoricalProcessor` into the shared resolver.
- Decide `LinePlot`/`CategoricalPlot` fate: either thin facades over
  `Plot(layout=...)` or removed entirely (open question, revisit after
  Phase 2).

## Open questions (parked until their phase)

- Facet / polar figure handling stays renderer-side (`create_figure`,
  `format_polar`) - not part of the spec.
- `bin_limits="common"`-style cross-group binning inside `Histogram`
  (resolved: `Histogram(bin_limits="common")` computes global edges, shared
  by every group/subject hist).
- Whether `LinePlot`/`CategoricalPlot` survive as facades (Phase 3).

## Layout decorations (`add_axline`, `add_significance`)

Not everything drawn on a plot formats transform geometry. Two decorations
live outside the `.add()` layer pipeline because they consume *no* transform
geometry: they are resolved against the layout at plot time and render once
(not once per group).

- `add_axline(linetype, lines)` — axis-spanning reference lines.
- `add_significance(...)` — GraphPad-style significance brackets. A bracket
  spans groups by name (`groups=[...]`; plain group values, or
  `(group, subgroup)` tuples; a plain group value on a subgrouped plot spans
  the whole cluster; 3+ entries span leftmost-to-rightmost) or absolute
  `x1`/`x2` positions (a continuous-layout bracket may combine a single
  `groups` facet key with an `x1`/`x2` window). `y=None` places the bracket
  automatically one `gap` (fraction of the y-range) above the maximum y under
  the span, stacking overlapping brackets one `step` apart; `style="bracket"`
  caps descend only to the top of the plotted data (or a nested bracket's
  line), `style="line"` draws a plain line without caps. Brackets render
  inside `_plot()`, before axis limits are formatted, so autoscaling expands
  to include them (explicit `ylim` still wins). Specs serialize to metadata
  (`"significances"`) with no geometry, replayed on load like layers.


Notes:

- `unique_id` plays two roles, both per-transform (never a plot-level
  attribute): in `Aggregate` it nests aggregation (first within subject,
  then across subjects — legacy `line`/`summaryu` semantics, including the
  `(uid, x)` matrix pivot for the legacy `paired` `agg_func` branch); in
  `Identity` it nests the group key so each geometry dict is one subject's
  ordered series — the legacy `paired` connector geometry. In `KDE`/
  `Histogram`/`ECDF` it plays both: nesting only (per-subject densities,
  `Histogram` on shared per-group edges) unless `agg_func` is given, in
  which case the densities are computed per subject on a *shared grid* (kde: pooled-group
  evaluation grid, FFT-evaluated; hist: shared edges; ecdf: spline on a
  shared probability grid — the aggregated geometry is the mean quantile
  function, legacy axis swap) and re-aggregated across subjects, with
  `err_func` giving per-grid-point errors — legacy `kde`/`hist`/`ecdf`
  `agg_func` semantics. The paired plot
  therefore decomposes as `Identity(unique_id=...) + Line + Marker`, stacked
  with `Aggregate(unique_id=..., how="matrix")` for the aggregated line.
  Stacked `.add()` calls over the same data keep subject-level and
  aggregate-level layers in one plot object without seaborn-style global
  coupling: `.add()` holds raw transform objects and all layers are
  processed in a single `_process_data()` pass at plot time, so every
  layer's numbers are computed against the same grouping and never
  disagree (grouping may also be set *after* `.add()`).
- Per-x aggregation (legacy `aggline` = pandas-groupby over `(levels, x)`
  for ragged data; legacy `line` with `unique_id` = dense `(uid, x)` matrix
  aggregation for aligned data) is **unified into `Aggregate`** rather than
  being a separate transform: both legacy paths produce the same geometry
  shape, and the groupby-vs-matrix choice is an implementation strategy
  exposed as `how="auto"|"groupby"|"matrix"`, not a statistical contract.
  Geometry is always arrays so elements/resolvers index uniformly whether
  there is one value per group (bars) or one per unique x (aggline).
- The `Fit` stats layer is unified on numpy's `Polynomial` model:
  `fit_func="linear"` is a degree-1 `Polynomial` fit (scipy `linregress`
  removed; slope/intercept come from the coefficients), `"polynomial"` takes
  `degree` via `fit_args` (the `"polynimial"` `FIT_DICT` key typo is fixed),
  and sine keeps scipy `curve_fit` with its residual computation fixed.
  CI machinery is generalized to the model: `dof = n - n_params` (linear 2,
  polynomial `degree + 1`, sine 4) and `bootstrap_ci` refits the
  `Polynomial` model per resample (works for linear and polynomial; constant
  x resamples collapse to the resampled mean). `CIFunc` gains
  `"bootstrap_ci"`. Validation: a fit raises `ValueError` when the number of
  points is not greater than the number of fit parameters. Constant-x
  bifurcation: a linear/polynomial fit with all-identical x collapses to
  slope 0 / intercept `mean(y)` and the interval becomes the mean's CI
  (`t * s_err / sqrt(n)`; the prediction interval adds 1 under the root,
  `dof = n - 1`) — a constant band instead of a division by zero. Sine
  raises on constant x (its FFT-based initial guess needs x spacing).
- `Fill` is the single source of fill styling (including hatch) for density
  fill-under curves and bar faces. `ErrorBand` is strictly the center +/- error
  band. No `Connector`: paired connectors are just `Line`
  elements over the connection geometry.
- Field names intentionally match the legacy processor argument names so that
  the serialized spec can reuse `preprocess_args` (any key containing `color`,
  plus `marker`/`linestyle`/`hatch`/`barwidth` special-casing) unchanged.
