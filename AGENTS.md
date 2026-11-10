# AGENTS.md

Guidance for LLM coding agents working in this repository. Read this before making changes.

## Project

Lithos is a matplotlib-based plotting package for scientific publications, focused on
clustered/categorical and nested data (e.g. neurons per mouse, repeated measures per subject).
Python >= 3.10. BSD-3-Clause license. Versioning is dynamic via setuptools-scm (git tags) —
**never hardcode a version** anywhere.

## Toolchain: uv + ruff + ty

This project uses [uv](https://docs.astral.sh/uv/) for environment/dependency management,
[ruff](https://docs.astral.sh/ruff/) for linting/formatting and [ty](https://docs.astral.sh/ty/)
for type checking. Always run tools through uv so the pinned environment is used:

| Task | Command |
| --- | --- |
| Install/sync env (dev + test + docs + lint groups) | `uv sync` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/transforms_test.py` |
| Run one test by name | `uv run pytest tests/ -k test_name` |
| Lint | `uv run ruff check .` |
| Lint + auto-fix | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Check formatting only | `uv run ruff format --check .` |
| Type check | `uv run ty check` |

CI (`.github/workflows/ci.yaml`) runs `uv sync --locked --no-default-groups --group test --group lint`,
then `ty check` and pytest on Python 3.10-3.14. Before pushing, `uv run ruff check .`,
`uv run ty check` and `uv run pytest` must all pass. Do not edit `uv.lock` by hand — regenerate
with `uv lock` if needed.

Dev tools (pytest, ruff, ty, ipykernel, nbconvert) live in `[dependency-groups]` in `pyproject.toml`.

### Type checking (ty)

ty is configured in `pyproject.toml` under `[tool.ty]` and currently checks the `lithos/`
package only (`[tool.ty.src] include = ["lithos"]`); the test suite is not type checked yet.
Update annotations to match real runtime types rather than `# ty: ignore` comments; the codebase
uses group-key dicts (`dict[tuple, ...]`) whose keys come from `DataHolder.groups()`.

## Ruff configuration

Config lives in `pyproject.toml` under `[tool.ruff]` (line-length 120) and `[tool.ruff.lint]`.
Enabled rule families: `E4/E7/E9` (pycodestyle subset), `F` (pyflakes), `I` (isort), `UP` (pyupgrade).
`F401`/`F403` are **intentionally ignored** — `lithos/__init__.py` re-exports symbols and uses
star imports; do not "fix" those warnings in `__init__.py` files.

## Repository layout

```
lithos/                  # package source
  plotting/
    plot_class/          # user-facing classes: base_class, categorical_class, line_class
    processing/          # base_processor (shared logic) + line/categorical processors
    elements.py          # element dataclasses (Marker, Line, Bar, Fill, ErrorBar, ...) -> to_spec()
    transforms.py        # transform dataclasses (Identity, Aggregate, Density, Summary, Fit)
    matplotlib_plotter.py  # Plotter / LinePlotter / CategoricalPlotter renderers
    plot_types.py, plot_utils.py
  stats/                 # pure statistics: kde, ecdf, histogram, curve_fit, circular_stats
  types/                 # input types: Group, Subgroup, UniqueGroups (plot_input), basic_types
  utils/                 # data_generation, dataholder, metadata_utils, transforms
tests/                   # pytest suite, mirrors package layout (dataholder/, processing/, ...)
doc/                     # architecture.md, migration_plan.md (read these first!)
README.md / README.ipynb # tutorial; README.md is generated from the notebook
format_notebook.py       # notebook formatting helper
```

## New element/transform API (in progress)

The package is mid-migration from per-method plotting (`LinePlot.line()`,
`CategoricalPlot.jitter()`, ...) toward a composable element/transform API:

```python
plot.add(Aggregate(func="mean", error="sem"), Marker(), ErrorBar(), ErrorBand())
```

**Read `doc/migration_plan.md` before touching the plotting API.** Key design decisions that
must be preserved:

- Transforms own aggregation AND error computation. Elements are pure formatting and never
  compute statistics. Stacked error elements must draw from the single error geometry produced
  by the transform.
- Transform `__call__` returns `dict[group_key, geometry_dict]`; the geometry dict is the only
  source of numbers that elements render.
- `.add()` holds raw transform/element objects (no `asdict`/`to_spec` at add time); all
  computation happens in one `Plot._process_data()` pass at plot time, so grouping/columns
  may be set after `.add()`. Metadata stores pure layer specs (no geometry);
  `load_metadata` recomputes geometry by replaying `.add()` against the data.
- Element field names intentionally match legacy processor argument names so serialized specs
  can reuse `preprocess_args`.
- `Fill` is the single source of fill styling; `ErrorBand` is strictly center +/- error; paired
  connectors are just `Line` elements (no `Connector` element).
- Legacy methods stay untouched during Phase 2 ports; parity tests in `tests/processing/`
  compare `Plot.add(...)` geometry against legacy processor output.

## Conventions and rules

- Type hints throughout; `typing-extensions` is available for backports.
- Keep new code formatted: run `uv run ruff format` on files you edit.
- Import sorting (isort) is enforced — run `uv run ruff check --fix` if imports are flagged.
- Tests use plain `pytest` with class-based grouping (`TestX`) and `pytest.raises` for errors.
  Follow the existing patterns in `tests/`.
- matplotlib artists are configured through the `Plotter` classes; do not call raw `plt.*`
  functions inside processors/transforms.
- The build artifacts `build/`, `.venv/`, `lithos.egg-info/` and root `test.py` are disposable
  and gitignored — do not edit them.
- Keep public API surface exported from `lithos/__init__.py` in mind: `CategoricalPlot`,
  `LinePlot`, all elements and transforms, stats (star import), `Group`/`Subgroup`/`UniqueGroups`.

## Windows note

Development happens on Windows. Use `uv run <cmd>` (works in any shell) rather than activating
`.venv` manually. Paths in commands should use forward slashes or be quoted.
