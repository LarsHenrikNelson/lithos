# CLAUDE.md

See [AGENTS.md](./AGENTS.md) for full agent guidance: build/test/lint commands (uv + ruff),
repository layout, code conventions, and the element/transform API migration rules.
Everything in AGENTS.md applies here too. Key points:

- `uv run pytest` to test; `uv run ruff check .` / `uv run ruff format .` before committing.
- Read `doc/migration_plan.md` and `doc/architecture.md` before changing the plotting API.
- Transforms compute; elements only format. `F401`/`F403` ignores in `__init__.py` are intentional.
- Versions come from setuptools-scm git tags — never hardcode versions.
