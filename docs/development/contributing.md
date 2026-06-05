# Contributing

## Code conventions

| Rule | Detail |
|------|--------|
| Formatter | `black` (line length 99) |
| Import order | `isort --profile black` |
| Linter | `flake8` (ignores: E731, E266, E501, C901, W503) |
| Logging | `loguru` exclusively — never stdlib `logging` |
| CLI | `typer.Typer()` with `@app.command()` and `pathlib.Path` parameters |
| Path constants | Import from `trap.config.config` — never construct manually |
| Docstrings | Google style with `Args:` / `Returns:` on public functions and classes |

## Workflow

```bash
# 1. Lint & format
make lint      # check only
make format    # apply black

# 2. Test
pytest -m "not slow" -q     # fast smoke check before committing
pytest --cov=trap            # full suite with coverage

# 3. Clean
make clean     # remove *.pyc / __pycache__
```

## Adding a new module

1. Place it under the appropriate sub-package (`trap/utils/`, `trap/modeling/`, etc.).
2. Add a module-level docstring.
3. Import path constants from `trap.config.config`, never construct `Path` manually.
4. Mirror the module with a `tests/` file using the `unit` / `integration` / `slow` markers.
5. Add an `.. automodule::` entry to the relevant `docs/api/*.rst` file.

## Building the docs

```bash
pip install -r docs/requirements.txt
cd docs && make html
# open _build/html/index.html
```

Live-reload during writing:

```bash
pip install sphinx-autobuild
cd docs && make livehtml
```
