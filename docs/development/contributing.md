# Contributing

## Code conventions

| Rule | Detail |
|------|--------|
| Formatter | `black` (line length 99) |
| Import order | `isort --profile black` |
| Linter | `flake8` (ignores: E731, E266, E501, C901, W503, E203) |
| Logging | `loguru` exclusively — never stdlib `logging` |
| CLI | `typer.Typer()` with `@app.command()` and `pathlib.Path` parameters |
| Path constants | Import from `trap.config.config` — never construct manually |
| Docstrings | Google style with `Args:` / `Returns:` on public functions and classes |
| Canonical utilities | Reuse, never duplicate: `GenomeDataset` (`trap/loaders/dataset.py`), `try_mkdir` / `genome_file_handle` (`trap/utils/io.py`), k-mer functions (`trap/utils/kmer.py`) |
| Notebook plotting | Shared R helpers live in `scripts/R/trap_plot_helpers.R`; `source()` it rather than pasting a copy into a notebook |

## Where code goes

| Path | Holds |
|------|-------|
| `trap/` | Importable, tested library code. New workstreams get their own sub-package (for example `trap/subfam/`). |
| `scripts/slurm/` | Grace HPRC stage wrappers, named `NN_verb_object.slurm`; the number is the stage order. `scripts/slurm/legacy/` is frozen. |
| `scripts/sh/` | Shell wrappers around external tools, called by the stages. |
| `scripts/python/` | Command-line scripts that are not worth a library module. |
| `scripts/R/` | Figure scripts and the shared notebook plot helpers. |
| `scripts/data/` | Reference fetchers. |
| `scripts/diagnostics/` | Build-time and data diagnostics. |
| `scripts/analyses/` | One-off analyses whose numbers are reported, one subdirectory per analysis, each with a `README.md` naming inputs and outputs. |
| `scripts/` (root) | Existing scripts only. Their paths are cited in `docs/guides/manuscript_provenance.md` and stay put; add nothing new here. |
| `config/<area>/` | Configs named `<task>[.<variant>].yaml`. |
| `notebooks/` | `<phase>.<nn>-ach-<topic>.ipynb`. The `5.0x` notebooks are the published figure sources and are frozen. |
| `docs/decisions/` | One-page decision records: context, decision, evidence, consequences. |
| `archive/` | Retired code; see below. |

Every entry point under `scripts/` starts with a header that states its purpose, inputs,
outputs, and the stage or command that calls it.

## Comments

Comments explain *why*, in domain terms: the biological or statistical reason for a choice,
or the failure a guard prevents. They do not cite revision rounds, private plan sections or
agent instruction files, and they never name a person or a role. The mapping from scientific
questions to the analyses that answer them lives in `docs/guides/manuscript_provenance.md`.

## Published results are frozen

`docs/guides/manuscript_provenance.md` maps every manuscript table and figure to the code
that produced it, valid at tag `v1.0.0`. A change that touches a file it names must keep
the path resolvable (or record the move) and must not change any reported number. After
editing a published notebook, re-execute it and compare every exported figure with the
tagged version; a figure that is not byte- or pixel-identical blocks the change.
`docs/guides/inventory.tsv` classifies every tracked file; refresh it with
`python scripts/python/audit_repo.py audit` and validate it with `... check`.

## Archiving: move, then remove

Nothing is deleted outright. A file leaves its place only when it is reachable at a tag and
either moved to `archive/` (mirroring its original path, for example
`archive/scripts/draft/`) or listed in `docs/guides/archive_index.md` with its last commit.
Every `archive/` subdirectory has a `README.md` giving the origin commit, the successor (if
any) and why the code was retired. Commit one class of change at a time and list the
touched files in the commit body.

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
