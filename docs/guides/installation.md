# Installation

## Prerequisites

- **Python 3.12.2** (strict, `~=3.12.2`)
- **Conda / Anaconda** for environment management
- A CUDA-capable GPU for training and quantification (CPU/MPS works for smoke tests)

## Install

```bash
git clone https://github.com/andchamorro/TrAP.git
cd TrAP

# Use the setup wrapper — do NOT run conda env create directly (see note below)
bash scripts/setup_conda_env.sh            # base 'trap' env
# OR: bash scripts/setup_conda_env.sh --dev  # development extras

conda activate trap
```

:::{important}
Do **not** run `conda env create -f envs/environment.yml` directly.
Conda resolves the `-e .` editable-install path relative to the YAML
file's own directory (`envs/`), not the repository root, which causes
a `pyproject.toml not found` error.
`scripts/setup_conda_env.sh` handles this by running
`pip install -e $REPO_ROOT` after env creation.
:::

## Update an existing environment

```bash
bash scripts/setup_conda_env.sh --update
# OR for the dev environment:
bash scripts/setup_conda_env.sh --update --dev
```

## Verify

```bash
conda activate trap
python -c "import trap; print('TrAP OK')"
pytest -m unit -q          # fast unit tests only
```

## Development install

The dev environment (`trap-dev`) pins all package versions and includes
linting and build tools:

```bash
bash scripts/setup_conda_env.sh --dev
conda activate trap-dev
make lint                  # flake8 + isort --check + black --check
make format                # black
```
