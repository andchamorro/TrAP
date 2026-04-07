#!/bin/bash
# Setup TrAP conda environment.
#
# conda resolves `-e .` in environment.yml relative to the YAML file location,
# not the CWD. The editable install is therefore handled here explicitly after
# the env is created/updated.
#
# Usage:
#   bash scripts/setup_conda_env.sh                  # create 'trap' env
#   bash scripts/setup_conda_env.sh --dev             # create 'trap-dev' env
#   bash scripts/setup_conda_env.sh --update          # update existing env
#   bash scripts/setup_conda_env.sh --update --dev    # update 'trap-dev'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_FILE="envs/environment.yml"
ENV_NAME="trap"
CMD="create"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            ENV_FILE="envs/environment-dev.yml"
            ENV_NAME="trap-dev"
            shift
            ;;
        --update)
            CMD="update"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Sanity-check: flit_core requires these files to exist alongside pyproject.toml
_missing=()
for _f in pyproject.toml README.md LICENSE; do
    [[ ! -f "${REPO_ROOT}/${_f}" ]] && _missing+=("${_f}")
done
if [[ ${#_missing[@]} -gt 0 ]]; then
    echo "ERROR: Missing required files in repo root (${REPO_ROOT}):" >&2
    printf '  %s\n' "${_missing[@]}" >&2
    echo "" >&2
    echo "Run 'git pull' to fetch all tracked files before running this script." >&2
    exit 1
fi

echo "Setting up conda environment: ${ENV_NAME}"
echo "Using: ${ENV_FILE}"
echo "Repo root: ${REPO_ROOT}"
echo ""

if [[ "${CMD}" == "create" ]]; then
    conda env create --name "${ENV_NAME}" -f "${ENV_FILE}"
elif [[ "${CMD}" == "update" ]]; then
    conda env update --name "${ENV_NAME}" -f "${ENV_FILE}" --prune
fi

# Editable install must be done from repo root — conda resolves `-e .` relative
# to the YAML file's directory, which would point into envs/ instead of the package.
echo ""
echo "Installing TrAP in editable mode..."
conda run -n "${ENV_NAME}" pip install -e "${REPO_ROOT}"

# Register the Python kernel so the base-env Jupyter sees this env.
echo ""
echo "Registering Python kernel (${ENV_NAME})..."
conda run -n "${ENV_NAME}" python -m ipykernel install \
    --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# Install R packages and register the R kernel.
echo ""
echo "Installing R packages and registering R kernel..."
conda run -n "${ENV_NAME}" --no-capture-output Rscript "${REPO_ROOT}/envs/r-packages.R"

echo ""
echo "✓ Done. Activate with:  conda activate ${ENV_NAME}"
echo "  Kernels registered in your base Jupyter:"
echo "    Python (${ENV_NAME})  — ipykernel"
echo "    R (trap)              — IRkernel"
