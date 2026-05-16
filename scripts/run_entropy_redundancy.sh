#!/bin/bash
# Run the entropy/redundancy k-mer analysis locally (no SLURM).
#
# Use this on a workstation or login node for quick iteration.
# On Grace, prefer: sbatch scripts/slurm/05_entropy_redundancy.slurm
#
# Usage:
#   bash scripts/run_entropy_redundancy.sh
#   ER_K_MAX=12 ER_BOOTSTRAP=50 bash scripts/run_entropy_redundancy.sh  # fast dev run
#   ER_CONFIG=config/analysis/entropy_redundancy.yaml bash scripts/run_entropy_redundancy.sh
#
# Outputs: results/entropy_redundancy/{spectrum,ablation}.csv|parquet,
#          selection.json, manifest.json
#
# Env overrides (all optional; defaults come from the YAML config):
#   ER_CONFIG      path to KmerSpectrumConfigSchema YAML
#   ER_OUT         output directory
#   ER_SEED        master RNG seed
#   ER_K_MIN/MAX   k-mer length range
#   ER_BOOTSTRAP   bootstrap replicates (0 = skip CIs, much faster)
#   ER_FRACS       comma-separated subsample fractions
#   ER_REPLICATES  subsampling replicates per fraction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ER_CONFIG="${ER_CONFIG:-config/analysis/entropy_redundancy.yaml}"
ER_OUT="${ER_OUT:-results/entropy_redundancy}"

# Optional per-run overrides — build an inline YAML patch if any are set.
_PATCH="${REPO_ROOT}/results/.er_config_patch.$$.yaml"
trap 'rm -f "${_PATCH}"' EXIT

_write_patch() {
    local any=0
    [[ -n "${ER_SEED:-}" ]]       && { echo "seed: ${ER_SEED}";         any=1; }
    [[ -n "${ER_K_MIN:-}" ]]      && { echo "k_min: ${ER_K_MIN}";       any=1; }
    [[ -n "${ER_K_MAX:-}" ]]      && { echo "k_max: ${ER_K_MAX}";       any=1; }
    [[ -n "${ER_BOOTSTRAP:-}" ]]  && { echo "bootstrap: ${ER_BOOTSTRAP}"; any=1; }
    [[ -n "${ER_REPLICATES:-}" ]] && { echo "replicates: ${ER_REPLICATES}"; any=1; }
    echo "output_dir: ${ER_OUT}"
    return ${any}
}

mkdir -p "${ER_OUT}" results

# Write a merged config if overrides are present.
_write_patch > "${_PATCH}" 2>/dev/null || true
if [[ -s "${_PATCH}" ]]; then
    MERGED="${REPO_ROOT}/results/.er_merged.$$.yaml"
    trap 'rm -f "${_PATCH}" "${MERGED}"' EXIT
    python - <<PY
import yaml, sys
base = yaml.safe_load(open("${ER_CONFIG}"))
patch = yaml.safe_load(open("${_PATCH}"))
base.update(patch)
yaml.dump(base, open("${MERGED}", "w"))
PY
    ER_CONFIG="${MERGED}"
fi

echo "[run_entropy_redundancy] config=${ER_CONFIG}"
echo "[run_entropy_redundancy] output=${ER_OUT}"

# Step 1: fetch / verify corpora.
python scripts/data/fetch_entropy_inputs.py --out-dir data/external

# Step 2: full compute pipeline.
python -m trap.analysis run --config "${ER_CONFIG}"

echo ""
echo "[run_entropy_redundancy] artifacts:"
ls -lh "${ER_OUT}"/*.csv "${ER_OUT}"/*.json 2>/dev/null || true
