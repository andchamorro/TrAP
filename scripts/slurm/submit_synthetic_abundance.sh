#!/bin/bash
# Build the synthetic-benchmark abundance tables with the legacy 4.08 per-element
# normalization, parallelised: a per-cell cache ARRAY (bedtools coverage of each STAR BAM,
# the slow step) → a single FINALIZE job (afterok) that aggregates the caches into
# abundance.csv + abundance_methods.csv. Re-runs are fast because the caches persist under
# <refdir>/star/<base>/coverage_features.json.
#
#   SIM_MODEL=insert bash scripts/slurm/submit_synthetic_abundance.sh
#   SIM_MODEL=insert bash scripts/slurm/submit_synthetic_abundance.sh --dry-run
#   SIM_MODEL=insert NORMALIZE=0 bash scripts/slurm/submit_synthetic_abundance.sh   # raw (no cache/array needed)
#
# Env: SIM_MODEL, POWERS, DELPROBS, CHR, FCOV, L1_SOURCE, REFDIR, OUT, METHODS_OUT,
#   NORMALIZE (default 1), THROTTLE (default 8), CPUS/MEM/TIME. Extra flags pass to sbatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SLURM="scripts/slurm/synthetic_abundance.slurm"

DRY_RUN=0; passthru=()
for a in "$@"; do
    case "${a}" in
        --dry-run) DRY_RUN=1 ;;
        *) passthru+=("${a}") ;;
    esac
done

export POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
export DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
export NORMALIZE="${NORMALIZE:-1}"
read -ra _P <<< "${POWERS}"; read -ra _D <<< "${DELPROBS}"
N=$(( ${#_P[@]} * ${#_D[@]} ))
THROTTLE="${THROTTLE:-8}"
CPUS="${CPUS:-2}"; MEM="${MEM:-8G}"; TIME="${TIME:-02:00:00}"
res=(--cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}")

submit() { if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] sbatch $* ${SLURM}"; else sbatch "$@" "${SLURM}"; fi; }

if [[ "${NORMALIZE}" == "1" ]]; then
    # 1) per-cell cache array (the slow bedtools step, parallel).
    arr="$(submit "${res[@]}" --parsable --array="0-$((N - 1))%${THROTTLE}" \
        --export=ALL,FINALIZE=0 "${passthru[@]+"${passthru[@]}"}")"
    echo "cache array: ${arr}"
    [[ "${DRY_RUN}" == "1" ]] && arr="<array_jobid>"
    dep="--dependency=afterok:${arr}"
else
    dep=""; echo "NORMALIZE=0 → raw counts; skipping the cache array (finalize only)."
fi

# 2) finalize: aggregate the caches into the CSVs (single task, no array).
fin_args=("${res[@]}" --parsable --job-name=trap_abundance_finalize --export=ALL,FINALIZE=1 "${passthru[@]+"${passthru[@]}"}")
[[ -n "${dep}" ]] && fin_args=("${dep}" "${fin_args[@]}")
fin="$(submit "${fin_args[@]}")"
echo "finalize (aggregate): ${fin}"
echo "grid: ${N} cells (POWERS='${POWERS}' × DELPROBS='${DELPROBS}'), NORMALIZE=${NORMALIZE}, ${CPUS} cpu / ${MEM} / ${TIME}"
