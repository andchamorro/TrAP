#!/bin/bash
#SBATCH --job-name=trap_abundance
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-44%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Build the synthetic-benchmark abundance tables with the legacy 4.08 per-element
# normalization (scripts/python/synthetic_abundance.py --normalize). The slow step is the
# bedtools coverage of each cell's STAR BAM, so it is split two ways:
#
#   * ARRAY task  → --cache-only for ONE (POWER, DELPROB) cell: computes + caches its
#                   per-element (insert_size, effLen, coverage) to
#                   <refdir>/star/<base>/coverage_features.json (parallel, one BAM each).
#   * FINALIZE    → the aggregation: reads every cache (fast) and writes abundance.csv +
#                   abundance_methods.csv. Runs when FINALIZE=1 or when there is no array id.
#
# submit_synthetic_abundance.sh chains the array → finalize (afterok). Env:
#   SIM_MODEL (insert|transcript), POWERS, DELPROBS, CHR, FCOV, L1_SOURCE, REFDIR,
#   OUT, METHODS_OUT, NORMALIZE (default 1), plus --normalize prerequisites (STAR BAMs).

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules      # SAMtools + BEDTools (+ STAR) for --normalize
activate_trap
cd "${REPO_ROOT}"

CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"; DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
source "${REPO_ROOT}/scripts/sh/_synthetic_paths.sh"
_rd="$(experiment_results_dir "${SIM_MODEL:-transcript}")"
OUT="${OUT:-${_rd}/abundance.csv}"
METHODS_OUT="${METHODS_OUT:-${_rd}/abundance_methods.csv}"
NORM_FLAG=(); [[ "${NORMALIZE:-1}" == "1" ]] && NORM_FLAG=(--normalize)

if [[ "${FINALIZE:-0}" == "1" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Aggregation: reads the per-cell caches; computes any that are still missing.
    echo "[abundance] FINALIZE: aggregating -> ${OUT} , ${METHODS_OUT}"
    python scripts/python/synthetic_abundance.py "${NORM_FLAG[@]}" \
        --out "${OUT}" --methods-out "${METHODS_OUT}"
    exit $?
fi

# Array task → one (POWER, DELPROB) cache.
read -ra _P <<< "${POWERS}"; read -ra _D <<< "${DELPROBS}"
nD=${#_D[@]}; total=$(( ${#_P[@]} * nD )); i="${SLURM_ARRAY_TASK_ID}"
if (( i >= total )); then echo "[abundance] array idx ${i} >= grid ${total} — nothing to do"; exit 0; fi
P="${_P[$((i / nD))]}"; D="${_D[$((i % nD))]}"
echo "[abundance] cache task ${i}/$((total - 1)) -> POWER=${P} DELPROB=${D}"
python scripts/python/synthetic_abundance.py --cache-only --only-power "${P}" --only-del-prob "${D}"
