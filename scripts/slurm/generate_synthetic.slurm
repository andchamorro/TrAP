#!/bin/bash
#SBATCH --job-name=trap_generate_synthetic
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-44%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# SLURM ARRAY generator: each task builds ONE grid cell (insert L1 → ART) of the
# synthetic L1-insertion benchmark. Shared inputs (chr1 transcripts, L1 elements,
# salmon index) are built ONCE by a prep job; the array workers reuse them
# (steps 0–3 are idempotent). Resumable: a cell whose reads already exist is skipped.
#
# Default --array=0-44%10 = the 45-cell grid (9 powers × 5 del-probs), 10 concurrent.
# Recommended entry chains prep → array (afterok) and sizes --array to the grid:
#   bash scripts/slurm/submit_generate_synthetic.sh
# Direct use:
#   PREP_ONLY=1 sbatch --array=0 scripts/slurm/generate_synthetic.slurm   # shared inputs only
#   sbatch scripts/slurm/generate_synthetic.slurm                         # array (assumes prep done)
#
# Options (env): POWERS, DELPROBS, FCOV, SEED, L1_MIN_LEN, L1_MAX_LEN, CHR, OUTPUT_DIR,
#   CHR1_TRANSCRIPTS, GENCODE_GTF, SKIP_EXISTING, KEEP_FASTA, PREP_ONLY, FORCE_PREP, DRY_RUN,
#   GZIP (1=gzip reads; 0=leave .fq uncompressed for faster validation).
# If you override POWERS/DELPROBS, set --array to match the cell count (the submit
# wrapper computes it for you).

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules
activate_trap
module load "${SEQKIT_MODULE:-SeqKit/2.9.0}" 2>/dev/null || echo "[gen] seqkit module not loaded — must be on PATH"
[[ -n "${ART_MODULE:-}" ]] && { module load "${ART_MODULE}" 2>/dev/null || true; }
for t in art_illumina seqkit bedtools samtools salmon python; do
    command -v "${t}" >/dev/null || echo "[gen] WARNING: ${t} not on PATH"
done

export POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
export DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"

# Array task → one (power, del_prob) cell. Prep / non-array run → the whole grid.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${PREP_ONLY:-0}" != "1" ]]; then
    read -ra _P <<< "${POWERS}"; read -ra _D <<< "${DELPROBS}"
    nD=${#_D[@]}; total=$(( ${#_P[@]} * nD )); i="${SLURM_ARRAY_TASK_ID}"
    if (( i >= total )); then echo "[gen] array idx ${i} ≥ grid ${total} — nothing to do"; exit 0; fi
    export POWERS="${_P[$((i / nD))]}"
    export DELPROBS="${_D[$((i % nD))]}"
    echo "[gen] array task ${i}/$((total - 1)) → POWER=${POWERS} DELPROB=${DELPROBS}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    source "${REPO_ROOT}/scripts/sh/_synthetic_paths.sh"
    _out="${OUTPUT_DIR:-$(experiment_dir "${SIM_MODEL:-transcript}" "${L1_SOURCE:-l1base}" "${CHR:-chr1}")}"
    echo "[gen] DRY_RUN — POWERS='${POWERS}' DELPROBS='${DELPROBS}' FCOV=${FCOV:-5} SEED=${SEED:-3469}"
    echo "  CHR=${CHR:-chr1}  OUTPUT_DIR=${_out}"
    echo "  SKIP_EXISTING=${SKIP_EXISTING:-1}  PREP_ONLY=${PREP_ONLY:-0}"
    exit 0
fi

bash "${REPO_ROOT:-$(pwd)}/scripts/sh/generate_synthetic_dataset.sh"
