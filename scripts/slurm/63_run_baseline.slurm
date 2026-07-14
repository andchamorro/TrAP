#!/bin/bash
#SBATCH --job-name=trap_baseline
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-44%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Generic SLURM array dispatcher for the baseline quantification methods on the synthetic
# benchmark. METHOD selects the per-cell script:
#   salmon        → scripts/sh/run_salmon.sh          (no filter; alignment-free)
#   star          → scripts/sh/align_star.sh          (BAM; run FIRST; PREP_ONLY builds index)
#   l1em          → scripts/sh/run_l1em.sh            (needs BAM)
#   tetranscripts → scripts/sh/run_tetranscripts.sh   (needs BAM)
#   htseq         → scripts/sh/run_htseq.sh           (needs BAM)
#   albertem      → scripts/sh/run_albertem.sh        (needs BAM + classifier-filtered reads)
# Each array task maps to one grid cell (POWER, DELPROB) over POWERS x DELPROBS.
#
#   METHOD=salmon bash scripts/slurm/submit_run_baseline.sh
#   METHOD=star   bash scripts/slurm/submit_run_baseline.sh          # prep(index) → align array
#
# Options (env): METHOD (required), POWERS, DELPROBS, FCOV, CHR, L1_SOURCE, OUTPUT_DIR,
#   THREADS, SKIP_EXISTING, PREP_ONLY, and method-specific: STAR_INDEX, GENOME_FA,
#   GENCODE_GTF, L1EM_PATH, L1EM_BED, TE_GTF, GENE_GTF, HTSEQ_GTF, FILTERED_DIR.

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules
activate_trap

METHOD="${METHOD:?set METHOD=salmon|star|l1em|tetranscripts|htseq|albertem}"
case "${METHOD}" in
    star) SCRIPT="scripts/sh/align_star.sh" ;;
    salmon|l1em|tetranscripts|htseq|albertem) SCRIPT="scripts/sh/run_${METHOD}.sh" ;;
    *) echo "[baseline] ERROR: unknown METHOD=${METHOD}" >&2; exit 1 ;;
esac
[[ -f "${SCRIPT}" ]] || { echo "[baseline] ERROR: missing ${SCRIPT}" >&2; exit 1; }

# Map array task -> one (POWER, DELPROB); pass to the method script as a single-cell grid.
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"; DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${PREP_ONLY:-0}" != "1" ]]; then
    read -ra _P <<< "${POWERS}"; read -ra _D <<< "${DELPROBS}"
    nD=${#_D[@]}; total=$(( ${#_P[@]} * nD )); i="${SLURM_ARRAY_TASK_ID}"
    if (( i >= total )); then echo "[baseline] array idx ${i} >= grid ${total} — nothing to do"; exit 0; fi
    export POWERS="${_P[$((i / nD))]}"; export DELPROBS="${_D[$((i % nD))]}"
    echo "[baseline] METHOD=${METHOD} task ${i}/$((total - 1)) -> POWER=${POWERS} DELPROB=${DELPROBS}"
fi

bash "${SCRIPT}"
