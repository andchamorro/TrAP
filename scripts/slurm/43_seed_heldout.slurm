#!/bin/bash
#SBATCH --job-name=trap_43_seed_heldout
# Inference only. One pass over 1,554,608 rows at batch 128 on one A100 took
# ~20 min for the model of record; 1h30 is ample and schedules well.
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Stage 43 — score each stage-42 seed replicate on the HELD-OUT split, so the
# Reviewer 3 Comment 5 stability table reports held-out metrics rather than the
# per-epoch ones. This stage is necessary, not redundant: the training config
# sets dataloader_persistent_workers=true, so each replicate's own
# eval_results.json carries the per-epoch split (the stage-40 cache defect), and
# reading it as a held-out number would repeat the original error.
#
#   sbatch --array=0-14%3 scripts/slurm/43_seed_heldout.slurm
#
# Run only after stage 42 finishes, or chain it:
#   sbatch --dependency=afterok:<42_jobid> --array=0-14%3 scripts/slurm/43_seed_heldout.slurm

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
cd "${REPO_ROOT}"

SEED=$(( 3470 + ${SLURM_ARRAY_TASK_ID:-0} ))
MODEL="albert.l1hs_l1pa2.v48.k17.salmon.seqlabel.seed${SEED}"
OUT="${REPO_ROOT}/results/seeds/${MODEL}"

if [[ ! -d "${MODELS_DIR}/${MODEL}/final" ]]; then
    echo "[43] ERROR: ${MODELS_DIR}/${MODEL}/final missing — did stage 42 task ${SLURM_ARRAY_TASK_ID:-0} succeed?" >&2
    exit 1
fi

echo "[43] seed=${SEED} model=${MODEL}"
mkdir -p "${OUT}"
python -I run_heldout.py \
    "${MODEL}" \
    "${PROCESSING_NAME}" \
    eval \
    "${OUT}" \
    "${SLURM_CPUS_PER_TASK}" \
    128

echo "[43] -> ${OUT}"
