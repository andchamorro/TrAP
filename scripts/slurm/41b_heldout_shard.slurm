#!/bin/bash
#SBATCH --job-name=trap_41b_heldout_shard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Stage 41b -- one contiguous quarter of the held-out split per array task, so the
# four can run on the idle T4s at once. A T4 does ~91 rows/s on this model against
# the A100's ~608, so one T4 needs ~4h45 for the whole split; a quarter needs ~71
# min and fits comfortably in the 3h limit.
#
#   RUN_CONFIG=config/runs/salmon_seqlabel.yaml \
#     HELDOUT_OUT="${SCRATCH}/Python/TrAP/results/heldout_shards" \
#     sbatch --array=0-3 scripts/slurm/41b_heldout_shard.slurm
#
# RUN_CONFIG must be salmon_seqlabel.yaml -- salmon.yaml names the other dataset
# build (~89% NEGATIVE). Verify at merge: supports must be L1HS 19,499 /
# L1PA 814,814 / NEGATIVE 720,295 = 1,554,608.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
cd "${REPO_ROOT}"

N_ROWS="${HELDOUT_N_ROWS:-1554608}"
N_SHARD="${HELDOUT_N_SHARD:-4}"
I="${SLURM_ARRAY_TASK_ID:-0}"
# Ceiling division, so the last shard absorbs the remainder and the union is exact.
PER=$(( (N_ROWS + N_SHARD - 1) / N_SHARD ))
START=$(( I * PER ))
END=$(( START + PER ))

echo "[41b] MODEL_CLS=${MODEL_CLS}"
echo "[41b] PROCESSING_NAME=${PROCESSING_NAME}"
echo "[41b] shard ${I}/${N_SHARD}: rows [${START},${END})"

python -I run_heldout_shard.py \
    "${MODEL_CLS}" "${PROCESSING_NAME}" "${HELDOUT_SPLIT:-eval}" \
    "${HELDOUT_OUT:-${REPO_ROOT}/results/heldout_shards}" \
    "${SLURM_CPUS_PER_TASK}" "${HELDOUT_BATCH:-128}" "${START}" "${END}"
