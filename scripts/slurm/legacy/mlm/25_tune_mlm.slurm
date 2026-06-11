#!/bin/bash
#SBATCH --job-name=trap_25_tune_mlm
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
# submit_pipeline.sh overrides --gres/--partition via CLI flags (CLI beats #SBATCH).
# To submit directly (CLI flags optional — defaults above apply):
#   RUN_CONFIG=config/runs/salmon.yaml sbatch [--gres=gpu:a40:1] scripts/slurm/25_tune_mlm.slurm
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Optional tuning — Optuna MLM HPO. Each array task is ONE single-GPU worker on
# the shared study $STUDY_MLM (parallelism via the shared Optuna journal, NOT
# DDP — see accelerate_args_single in _common.sh for why). Trials run a SHORT
# proxy schedule (max_trial_epochs in the search YAML) so a sweep is affordable
# vs the full 40-epoch pretrain; Hyperband prunes the rest. 6 workers x
# TRIALS_PER_WORKER trials. The afterok `tune_finalize` writes
# config/training/mlm.tuned.json. submit_tuning.sh requests 1 GPU per task.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
mkdir -p "${TUNE_DIR}"

# --num_processes 1 makes each trial unambiguously single-GPU (no DDP) — it uses
# the first GPU SLURM allocated and ignores any extra, so we never overwrite the
# SLURM-set CUDA_VISIBLE_DEVICES. For deep debugging of a residual CUDA gather
# OOB, submit with CUDA_LAUNCH_BLOCKING=1 (synchronous Python traceback instead
# of an async NCCL hang); leave it OFF normally — it serialises kernels.

# Optional fast-HPO subsampling (fixed-seed): export SUBSAMPLE_FRAC=0.2 or
# SUBSAMPLE_N=80000 before submit_tuning.sh; sbatch propagates the env.
_SUB_ARGS=()
[[ -n "${SUBSAMPLE_FRAC:-}" ]] && _SUB_ARGS+=(--subsample-frac "${SUBSAMPLE_FRAC}")
[[ -n "${SUBSAMPLE_N:-}" ]] && _SUB_ARGS+=(--subsample-n "${SUBSAMPLE_N}")

accelerate launch $(accelerate_args_single) "${REPO_ROOT}/trap/modeling/tune.py" masking \
    "${MODEL_MLM}" \
    --search-config "${MLM_SEARCH_CONFIG}" \
    --study-name "${STUDY_MLM}" \
    --storage "${TUNE_JOURNAL_MLM}" \
    --n-trials "${TRIALS_PER_WORKER}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --preprocessing-name "${MLM_PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" ${_SUB_ARGS[@]+"${_SUB_ARGS[@]}"}

echo "[25_tune_mlm] worker ${SLURM_ARRAY_TASK_ID} done -> study ${STUDY_MLM} (${TUNE_JOURNAL_MLM})"
