#!/bin/bash
#SBATCH --job-name=trap_35_tune_cls
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
# submit_pipeline.sh overrides --gres/--partition via CLI flags (CLI beats #SBATCH).
# To submit directly (CLI flags optional — defaults above apply):
#   RUN_CONFIG=config/runs/salmon.yaml sbatch [--gres=gpu:a40:1] scripts/slurm/35_tune_classification.slurm
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Optional tuning — Optuna classification HPO. Each array task is ONE single-GPU
# worker on the shared study $STUDY_CLS (JournalFileBackend); Optuna coordinates
# trial assignment via the shared journal, NOT DDP (see accelerate_args_single
# in _common.sh for why DDP + Hyperband pruning desyncs the ranks). 8 workers x
# TRIALS_PER_WORKER trials; %4 caps concurrency. Run the afterok `tune_finalize`
# to write config/training/classification_final.tuned.json. submit_tuning.sh
# requests 1 GPU per task; override GPU via --gres=gpu:a40:1.
#
# Wall time 48h (gpu-partition max): at SUBSAMPLE_FRAC=0.4 a full 8-epoch trial
# is ~4x the 10% run (job 18853148: longest worker 15:43h at 10% -> ~63h at 40%),
# so at 40% submit with TRIALS_PER_WORKER=1 (8 workers -> 8 trials, ~34h each,
# fits 48h). The journal is persistent: resubmit the SAME array to ADD 8 more
# TPE-informed trials to the study. mem 64G: 10% peak was 24.4 GB (seff 18853148).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
mkdir -p "${TUNE_DIR}"

# --num_processes 1 makes each trial unambiguously single-GPU (no DDP) — it uses
# the first GPU SLURM allocated and ignores any extra, so we never overwrite the
# SLURM-set CUDA_VISIBLE_DEVICES. For deep debugging of a residual CUDA gather
# OOB, submit with CUDA_LAUNCH_BLOCKING=1 (synchronous Python traceback instead
# of an async NCCL hang); leave it OFF normally — it serialises kernels.

# Optional fast-HPO subsampling (fixed-seed, stratified by label): export
# SUBSAMPLE_FRAC=0.1 or SUBSAMPLE_N=480000 before submit_tuning.sh.
# Strongly recommended here — the classification train split is ~4.8M reads.
_SUB_ARGS=()
[[ -n "${SUBSAMPLE_FRAC:-}" ]] && _SUB_ARGS+=(--subsample-frac "${SUBSAMPLE_FRAC}")
[[ -n "${SUBSAMPLE_N:-}" ]] && _SUB_ARGS+=(--subsample-n "${SUBSAMPLE_N}")

# Track A: no --pretrained-model-path, so tune.py builds a RANDOM-INIT classifier
# from ALBERT_CONFIG — matching 40_classification.slurm. (MLM init is dropped; the
# old MODEL_MLM was the deprecated 32k regime, so tuning it would not transfer.)
accelerate launch $(accelerate_args_single) "${REPO_ROOT}/trap/modeling/tune.py" classification \
    "${MODEL_CLS}" \
    --search-config "${CLS_SEARCH_CONFIG}" \
    --study-name "${STUDY_CLS}" \
    --storage "${TUNE_JOURNAL_CLS}" \
    --n-trials "${TRIALS_PER_WORKER}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" ${_SUB_ARGS[@]+"${_SUB_ARGS[@]}"}

echo "[35_tune_classification] worker ${SLURM_ARRAY_TASK_ID} done -> study ${STUDY_CLS} (${TUNE_JOURNAL_CLS})"
