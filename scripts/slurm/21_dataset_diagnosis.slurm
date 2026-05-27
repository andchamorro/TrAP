#!/bin/bash
#SBATCH --job-name=trap_21_dataset_diag
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 21 — dataset diagnostic suite.
# Runs immediately after stage 20 (can be submitted in the afterok chain).
# CPU-only: reads datasets from disk, computes stats and warmup-steps, checks
# class balance, transcript-level leakage, and deprecation warnings.
#
# Exits non-zero if any FAIL check fires, which aborts the downstream chain
# (stage 25/30 won't start if the dataset is broken).
#
# Override paths with environment variables (same as stage 20):
#   SKIP_CLASSIFICATION=1   — skip CLS diagnostics (dataset not built)
#   MLM_PROCESSING_NAME     — MLM dataset name (default from _common.sh)
#   PROCESSING_NAME         — CLS dataset name
#   MLM_TRAINER_CONFIG      — path to mlm.json
#   CLS_TRAINER_CONFIG      — path to classification_final.json
#   MLM_TUNE_CONFIG         — path to mlm_optuna.yaml
#   NUM_GPUS                — override GPU count (default: from HPC config)

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

# Paths resolved from _common.sh exports
MLM_DATASET="${MLM_PREPROCESSING_OUT}/masking/grouped"
CLS_DATASET="${PREPROCESSING_OUT}/classification/tokenized"
HPC_CFG="${HPC_CONFIG:-${REPO_ROOT}/config/hpc/grace.yaml}"

DIAG_ARGS=(
    --mlm-dataset   "${MLM_DATASET}"
    --mlm-config    "${MLM_TRAINER_CONFIG}"
    --mlm-tune-config "${REPO_ROOT}/config/tuning/mlm_optuna.yaml"
    --hpc-config    "${HPC_CFG}"
    --max-position-mlm "${MAX_POSITION}"
    --max-position-cls "${MAX_POSITION_CLS}"
    --out-json      "${MLM_PREPROCESSING_OUT}/diagnostics/21_dataset_$(date +%Y%m%d_%H%M%S).json"
)

# Include classification dataset unless stage (b) was explicitly skipped
if [[ "${SKIP_CLASSIFICATION:-0}" != "1" ]]; then
    DIAG_ARGS+=(
        --cls-dataset   "${CLS_DATASET}"
        --cls-config    "${CLS_TRAINER_CONFIG}"
    )
fi

python -m trap.analysis.diagnostics dataset "${DIAG_ARGS[@]}"
