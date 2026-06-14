#!/bin/bash
#SBATCH --job-name=trap_40_classification
# 48h: 12-epoch from-scratch run; job 18814270 timed out at 24h by ~epoch 3.
# 64G: Grace seff peak was 42.25 GB (was 128 GB). See config/runs/default.yaml §40.
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
# submit_pipeline.sh overrides --gres/--partition via CLI flags (CLI beats #SBATCH).
# To submit directly (CLI flags optional — defaults above apply):
#   RUN_CONFIG=config/runs/salmon.yaml sbatch [--gres=gpu:a40:2] scripts/slurm/40_classification.slurm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 4 — classification fine-tuning (L1HS / L1PA / NEGATIVE) from a
# RANDOM-INIT classifier (Track A: MLM pre-training is dropped — the hashed
# k-mer vocab makes MLM unlearnable; see legacy/mlm/README.md). Built from
# ALBERT_CONFIG (vocab 65541) + the salmon tokenizer, with class-weighted loss.
# PBT-final hyperparameters live in config/training/classification_final.json.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

# Track A: no --pretrained-model-path, so model_init builds a random-init
# AlbertForSequenceClassification from ALBERT_CONFIG. For the optional P3
# MLM-init ablation, add --pretrained-model-path "${MODEL_MLM}".
accelerate launch $(accelerate_args) "${REPO_ROOT}/trap/modeling/train.py" classification \
    "${MODEL_CLS}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --trainer-config-path "${CLS_TRAINER_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --do-eval

echo "[40_classification] model -> ${MODELS_DIR}/${MODEL_CLS}/final"
