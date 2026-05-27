#!/bin/bash
#SBATCH --job-name=trap_40_classification
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
# submit_pipeline.sh overrides --gres/--partition via CLI flags (CLI beats #SBATCH).
# To submit directly (CLI flags optional — defaults above apply):
#   RUN_CONFIG=config/runs/salmon.yaml sbatch [--gres=gpu:a40:2] scripts/slurm/40_classification.slurm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 4 — classification fine-tuning (L1HS / L1PA / NEGATIVE) on top of the
# MLM-pretrained checkpoint (plan §6.4). PBT-final hyperparameters live in
# config/training/classification_final.json.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

accelerate launch $(accelerate_args) "${REPO_ROOT}/trap/modeling/train.py" classification \
    "${MODEL_CLS}" \
    --pretrained-model-path "${MODEL_MLM}" \
    --trainer-config-path "${CLS_TRAINER_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --do-eval

echo "[40_classification] model -> ${MODELS_DIR}/${MODEL_CLS}/final"
