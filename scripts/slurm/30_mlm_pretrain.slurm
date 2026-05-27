#!/bin/bash
#SBATCH --job-name=trap_30_mlm
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
# submit_pipeline.sh overrides --gres/--partition via CLI flags (CLI beats #SBATCH).
# To submit directly (CLI flags optional — defaults above apply):
#   RUN_CONFIG=config/runs/salmon.yaml \
#   MLM_TRAINER_CONFIG=config/training/mlm.tuned.json \
#   sbatch [--gres=gpu:a40:2] [--partition=gpu] scripts/slurm/30_mlm_pretrain.slurm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 3 — ALBERT masked-language-model pretraining on GENCODE k=17 tokens
# (plan §6.3). Default 2x A100; override GPU via --gres=gpu:a40:2.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

accelerate launch $(accelerate_args) "${REPO_ROOT}/trap/modeling/train.py" masking \
    "${MODEL_MLM}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --trainer-config-path "${MLM_TRAINER_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${MLM_PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo "[30_mlm] model -> ${MODELS_DIR}/${MODEL_MLM}/final"
