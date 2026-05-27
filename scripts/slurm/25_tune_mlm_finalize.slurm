#!/bin/bash
#SBATCH --job-name=trap_25_tune_mlm_fin
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Optional tuning — finalize MLM study. Single-process writer: reads the shared
# JournalFileBackend study and writes the winning hyperparameters to
# config/training/mlm.tuned.json. Submitted afterok of 25_tune_mlm by
# submit_tuning.sh. Feed the output into 30_mlm_pretrain:
#
#   MLM_TRAINER_CONFIG=config/training/mlm.tuned.json sbatch scripts/slurm/30_mlm_pretrain.slurm

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

python -m trap.modeling.tune finalize \
    --search-config "${MLM_SEARCH_CONFIG}" \
    --study-name "${STUDY_MLM}" \
    --storage "${TUNE_JOURNAL_MLM}"

echo "[25_tune_mlm_finalize] wrote config/training/mlm.tuned.json"
