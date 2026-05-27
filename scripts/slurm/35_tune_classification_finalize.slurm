#!/bin/bash
#SBATCH --job-name=trap_35_tune_cls_fin
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Optional tuning — finalize classification study. Single-process writer: reads
# the shared JournalFileBackend study and writes the winning hyperparameters to
# config/training/classification_final.tuned.json. Submitted afterok of
# 35_tune_classification by submit_tuning.sh. Feed the output into 40_classification:
#
#   CLS_TRAINER_CONFIG=config/training/classification_final.tuned.json sbatch scripts/slurm/40_classification.slurm

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

python -m trap.modeling.tune finalize \
    --search-config "${CLS_SEARCH_CONFIG}" \
    --study-name "${STUDY_CLS}" \
    --storage "${TUNE_JOURNAL_CLS}"

echo "[35_tune_classification_finalize] wrote config/training/classification_final.tuned.json"
