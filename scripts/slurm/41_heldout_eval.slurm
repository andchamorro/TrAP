#!/bin/bash
#SBATCH --job-name=trap_41_heldout_eval
# Inference only: no training. The observed wall time for one pass over ~1.6M rows
# on one A100 is 44 min including model load; 1h10 leaves margin and, unlike a 2h
# request, is short enough for the backfill scheduler. 64G matches stage 40's peak.
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
# To submit directly -- the RUN_CONFIG is NOT optional, and NOT salmon.yaml:
#   RUN_CONFIG=config/runs/salmon_seqlabel.yaml \
#     HELDOUT_OUT="${SCRATCH}/Python/TrAP/results/heldout_seqlabel" \
#     sbatch scripts/slurm/41_heldout_eval.slurm
#
# Two dataset builds coexist and only one is the model of record. salmon.yaml names
# the natural-prior build (~89% NEGATIVE); salmon_seqlabel.yaml names the build
# Table 1 reports (46% NEGATIVE). Job 20215036 was submitted with salmon.yaml by
# mistake and scored the wrong model on the wrong split -- it completed cleanly and
# the error was only visible in the per-class supports. Check them before using any
# number: the held-out split of record is L1HS 19,499 / L1PA 814,814 /
# NEGATIVE 720,295 = 1,554,608 rows. Anything near 1,599,663 is the other build.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 41 — score the HELD-OUT split of the reported classifier (Reviewer 3,
# Comment 3). Trains nothing; loads ${MODEL_CLS}/final and runs one forward pass
# over the split named `eval`, which Table 1 reports as "Test (held-out)".
#
# This stage exists because stage 40 never scored that split. train.py builds the
# Trainer with eval_dataset=lm_datasets["test"] (per-epoch monitoring) and then
# calls trainer.evaluate(eval_dataset=lm_datasets["eval"]); with
# dataloader_persistent_workers=true, WeightedLossTrainer.get_eval_dataloader
# keys its cache on the string "eval" when handed a Dataset object, so the final
# call replays the monitoring loader. eval_results.json carries the evidence:
# eval_f1/eval_loss are bit-identical to the best per-epoch entry in
# trainer_state.json, while eval_samples holds the other split's row count.
#
# Single GPU on purpose: Trainer.predict under DDP pads and regathers across
# ranks, and this run must be exactly one clean pass over 1,554,608 rows.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

HELDOUT_SPLIT="${HELDOUT_SPLIT:-eval}"
HELDOUT_OUT="${HELDOUT_OUT:-${REPO_ROOT}/results/heldout}"

python -m trap.analysis heldout-eval \
    --model "${MODEL_CLS}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --split "${HELDOUT_SPLIT}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${HELDOUT_OUT}"

echo "[41_heldout_eval] tables -> ${HELDOUT_OUT}/heldout_{per_class,confusion}.${HELDOUT_SPLIT}.csv"
