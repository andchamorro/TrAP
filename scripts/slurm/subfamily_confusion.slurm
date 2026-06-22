#!/bin/bash
#SBATCH --job-name=trap_subfamily_confusion
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Per-raw-subfamily prediction breakdown for BOTH the SPM and Salmon classifiers
# (trap/analysis/subfamily_confusion.py; summary §17.5). Localises the dominant
# L1PA<->NEGATIVE error axis along the L1 age gradient (L1PA2 -> L1PA17 -> L1PB ->
# L1PREC2) and shows where the dropped ancient L1s (L1M*/L1ME*) land.
#
# Each model is run at its OWN padding: SPM raw-read pairs are ~63 tokens (pad
# 128); Salmon k-mer pairs are ~271 tokens (pad 280 — 128 would truncate them).
#
# Model dirs are built as <MODELS_DIR>/<MODEL_CLS>/final (HF save layout). The
# Salmon deploy run suffixes the class with its lr/epochs, so pass the exact class
# that produced macro-F1 0.664 via SALMON_MODEL_CLS:
#   SALMON_MODEL_CLS=albert.l1hs_l1pa2.v48.k17.salmon.lr5e5.e12 \
#     sbatch scripts/slurm/subfamily_confusion.slurm
# (SPM_MODEL_CLS / *_TOK_NAME are overridable the same way.)

export TRAP_SPM_EXPERIMENTAL=1                       # to load the gated .spm tokenizer
export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"  # else logger.info table is dropped

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
mkdir -p results

R1="${R1:-data/external/l1hs_l1pa2_negative.5x_R1.fq}"
R2="${R2:-data/external/l1hs_l1pa2_negative.5x_R2.fq}"
PER_SUBFAMILY="${PER_SUBFAMILY:-4000}"

# Built as <MODELS_DIR>/<MODEL_CLS>/final and <MODELS_DIR>/<TOK_NAME>.
SPM_MODEL_CLS="${SPM_MODEL_CLS:-albert.l1hs_l1pa2.v48.k17.spm}"
SALMON_MODEL_CLS="${SALMON_MODEL_CLS:-albert.l1hs_l1pa2.v48.k17.salmon}"
SPM_TOK_NAME="${SPM_TOK_NAME:-tokenizer.gencode.v48.k17.spm}"
SALMON_TOK_NAME="${SALMON_TOK_NAME:-tokenizer.gencode.v48.k17.salmon}"

SPM_MODEL="${MODELS_DIR}/${SPM_MODEL_CLS}/final"
SPM_TOK="${MODELS_DIR}/${SPM_TOK_NAME}"
SALMON_MODEL="${MODELS_DIR}/${SALMON_MODEL_CLS}/final"
SALMON_TOK="${MODELS_DIR}/${SALMON_TOK_NAME}"

echo "[subfamily_confusion] === SPM (raw-read, pad 128) ==="
python -m trap.analysis.subfamily_confusion \
    --model-path "${SPM_MODEL}" --tokenizer-path "${SPM_TOK}" \
    --r1 "${R1}" --r2 "${R2}" \
    --per-subfamily "${PER_SUBFAMILY}" --max-position-embeddings 128 \
    --out results/subfamily_confusion.spm.csv \
    || echo "[subfamily_confusion] SPM run FAILED (continuing to Salmon)"

echo "[subfamily_confusion] === Salmon (k-mer, pad 280) ==="
python -m trap.analysis.subfamily_confusion \
    --model-path "${SALMON_MODEL}" --tokenizer-path "${SALMON_TOK}" \
    --r1 "${R1}" --r2 "${R2}" \
    --per-subfamily "${PER_SUBFAMILY}" --max-position-embeddings 280 \
    --out results/subfamily_confusion.salmon.csv \
    || echo "[subfamily_confusion] Salmon run FAILED"

echo "[subfamily_confusion] done -> results/subfamily_confusion.{spm,salmon}.csv"
