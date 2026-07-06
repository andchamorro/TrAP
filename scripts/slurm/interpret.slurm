#!/bin/bash
#SBATCH --job-name=trap_interpret
#SBATCH --time=02:00:00
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
# Stage A of the model-interpretability analysis (manuscript §2.6; reviewer
# R2-Major-13). Runs trap/analysis/interpret.py:
#   * per-token Layer Integrated Gradients (the attribution behind the "relevance
#     scores" — NOT raw attention) over sampled L1HS/L1PA/NEGATIVE reads;
#   * the naive last-layer attention proxy, recorded for the contrast panel;
#   * minimap2 alignment of each read to the L1.3 consensus (L19088.1) so every
#     k-mer gets a consensus coordinate + functional domain (5'UTR/ORF1/ORF2_EN/
#     ORF2_RT/…).
# Writes results/interpret/{per_kmer_attribution.csv, read_examples.csv,
# manifest.json}; render them with notebooks/5.07-ach-model-interpretability.ipynb.
#
# Defaults to the SEQUENCE-anchored classifier (labels reflect real L1 content),
# which is the defensible model for a biological-meaning claim. Override the model
# / FASTQ via the env vars below:
#   sbatch scripts/slurm/interpret.slurm                       # seqlabel (default)
#   MODEL_CLS=albert.l1hs_l1pa2.v48.k17.salmon \
#     L1_R1=data/external/l1hs_l1pa2_negative.5x_R1.fq \
#     L1_R2=data/external/l1hs_l1pa2_negative.5x_R2.fq \
#     sbatch scripts/slurm/interpret.slurm                     # coordinate-label model

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
# minimap2 for the L1.3 consensus alignment (interpret.py skips domains if absent).
module load GCCcore/13.2.0 minimap2/2.29 2>/dev/null || module load minimap2/2.29 2>/dev/null \
    || echo "[interpret] minimap2 module not loaded — k-mers will not get L1.3 domains"
mkdir -p results/interpret

MODEL_CLS="${MODEL_CLS:-albert.l1hs_l1pa2.v48.k17.salmon.seqlabel}"
TOK_NAME="${TOK_NAME:-tokenizer.gencode.v48.k17.salmon}"
# seqlabel dataset uses the *.seqlabel FASTQ; override L1_R1/L1_R2 for other models.
L1_R1="${L1_R1:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R1.fq}"
L1_R2="${L1_R2:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R2.fq}"
L1_CONSENSUS="${L1_CONSENSUS:-data/external/L19088.1.fa}"
PER_CLASS="${PER_CLASS:-300}"
N_EXAMPLES="${N_EXAMPLES:-2}"
IG_STEPS="${IG_STEPS:-50}"
TARGET="${TARGET:-predicted}"
MAX_POS="${MAX_POS:-280}"

MODEL="${MODELS_DIR}/${MODEL_CLS}/final"
TOK="${MODELS_DIR}/${TOK_NAME}"

echo "[interpret] model=${MODEL} tok=${TOK} target=${TARGET} per_class=${PER_CLASS}"
python -m trap.analysis.interpret attribute \
    --model-path "${MODEL}" \
    --tokenizer-path "${TOK}" \
    --r1 "${L1_R1}" --r2 "${L1_R2}" \
    --l1-consensus "${L1_CONSENSUS}" \
    --max-position-embeddings "${MAX_POS}" \
    --per-class "${PER_CLASS}" \
    --n-examples "${N_EXAMPLES}" \
    --ig-steps "${IG_STEPS}" \
    --target "${TARGET}" \
    --out-dir results/interpret

echo "[interpret] done -> results/interpret/{per_kmer_attribution,read_examples}.csv + manifest.json"
