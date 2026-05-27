#!/bin/bash
#SBATCH --job-name=trap_20_dataset
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
# NOTE: the effective time/mem/cpus are set in config/runs/default.yaml (stage 20)
# and injected as sbatch CLI flags by submit_pipeline.sh, overriding these headers.
#
# Stage 2 — dataset preprocessing (plan §6.2):
#   (a) ART -> STAR -> bedtools -> labelled paired L1 FASTQ (bio tools)
#   (b) classification dataset (transcript-level split) from the L1 FASTQ
#   (c) MLM masking dataset from GENCODE transcripts (for stage 3)
#
# NOTE: the (a) ART/STAR/bedtools step is the least-certain part of the pipeline
# (scripts/divide_gff.py is reconstructed). Validate it once against a real
# RepeatMasker GFF before the full re-do. Set SKIP_BUILD=1 to reuse existing
# L1_R1/L1_R2 FASTQ and only run preprocessing.
#
# Set SKIP_CLASSIFICATION=1 to skip step (b) and re-run only the MLM masking
# dataset (e.g. when the classification dataset is already on disk).
# Set SKIP_MASKING=1 to skip step (c) and re-run only the classification dataset
# (e.g. SKIP_BUILD=1 SKIP_MASKING=1 to add the transcript_id column without
# re-running the expensive ART/STAR simulation or GENCODE re-tokenization).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"

# --- (a) simulate + align + label -----------------------------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    load_bio_modules
    activate_trap
    bash "${REPO_ROOT}/scripts/build_dataset.sh"
fi

# --- (b) classification dataset (from L1 paired FASTQ) ---------------------
# Forked workers inherit the parent RSS via CoW; SLURM charges every physical
# page to the job's cgroup. Observed: 48 workers → 554 GB (≈ 11.5 GB/worker).
# Use CPUS/6 so 48 CPUs → 8 workers → ≈ 92 GB, well within the 160 GB budget.
_DATASET_NPROC=$(( SLURM_CPUS_PER_TASK / 6 ))
_DATASET_NPROC=$(( _DATASET_NPROC < 1 ? 1 : _DATASET_NPROC ))

load_cpu_modules
activate_trap
if [[ "${SKIP_CLASSIFICATION:-0}" != "1" ]]; then
python -m trap.utils.preprocessing_sequences classification \
    --pretrained-model-path "${TOKENIZER_PATH}" \
    --builder "${L1_R1}" \
    --pair "${L1_R2}" \
    --file-format fastq \
    --k "${K}" \
    --max-position-embeddings "${MAX_POSITION_CLS}" \
    --test-split 0.2 \
    --val-split 0.2 \
    --class-threshold 10 \
    --preprocessing-dataset "${PREPROCESSING_OUT}" \
    --num-proc "${_DATASET_NPROC}" \
    --split-strategy transcript-level \
    --seed "${SEED}"
fi

# --- (c) MLM masking dataset (from GENCODE transcripts) --------------------
if [[ "${SKIP_MASKING:-0}" != "1" ]]; then
python -m trap.utils.preprocessing_sequences masking \
    --pretrained-model-path "${TOKENIZER_PATH}" \
    --builder "${GENCODE_FASTA}" \
    --file-format fasta \
    --k "${K}" \
    --max-position-embeddings "${MAX_POSITION}" \
    --chunk-size "${MAX_POSITION}" \
    --test-split 0.1 \
    --preprocessing-dataset "${MLM_PREPROCESSING_OUT}" \
    --num-proc "${_DATASET_NPROC}" \
    --split-strategy transcript-level \
    --seed "${SEED}"
fi

echo "[20_dataset] classification -> ${PREPROCESSING_OUT}/classification"
echo "[20_dataset] masking        -> ${MLM_PREPROCESSING_OUT}/masking"
