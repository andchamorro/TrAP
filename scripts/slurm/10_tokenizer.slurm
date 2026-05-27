#!/bin/bash
#SBATCH --job-name=trap_10_tokenizer
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 1 — BPE k-mer tokenizer (k=17, vocab=32k) on GENCODE v48 transcripts
# (plan §6.1). Bakes the paired template + writes a manifest.
# Algorithm override: TOKENIZER_ALGORITHM=unigram sbatch ... (slower, suffix-array based)
#
# Debug mode (fast smoke-test, completes in <60s):
#   TOKENIZER_DEBUG=1 sbatch scripts/slurm/10_tokenizer.slurm
#
# Or invoke directly (no sbatch):
#   TOKENIZER_DEBUG=1 bash scripts/slurm/10_tokenizer.slurm
#   python -m trap.loaders.tokenizer train --debug --corpus ... --out ... --name ...

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

if [[ "${TOKENIZER_ALGORITHM:-bpe}" == "salmon" ]]; then
    # Salmon-consistent canonical k-mer tokenizer (index-build phase).
    # Default (Design B): pure canonical feature-hashing, no target index.
    # Set SALMON_TARGET=1 to build the L1 target index from conserved canonical
    # k-mers (Design C) via jellyfish on L1_CORPUS.
    KMER_COUNTS_ARG=""
    if [[ "${SALMON_TARGET:-0}" == "1" ]]; then
        [[ -n "${GCC_MODULE:-}" ]] && module load "${GCC_MODULE}" 2>/dev/null || true
        [[ -n "${JELLYFISH_MODULE:-}" ]] && module load "${JELLYFISH_MODULE}" 2>/dev/null || true
        L1_JF="${TMPDIR}/l1.k${K}.jf"
        L1_DUMP="${TMPDIR}/l1.k${K}.L${SALMON_MIN_COUNT:-50}.dump"
        echo "[10_tokenizer] jellyfish count -C (canonical) on ${L1_CORPUS}"
        jellyfish count -C -m "${K}" -s 2G -t "${SLURM_CPUS_PER_TASK}" \
            -o "${L1_JF}" "${L1_CORPUS}"
        jellyfish dump -c -L "${SALMON_MIN_COUNT:-50}" "${L1_JF}" > "${L1_DUMP}"
        KMER_COUNTS_ARG="--kmer-counts ${L1_DUMP} --min-count ${SALMON_MIN_COUNT:-50}"
        [[ -n "${SALMON_MAX_TARGET:-}" ]] && \
            KMER_COUNTS_ARG="${KMER_COUNTS_ARG} --max-target ${SALMON_MAX_TARGET}"
    fi
    python -m trap.loaders.tokenizer salmon-index \
        --out "${MODELS_DIR}" \
        --name "${TOKENIZER_NAME}" \
        --k "${K}" \
        --n-hash "${SALMON_NHASH:-65536}" \
        ${KMER_COUNTS_ARG} \
        --seed "${SEED}"
else
    # TOKENIZER_DEBUG=1 activates debug mode (500 seqs, vocab=200, name+=.debug).
    # Passed as an env var so the flag is read by typer's envvar= binding.
    python -m trap.loaders.tokenizer train \
        --corpus "${GENCODE_FASTA}" \
        --extra-corpus "${L1_CORPUS}" \
        --out "${MODELS_DIR}" \
        --name "${TOKENIZER_NAME}" \
        --algorithm "${TOKENIZER_ALGORITHM:-bpe}" \
        --k "${K}" \
        --vocab-size "${VOCAB}" \
        --batch-size 1024 \
        --file-format fasta \
        --seed "${SEED}"
fi

echo "[10_tokenizer] tokenizer -> ${TOKENIZER_PATH}"
