#!/bin/bash
#SBATCH --job-name=trap_05_entropy_redundancy
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 05 — entropy/redundancy k-mer analysis (deterministic, CPU-only).
#
# Computes the bias-corrected entropy/redundancy spectrum + rarefaction over
# GENCODE v48 transcripts and the RepeatMasker LINE-1 corpus, runs the k ->
# classification ablation on the labelled fixture reads, then applies the
# quantitative 3-gate k-selection criterion and writes tidy artifacts for the
# R notebook (notebooks/5.01-ach-entropy-redundancy-kmer-selection.ipynb):
#
#   results/entropy_redundancy/spectrum.csv     (all corpora, all k, all fracs)
#   results/entropy_redundancy/ablation.csv     (k -> macro-F1)
#   results/entropy_redundancy/selection.json   (recommended k + gate values)
#   results/entropy_redundancy/manifest.json    (git commit, seed, input SHA-256)
#
# Run from the repo root:
#   sbatch scripts/slurm/05_entropy_redundancy.slurm
#
# Overridable knobs (all have defaults from config/analysis/entropy_redundancy.yaml):
#   ER_CONFIG=config/analysis/entropy_redundancy.yaml
#   ER_SEED=3469
#   ER_K_MAX=20
#   ER_BOOTSTRAP=200
#   ER_OUT=results/entropy_redundancy
#
# Depends on: 00_fetch_references (GENCODE transcripts + L1 corpus must exist).
# Does NOT depend on the tokenizer or training stages — can run independently.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules

activate_trap

# Load Jellyfish and SeqKit AFTER conda activate so the module bins win PATH
# over any conda-env binaries.  Jellyfish 2.x uses -k; 1.x uses -m — loading
# after conda ensures the correct EasyBuild binary is found by shutil.which.
if [[ -n "${JELLYFISH_MODULE:-}" ]]; then
    module load ${GCC_MODULE} ${JELLYFISH_MODULE} 2>/dev/null \
        && echo "[05_entropy_redundancy] loaded Jellyfish: ${JELLYFISH_MODULE}" \
        || echo "[05_entropy_redundancy] WARNING: failed to load ${JELLYFISH_MODULE}; falling back to numpy"
else
    echo "[05_entropy_redundancy] JELLYFISH_MODULE not set in grace.yaml; using numpy backend"
fi

# Load seqkit for the streaming corpus path used by trap.analysis run.
if [[ -n "${SEQKIT_MODULE:-}" ]]; then
    module load ${SEQKIT_MODULE} 2>/dev/null \
        && echo "[05_entropy_redundancy] loaded SeqKit: ${SEQKIT_MODULE}" \
        || echo "[05_entropy_redundancy] WARNING: failed to load ${SEQKIT_MODULE}; seqkit must already be on PATH or trap.analysis will use the legacy path"
else
    echo "[05_entropy_redundancy] SEQKIT_MODULE not set in grace.yaml; seqkit must already be on PATH or trap.analysis will use the legacy path"
fi

ER_CONFIG="${ER_CONFIG:-${REPO_ROOT}/config/analysis/entropy_redundancy.yaml}"
ER_OUT="${ER_OUT:-${REPO_ROOT}/results/entropy_redundancy}"
mkdir -p "${ER_OUT}" "${REPO_ROOT}/logs"

echo "[05_entropy_redundancy] config=${ER_CONFIG} out=${ER_OUT}"
echo "[05_entropy_redundancy] cpus=${SLURM_CPUS_PER_TASK} jellyfish=$(command -v jellyfish 2>/dev/null || echo 'not found') seqkit=$(command -v seqkit 2>/dev/null || echo 'not found')"

# Step 1: ensure corpora are present (lightweight — skips if checksums match).
python "${REPO_ROOT}/scripts/data/fetch_entropy_inputs.py" \
    --out-dir "${DATA_EXTERNAL}" 2>&1

# Step 2: full pipeline (spectrum + ablation + select-k + manifest).
# n_jobs in the YAML config controls parallelism; the config reads
# SLURM_CPUS_PER_TASK as a fallback when n_jobs is null.
python -m trap.analysis run \
    --config "${ER_CONFIG}" 2>&1

echo "[05_entropy_redundancy] outputs:"
ls -lh "${ER_OUT}"/*.csv "${ER_OUT}"/*.json 2>/dev/null || true
echo "[05_entropy_redundancy] done"
