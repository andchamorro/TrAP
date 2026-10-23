#!/bin/bash
#SBATCH --job-name=trap_44_logreg_kmer
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=medium
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 44 — EXPLORATION ONLY (Reviewer 3, Comment 4). Multinomial logistic
# regression on hashed canonical 17-mer counts, trained on the same split as the
# model of record and scored on the same held-out split.
#
# CPU partition on purpose: the model is 65,536 x 3 weights and the cost is
# entirely Arrow I/O plus sparse matrix construction, so a GPU would idle.
#
#   sbatch scripts/slurm/44_logreg_kmer_baseline.slurm
#
# Nothing here writes into models/ or into any existing results directory, and
# the trained ALBERT model is not read at all. Output lands in
# results/baselines/logreg_kmer/.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules          # conda is a module here; activate_trap needs it on PATH
activate_trap
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/results/baselines/logreg_kmer"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-20000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
# _common.sh sets HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE but not this one, and
# recent huggingface_hub keys its offline behaviour on HF_HUB_OFFLINE. The CPU
# partitions get no WebProxy module, so any hub lookup would hang on connect.
export HF_HUB_OFFLINE=1
# trap.config.config defaults TRAP_VERBOSITY to "off", which filters out INFO
# and STAGE -- including the per-epoch running train accuracy.
export TRAP_VERBOSITY="${TRAP_VERBOSITY:-normal}"

echo "[44] OUT=${OUT}  EPOCHS=${EPOCHS}  BATCH=${BATCH}"
python scripts/python/explore_logreg_kmer_baseline.py "${OUT}" "${EPOCHS}" "${BATCH}"
echo "[44] -> ${OUT}"
