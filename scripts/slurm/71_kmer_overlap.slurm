#!/bin/bash
#SBATCH --job-name=trap_71_kmer_overlap
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Utility (not part of the 00→50 pipeline) — measure how many of each read's
# canonical k-mers are present in the L1 reference, by L1 subfamily group vs
# NEGATIVE.  Re-checks the ~19% presence seen on the balanced sample, at scale
# on the full data/external/*_R*.fq reads.
#
# Reproducible: rebuilds the canonical jellyfish index (l1.jf) from L1_CORPUS if
# it does not already exist.  Override paths/knobs via env:
#   L1_JF=...           (index path; default DATA_EXTERNAL/l1hs_l1pa2.k${K}.canonical.jf)
#   READS=...           (default L1_R1)
#   SAMPLE_PER_GROUP=N  (default 5000)
#   CONSERVED=1,10,100  (count thresholds)
#
# Usage:
#   sbatch scripts/slurm/71_kmer_overlap.slurm
#   READS="${L1_R2}" sbatch scripts/slurm/71_kmer_overlap.slurm   # check R2 instead

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

# Load jellyfish via the site module names from config/hpc/grace.yaml
# ($GCC_MODULE and $JELLYFISH_MODULE are set by _common.sh from grace.yaml).
[[ -n "${GCC_MODULE:-}" ]] && module load "${GCC_MODULE}" 2>/dev/null || true
[[ -n "${JELLYFISH_MODULE:-}" ]] && module load "${JELLYFISH_MODULE}" 2>/dev/null || true
if ! command -v jellyfish >/dev/null 2>&1; then
    echo "[util_kmer_overlap] ERROR: 'jellyfish' not found (load a module or install it)." >&2
    exit 1
fi

L1_JF="${L1_JF:-${DATA_EXTERNAL}/l1hs_l1pa2.k${K}.canonical.jf}"
READS="${READS:-${L1_R1}}"
READS_PAIR="${READS_PAIR:-${L1_R2}}"   # set to "" to skip R2
SAMPLE_PER_GROUP="${SAMPLE_PER_GROUP:-5000}"
CONSERVED="${CONSERVED:-10,100}"
OUT="${REPO_ROOT}/results/diagnostics/kmer_overlap.$(basename "${READS%.*}").tsv"

# Combine R1+R2 into a single input: canonical encoding collapses R2 (reverse
# complement reads) onto the same k-mer IDs as R1, so concatenating doubles the
# sample per group without inflating the measured presence rate.
COMBINED_READS="${READS}"
if [[ -n "${READS_PAIR:-}" && -s "${READS_PAIR}" ]]; then
    COMBINED_READS="${TMPDIR}/combined_reads.fq"
    cat "${READS}" "${READS_PAIR}" > "${COMBINED_READS}"
    echo "[util_kmer_overlap] combined R1+R2 -> ${COMBINED_READS}"
fi

# --- 1. Build the canonical L1 index if absent (reproducible) --------------
if [[ ! -s "${L1_JF}" ]]; then
    echo "[util_kmer_overlap] building canonical index ${L1_JF} from ${L1_CORPUS}"
    [[ -s "${L1_CORPUS}" ]] || { echo "ERROR: L1_CORPUS not found: ${L1_CORPUS}" >&2; exit 1; }
    mkdir -p "$(dirname "${L1_JF}")"
    # -C = canonical (matches the SalmonKmerTokenizer / Salmon k-mer identity).
    jellyfish count -C -m "${K}" -s 3G -t "${SLURM_CPUS_PER_TASK}" \
        -o "${L1_JF}" "${L1_CORPUS}"
    echo "[util_kmer_overlap] jellyfish stats:"
    jellyfish stats "${L1_JF}"
else
    echo "[util_kmer_overlap] reusing existing index ${L1_JF}"
fi

# --- 2. Read ↔ index overlap by subfamily group ----------------------------
echo "[util_kmer_overlap] reads=${COMBINED_READS}  sample/group=${SAMPLE_PER_GROUP}  k=${K}"
python "${REPO_ROOT}/scripts/diagnostics/kmer_overlap.py" \
    --reads "${COMBINED_READS}" \
    --jf "${L1_JF}" \
    --k "${K}" \
    --sample-per-group "${SAMPLE_PER_GROUP}" \
    --conserved "${CONSERVED}" \
    --seed "${SEED}" \
    --out "${OUT}"

echo "[util_kmer_overlap] table -> ${OUT}"
