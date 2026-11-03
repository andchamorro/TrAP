#!/bin/bash
# Route O (long-read CAGE calibration arm) -- 5'-end-restricted counts.
#
# This is the primary measurement for Route O, not a post-hoc check: the library
# is cap-trapped, so a read's 5' end is a TSS.  Counts are emitted PER RUN so the
# two internal contrasts survive -- LRCAGE vs LRhex (cap selection is the only
# difference) and LRCAGE DMSO vs DACSB (L1s driven up epigenetically).
#
# The charge account is NOT inlined here; it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_o_lrcage/submit_lrcage_calibration.sh.
#SBATCH --job-name=lrcage_5p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lrcage_5p_%j.out
#SBATCH --error=logs/lrcage_5p_%j.err

set -euo pipefail

module load Anaconda3/2025.12-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trap

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_lrcage_calibration
L1EM_ANNOT_DIR="${L1EM_ANNOT_DIR:-/scratch/user/achamorro/analysis/5race_isoseq_calibration/annotation}"
cd "$WORK"

# -I: the BAMs are built from downloaded data, so the interpreter must not pick
# up anything from their directory; the script is passed by ABSOLUTE path.
# (The Iso-Seq arm's version of this line used a repo-relative path from $WORK,
# which only resolved because a copy had been staged under $WORK/scripts.)
python -I "$REPO/scripts/analyses/5race_substitution/route_o_lrcage/lrcage_5p_ends.py" \
    --saf "$L1EM_ANNOT_DIR/L1EM.400.saf" \
    --bam "LRCAGE_DMSO=$WORK/align/SRR23517790.sorted.bam" \
    --bam "LRCAGE_DACSB=$WORK/align/SRR23517787.sorted.bam" \
    --bam "LRhex_DMSO=$WORK/align/SRR23517789.sorted.bam" \
    --out "$WORK/counts/lrcage_5p_end_counts.csv" \
    --offsets-out "$WORK/counts/lrcage_5p_end_offsets.csv"

wc -l "$WORK/counts/lrcage_5p_end_counts.csv"
