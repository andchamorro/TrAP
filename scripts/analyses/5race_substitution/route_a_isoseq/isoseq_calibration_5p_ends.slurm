#!/bin/bash
# Phase 1 (HEK-293 Iso-Seq calibration arm) -- 5'-end-restricted counts.
#
# The charge account is NOT inlined here; it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_a_isoseq/submit_isoseq_calibration.sh.
#SBATCH --job-name=isoseq_5p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/5p_%j.out
#SBATCH --error=logs/5p_%j.err

set -euo pipefail

module load Anaconda3/2025.12-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trap

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_isoseq_calibration
cd "$WORK"

# -I: the BAMs are downloaded data, so the interpreter must not pick up anything
# from their directory; the script is passed by ABSOLUTE path.  It used to be a
# repo-relative path, which only resolved because a copy had been staged under
# $WORK/scripts; after the move to scripts/analyses/5race_substitution/route_a_isoseq/ that no longer holds.
python -I "$REPO/scripts/analyses/5race_substitution/route_a_isoseq/isoseq_5p_ends.py" \
    --saf "$WORK/annotation/L1EM.400.saf" \
    --bam "$WORK/align/SRR31072049.sorted.bam" \
    --bam "$WORK/align/SRR31072050.sorted.bam" \
    --bam "$WORK/align/SRR31072047.sorted.bam" \
    --bam "$WORK/align/SRR31072048.sorted.bam" \
    --out "$WORK/counts/isoseq_5p_end_counts.csv"

wc -l "$WORK/counts/isoseq_5p_end_counts.csv"
