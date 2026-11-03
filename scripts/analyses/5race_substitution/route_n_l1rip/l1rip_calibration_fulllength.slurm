#!/bin/bash
# Route N (L1-ORF1p RIP nanopore calibration arm) -- per-locus 5'-end and
# body-coverage counts.
#
# Route N's claim is not "5'-anchored" but "L1-targeted", so the measurement that
# decides it is how much of each element its reads actually cover, split from the
# separate question of whether they start at the promoter.  Counts are emitted per
# library so the two replicates can be compared.
#
# The charge account is NOT inlined here; it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=l1rip_fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/l1rip_fl_%j.out
#SBATCH --error=logs/l1rip_fl_%j.err

set -euo pipefail

module load Anaconda3/2025.12-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trap

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_l1rip_calibration
L1EM_ANNOT_DIR="${L1EM_ANNOT_DIR:-/scratch/user/achamorro/analysis/5race_isoseq_calibration/annotation}"
cd "$WORK"

# -I: the BAMs are built from downloaded data, so the interpreter must not pick
# up anything from their directory; the script is passed by ABSOLUTE path.
python -I "$REPO/scripts/analyses/5race_substitution/route_n_l1rip/l1rip_fulllength.py" \
    --saf "$L1EM_ANNOT_DIR/L1EM.400.saf" \
    --bam "Library_8=$WORK/align/SRR12474736.sorted.bam" \
    --bam "Library_10=$WORK/align/SRR12474735.sorted.bam" \
    --out "$WORK/counts/l1rip_fulllength_counts.csv"

wc -l "$WORK/counts/l1rip_fulllength_counts.csv"
