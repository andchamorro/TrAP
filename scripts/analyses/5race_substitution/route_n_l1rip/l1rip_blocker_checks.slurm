#!/bin/bash
# Route N, blockers 2 and 3 -- one job, three cheap steps.
#
# The charge account never appears here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# submit_l1rip_calibration.sh.  Mail options must be CLI flags; sbatch does not
# honour SBATCH_MAIL_USER/SBATCH_MAIL_TYPE.
#
# STEP 1  Align the published reference (SRR4099955, 34,128 reads) with the SAME
#         aligner settings and the SAME shared index as route N's own libraries.
#         It is used ONLY as the positive control for step 2's positional measure --
#         the published counts table remains the x-axis everywhere else, because it
#         is what makes the comparison drift-free.  34k reads: seconds of compute.
#
# STEP 2  BLOCKER 2.  Where along an element do reads start?  Flat means
#         positionally-random binding and the `five_prime` column is an artifact of
#         window length; a pile-up at 0 means the library is 5'-informative in
#         practice even though the protocol does not guarantee it.  5'RACE is run
#         through the identical code path, so the control and the test differ only
#         in the input BAM.
#
# STEP 3  BLOCKER 3.  The paper reports 90 loci with the top 10 carrying >60 % of
#         reads; our `-M --fraction` counts gave 2,351 loci and 22.9 %.  The paper
#         required UNIQUE assignment, so recount that way.  If unique assignment
#         lands near the published numbers, the discrepancy is a documented flag
#         choice rather than a reading error -- which converts this arm from "no
#         independent validation" into a pipeline check that passes.
#         -M --fraction stays the primary table: it is what the reference used.
#
# Sized deliberately small: nothing here is heavy, and the credit budget is finite.
#SBATCH --job-name=l1rip_blockers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/l1rip_blockers_%j.out
#SBATCH --error=logs/l1rip_blockers_%j.err

set -euo pipefail

module load GCC/13.2.0 minimap2/2.28 SAMtools/1.21 Subread/2.0.8

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_l1rip_calibration
REF_SRA=/scratch/user/achamorro/data/SRA/PRJNA341059
L1EM_ANNOT_DIR="${L1EM_ANNOT_DIR:-/scratch/user/achamorro/analysis/5race_isoseq_calibration/annotation}"
SAF="$L1EM_ANNOT_DIR/L1EM.400.saf"
MMI="$L1EM_ANNOT_DIR/GRCh38.p14.splice-hq.mmi"
THREADS="${SLURM_CPUS_PER_TASK}"
# The trap env's interpreter is called by absolute path rather than via
# `module load Anaconda3 + conda activate`, because this job also needs the GCC
# toolchain loaded for minimap2/samtools/featureCounts and Lmod can unload one when
# the other arrives.  Verified on Grace: this python is 3.12.2 with pysam 0.23.3 and
# imports cleanly under -I with no activation.
PY=/scratch/user/achamorro/.conda/envs/trap/bin/python

ACCS=(SRR12474736 SRR12474735)        # Library_8, Library_10
REF=SRR4099955

mkdir -p "$WORK"/{align,counts,report}
cd "$WORK"

[[ -s "$SAF" ]] || { echo "MISSING shared SAF $SAF" >&2; exit 1; }
[[ -s "$MMI" ]] || { echo "MISSING shared index $MMI" >&2; exit 1; }
for ACC in "${ACCS[@]}"; do
    [[ -s "$WORK/align/${ACC}.sorted.bam" ]] || { echo "MISSING $ACC BAM" >&2; exit 1; }
done

# ---------------------------------------------------------------- step 1
REF_BAM="$WORK/align/${REF}.sorted.bam"
if [[ -s "$REF_BAM" ]]; then
    echo "[$(date)] [skip] $REF_BAM"
else
    FQ="$REF_SRA/fastqs/${REF}.fastq.gz"
    [[ -s "$FQ" ]] || { echo "MISSING $FQ (fetch it on the login node)" >&2; exit 1; }
    echo "[$(date)] $REF : minimap2 -ax splice --secondary=no (positive control only)"
    minimap2 -ax splice --secondary=no -t "$THREADS" "$MMI" "$FQ" \
        | samtools sort -@ 4 -m 2G -T "$WORK/align/tmp.${REF}" -o "$REF_BAM" -
    samtools index -@ "$THREADS" "$REF_BAM"
    samtools flagstat -@ 4 "$REF_BAM" > "$WORK/align/${REF}.flagstat.txt"
    grep -E "in total|primary mapped" "$WORK/align/${REF}.flagstat.txt"
fi

# ---------------------------------------------------------------- step 2
echo "[$(date)] blocker 2 : positional profile of read 5' ends"
"$PY" -I "$REPO/scripts/analyses/5race_substitution/route_n_l1rip/l1rip_fiveprime_profile.py" \
    --saf "$SAF" \
    --bam "Library_8=$WORK/align/${ACCS[0]}.sorted.bam" \
    --bam "Library_10=$WORK/align/${ACCS[1]}.sorted.bam" \
    --bam "reference_5pRACE=$REF_BAM" \
    --hist-out "$WORK/counts/l1rip_position_hist.csv" \
    --summary-out "$WORK/counts/l1rip_position_summary.csv"

# ---------------------------------------------------------------- step 3
echo "[$(date)] blocker 3 : featureCounts with UNIQUE assignment (no -M, no --fraction)"
featureCounts -F SAF -T "$THREADS" \
    -a "$SAF" -o "$WORK/counts/counts.l1em.unique.txt" \
    "$WORK/align/${ACCS[0]}.sorted.bam" "$WORK/align/${ACCS[1]}.sorted.bam"
cat "$WORK/counts/counts.l1em.unique.txt.summary"

echo "[$(date)] done"
