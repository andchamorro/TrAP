#!/bin/bash
# Phase 1 (HEK-293 Iso-Seq calibration arm) -- align the Revio HiFi reads and
# quantify them over the SAME L1EM.400 intervals used for the published 5'RACE
# reference, with the SAME featureCounts flags, so that `race5` is comparable.
#
# Deviations from scripts/5race_read_count.sh, and why:
#   * minimap2 -ax splice:hq replaces bwa mem   (plan sec. 4a: long reads)
#   * --secondary=no mirrors bwa mem's output shape (one primary per read, no
#     0x100 records), so `-M --fraction` behaves as it did on SRR4099955.
#   * the SAF is reconstructed from the published counts.l1em.txt columns, so
#     the annotation is identical by construction, not by re-derivation.
# The charge account is NOT inlined here.  It is sensitive, so it is read from
# the git-ignored config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_a_isoseq/submit_isoseq_calibration.sh, which sbatch honours as the
# default --account.  Convention borrowed from Sertorius scripts/hpc/submit.sh.
#SBATCH --job-name=isoseq_align
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/align_%j.out
#SBATCH --error=logs/align_%j.err

set -euo pipefail

module load GCC/13.2.0 minimap2/2.28 SAMtools/1.21 Subread/2.0.8

WORK=/scratch/user/achamorro/analysis/5race_isoseq_calibration
SRA=/scratch/user/achamorro/data/SRA/PRJNA1176011
GENOME=/scratch/user/achamorro/Python/TrAP/data/external/GRCh38.p14.genome.fa
SAF="$WORK/annotation/L1EM.400.saf"
MMI="$WORK/annotation/GRCh38.p14.splice-hq.mmi"
THREADS="${SLURM_CPUS_PER_TASK}"

ACCS=(SRR31072049 SRR31072050 SRR31072047 SRR31072048)   # rep1 x2, rep2 x2

cd "$WORK"

if [[ ! -s "$MMI" ]]; then
    echo "[$(date)] building minimap2 splice:hq index"
    minimap2 -x splice:hq -t "$THREADS" -d "$MMI" "$GENOME"
fi

for ACC in "${ACCS[@]}"; do
    FQ="$SRA/fastqs/${ACC}.fastq.gz"
    BAM="$WORK/align/${ACC}.sorted.bam"
    [[ -s "$FQ" ]] || { echo "MISSING $FQ" >&2; exit 1; }
    if [[ -s "$BAM" ]]; then echo "[skip] $BAM"; continue; fi
    echo "[$(date)] $ACC : minimap2 -ax splice:hq --secondary=no"
    minimap2 -ax splice:hq --secondary=no -t "$THREADS" "$MMI" "$FQ" \
        | samtools sort -@ 4 -m 2G -T "$WORK/align/tmp.${ACC}" -o "$BAM" -
    samtools index -@ "$THREADS" "$BAM"
    samtools flagstat -@ 4 "$BAM" > "$WORK/align/${ACC}.flagstat.txt"
    echo "[$(date)] $ACC : aligned"
done

BAMS=()
for ACC in "${ACCS[@]}"; do BAMS+=("$WORK/align/${ACC}.sorted.bam"); done

echo "[$(date)] featureCounts -- published flags (primary table)"
featureCounts -F SAF -M --fraction -T "$THREADS" \
    -a "$SAF" -o "$WORK/counts/counts.l1em.txt" "${BAMS[@]}"

echo "[$(date)] featureCounts -- long-read mode (sensitivity check)"
featureCounts -F SAF -L -T "$THREADS" \
    -a "$SAF" -o "$WORK/counts/counts.l1em.longread.txt" "${BAMS[@]}" \
    || echo "WARNING: long-read mode failed; primary table is unaffected"

echo "[$(date)] done"
wc -l "$WORK/counts/counts.l1em.txt"
