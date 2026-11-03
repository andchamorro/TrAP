#!/bin/bash
# Route N (L1-ORF1p RIP nanopore calibration arm) -- align the two GridION runs
# and quantify them over the SAME L1EM.400 intervals as the published 5'RACE
# reference, with the SAME featureCounts flags.
#
# WHY -ax splice AND NOT splice:hq
# splice:hq assumes high-identity reads (PacBio CCS).  These are ONT cDNA reads,
# so the plain `splice` preset's scoring is correct.  The two presets differ in
# SCORING ONLY -- both index at k=15, w=5 (verified on Grace with minimap2/2.28,
# `mm_idx_stat` reporting "kmer size: 15; skip: 5" for each) -- so the existing
# shared index is valid here and does not need rebuilding.  map-ont would NOT be
# reusable: it indexes at w=10, and it is unspliced anyway.
#
# WHY THE ANNOTATION IS SHARED, NOT RE-DERIVED
# L1EM.400.saf was reconstructed once from the published counts.l1em.txt column
# headers, so its intervals are identical to the reference's BY CONSTRUCTION.
# Re-deriving it per route would reopen the question the calibration closes.
#
#   * --secondary=no mirrors bwa mem's output shape (one primary per read), so
#     `-M --fraction` behaves as it did on SRR4099955.
#   * the two libraries stay in separate BAMs and separate featureCounts columns:
#     they are replicates, and their agreement is one of this route's few
#     internal checks.
#SBATCH --job-name=l1rip_align
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/l1rip_align_%j.out
#SBATCH --error=logs/l1rip_align_%j.err

set -euo pipefail

module load GCC/13.2.0 minimap2/2.28 SAMtools/1.21 Subread/2.0.8

WORK=/scratch/user/achamorro/analysis/5race_l1rip_calibration
SRA=/scratch/user/achamorro/data/SRA/PRJNA657989
GENOME=/scratch/user/achamorro/Python/TrAP/data/external/GRCh38.p14.genome.fa
L1EM_ANNOT_DIR="${L1EM_ANNOT_DIR:-/scratch/user/achamorro/analysis/5race_isoseq_calibration/annotation}"
SAF="$L1EM_ANNOT_DIR/L1EM.400.saf"
MMI="$L1EM_ANNOT_DIR/GRCh38.p14.splice-hq.mmi"
THREADS="${SLURM_CPUS_PER_TASK}"

ACCS=(SRR12474736 SRR12474735)       # Library_8, Library_10

mkdir -p "$WORK"/{align,counts,logs,report}
cd "$WORK"

[[ -s "$SAF" ]] || { echo "MISSING shared SAF $SAF" >&2; exit 1; }
if [[ ! -s "$MMI" ]]; then
    echo "[$(date)] building minimap2 index (k=15,w=5; serves splice and splice:hq)"
    minimap2 -x splice -t "$THREADS" -d "$MMI" "$GENOME"
fi

for ACC in "${ACCS[@]}"; do
    FQ="$SRA/fastqs/${ACC}.fastq.gz"
    BAM="$WORK/align/${ACC}.sorted.bam"
    [[ -s "$FQ" ]] || { echo "MISSING $FQ" >&2; exit 1; }
    if [[ -s "$BAM" ]]; then echo "[skip] $BAM"; continue; fi
    echo "[$(date)] $ACC : minimap2 -ax splice --secondary=no"
    minimap2 -ax splice --secondary=no -t "$THREADS" "$MMI" "$FQ" \
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

# THE TARGETING CHECK, and the one number that separates this route from Iso-Seq.
# An ORF1p pull-down should put a large share of its MAPPED reads on L1EM.400
# intervals.  Untargeted Iso-Seq put a trickle there, and they were passengers.
# If this rate is low, the pull-down did not work on the deposited material and
# the route fails before any locus-level statistic is worth reading.
echo "[$(date)] L1 assignment rate per library"
for ACC in "${ACCS[@]}"; do
    MAPPED=$(awk '/primary mapped/{print $1; exit}' "$WORK/align/${ACC}.flagstat.txt")
    echo "  ${ACC}: primary_mapped=${MAPPED}"
done
cat "$WORK/counts/counts.l1em.txt.summary"

echo "[$(date)] done"
wc -l "$WORK/counts/counts.l1em.txt"
