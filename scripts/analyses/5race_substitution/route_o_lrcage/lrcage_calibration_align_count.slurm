#!/bin/bash
# Route O (long-read CAGE calibration arm) -- align the three Sequel II runs and
# quantify them over the SAME L1EM.400 intervals used for the published 5'RACE
# reference, with the SAME featureCounts flags, so the numbers are comparable to
# both the 5'RACE reference and the rejected Iso-Seq arm.
#
# WHY THE ANNOTATION IS SHARED, NOT RE-DERIVED
# L1EM.400.saf was reconstructed once from the published counts.l1em.txt column
# headers, so its 122,626 intervals are identical to the reference's BY
# CONSTRUCTION rather than by re-derivation.  Re-deriving it here would reopen
# exactly the question the calibration is supposed to close, so this arm reads
# the same file.  Override with L1EM_ANNOT_DIR if it ever moves.
#
# The minimap2 index is shared for economy only (12 GB, ~25 min to build).
# `splice` and `splice:hq` differ in scoring, not in k/w, so one index serves
# both presets -- verified before reuse, not assumed.
#
# Deviations from scripts/5race_read_count.sh, and why:
#   * minimap2 -ax splice:hq replaces bwa mem        (long reads; PacBio CCS)
#   * --secondary=no mirrors bwa mem's output shape (one primary per read), so
#     `-M --fraction` behaves as it did on SRR4099955.
#   * every run is kept as its own BAM and its own featureCounts column: the
#     LRCAGE-vs-LRhex contrast is the whole experiment and pooling would destroy it.
#SBATCH --job-name=lrcage_align
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lrcage_align_%j.out
#SBATCH --error=logs/lrcage_align_%j.err

set -euo pipefail

module load GCC/13.2.0 minimap2/2.28 SAMtools/1.21 Subread/2.0.8

WORK=/scratch/user/achamorro/analysis/5race_lrcage_calibration
SRA=/scratch/user/achamorro/data/SRA/PRJNA936447
GENOME=/scratch/user/achamorro/Python/TrAP/data/external/GRCh38.p14.genome.fa
L1EM_ANNOT_DIR="${L1EM_ANNOT_DIR:-/scratch/user/achamorro/analysis/5race_isoseq_calibration/annotation}"
SAF="$L1EM_ANNOT_DIR/L1EM.400.saf"
MMI="$L1EM_ANNOT_DIR/GRCh38.p14.splice-hq.mmi"
THREADS="${SLURM_CPUS_PER_TASK}"

# LRCAGE DMSO, LRCAGE DACSB, LRhex DMSO -- order fixed so the featureCounts
# column order is predictable downstream.
ACCS=(SRR23517790 SRR23517787 SRR23517789)

mkdir -p "$WORK"/{align,counts,logs,report}
cd "$WORK"

[[ -s "$SAF" ]] || { echo "MISSING shared SAF $SAF" >&2; exit 1; }
if [[ ! -s "$MMI" ]]; then
    echo "[$(date)] building minimap2 splice:hq index (shared annotation dir)"
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

# Library-level L1 enrichment: the fraction of MAPPED reads that land on any
# L1EM.400 interval.  Untargeted Iso-Seq put a trickle of its reads there and
# they were passengers; a cap-trapped library has no L1 targeting either, so
# this number is expected to be small -- it is recorded because the comparison
# that matters is LRCAGE vs LRhex, not LRCAGE vs zero.
echo "[$(date)] library-level L1 assignment rate"
for ACC in "${ACCS[@]}"; do
    MAPPED=$(awk '/ primary mapped /{print $1; exit}' "$WORK/align/${ACC}.flagstat.txt")
    echo "  ${ACC}: primary_mapped=${MAPPED}"
done
grep -A4 "Successfully assigned" "$WORK/counts/counts.l1em.txt.summary" 2>/dev/null || true

echo "[$(date)] done"
wc -l "$WORK/counts/counts.l1em.txt"
