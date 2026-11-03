#!/bin/bash
# Route N, QUERY side -- genome alignment, the single BAM every alignment-dependent
# method consumes.
#
# The alignment-based methods must be compared on the SAME alignment, otherwise a
# difference between them could be a difference between aligner runs.  So there is
# exactly one bwa mem run per library and L1EM reads its output rather than aligning
# its own.  (L1EM then does its own bwa aln realignment
# of the unmapped pairs internally; that is part of L1EM, not a second alignment of
# the library.)
#
# Flags are plain `bwa mem -t N`, matching scripts/sh/bwamemAndCount.sh, which is
# what produced the published HEK293 BAM -- no multimapping flags are added, because
# bwa mem reports one primary alignment per read by design and L1EM's whole purpose
# is to resolve the multimapping itself from candidate alignments it enumerates.
# The SAM is streamed into sort instead of landing on disk: at this depth the
# intermediate would be ~200 GB and scratch has ~750 GB free.
#
# 8 cores, not 24: the project cap is 8 concurrent processes, and `bwa mem -t N` runs N
# worker threads.  That costs wall clock rather than SUs -- the charge is cores x elapsed,
# so a 3x longer run on a third of the cores is roughly the same spend -- but --time had
# to grow from 8 h to 20 h to hold 41.4 M pairs at this width.  samtools sort gets 4 of
# the 8 and 3 GB per thread, which is inside the 8 GB/core Grace allocates.
#
# Defaults to the PRIMARY replicate only.  The EM arms are the expensive ones, so
# which libraries get them is a measured decision taken after query_probe.slurm
# reports L1EM's real cost at this depth -- not a default.  To align more:
#   ACC_LIST="SRR14758878 SRR14758879 SRR14758880" ... --array=0-2
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_align
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rnq_align_%A_%a.out
#SBATCH --error=logs/rnq_align_%A_%a.err

set -euo pipefail

module load GCC/12.3.0 BWA/0.7.17 SAMtools/1.18

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_route_n_query
GENOME="$REPO/data/external/GRCh38.p14.genome.fa"
MATCHED="$WORK/fastq_matched"
THREADS="${SLURM_CPUS_PER_TASK}"

# READS=plain (depth-matched library) or READS=filtered (classifier-kept reads, the
# AlbertEM input).  Both go through this one script so the two EM arms cannot drift
# in alignment settings -- the only difference permitted between L1EM and AlbertEM is
# which reads went in.
READS="${READS:-plain}"

read -ra _ACCS <<< "${ACC_LIST:-SRR14758878}"
ACC="${_ACCS[${SLURM_ARRAY_TASK_ID:-0}]}"

case "$READS" in
    plain)
        R1="$MATCHED/${ACC}_matched_1.fastq.gz"
        R2="$MATCHED/${ACC}_matched_2.fastq.gz"
        OUT="$WORK/align/${ACC}" ;;
    filtered)
        R1="$WORK/trap/${ACC}/filtered_1.fq.gz"
        R2="$WORK/trap/${ACC}/filtered_2.fq.gz"
        OUT="$WORK/align/${ACC}_filtered" ;;
    *) echo "READS must be plain|filtered, got '$READS'" >&2; exit 1 ;;
esac

[[ -s "$GENOME.bwt" ]] || { echo "MISSING bwa index for $GENOME" >&2; exit 1; }
for f in "$R1" "$R2"; do
    [[ -s "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done

mkdir -p "$OUT"
BAM="$OUT/sorted_reads.bam"
if [[ -s "$BAM.bai" ]]; then
    echo "[$(date)] [skip] $BAM already built"
else
    echo "[$(date)] $ACC ($READS) : bwa mem -t $THREADS | samtools sort"
    t0=$SECONDS
    bwa mem -t "$THREADS" "$GENOME" "$R1" "$R2" 2> "$OUT/bwa.err" \
        | samtools sort -@ 4 -m 3G -T "$OUT/tmp.sort" -o "$BAM" -
    samtools index -@ "$THREADS" "$BAM"
    echo "  bwa mem + sort + index: $((SECONDS - t0)) s"
fi

samtools flagstat -@ 4 "$BAM" > "$OUT/flagstat.txt"
grep -E "in total|primary mapped|properly paired" "$OUT/flagstat.txt"
echo "[$(date)] done -> $BAM"
