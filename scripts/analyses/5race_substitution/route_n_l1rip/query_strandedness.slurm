#!/bin/bash
# Route N, QUERY side -- corroborate the IU (unstranded) call genome-wide, annotation-free.
#
# WHY THIS EXISTS
# The library was called unstranded from salmon's fragment counts (ISF vs ISR 873/868,
# 931/1047, 811/890; strand_mapping_bias 0.471-0.501).  That is a real measurement, not a
# verdict string -- but it was taken on ~3,000 fragments mapping to the L1-ONLY index, and
# the whole choice of L1EM model rests on it (run_L1EM_unstranded.sh vs run_L1EM.sh is a
# different likelihood, not a flag).  The sibling project measured strandedness genome-wide
# against exons before trusting it; this is the same check with the annotation removed.
#
# THE TEST, AND WHY IT NEEDS NO GENE ANNOTATION
# Bin the genome and, per bin, take the fraction of R1 alignments on the + strand.
#   * STRANDED library: a bin covered by one gene is read in one orientation, so the
#     fraction sits near 0 or near 1, and the across-bin distribution is BIMODAL at the
#     edges.
#   * UNSTRANDED library: each fragment's orientation is independent of transcript
#     strand, so every sufficiently-covered bin sits near 0.5 and the distribution is
#     UNIMODAL and central.
# Gene strand never has to be known -- it cancels.  Only bins with enough R1 reads are
# used, so the statistic is not dominated by sampling noise.
#
# chr1 alone, which is ~8 % of the genome and thousands of qualifying bins: ample power,
# and it reads a slice of the BAM through the index instead of all ~45 GB.
#
# Deliberately tiny: 2 cores (well inside the 8-process cap), 20 minutes, ~0.3 SU.
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_strand
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rnq_strand_%j.out
#SBATCH --error=logs/rnq_strand_%j.err

set -euo pipefail

module load GCC/12.3.0 SAMtools/1.18

WORK=/scratch/user/achamorro/analysis/5race_route_n_query
ACC="${ACC:-SRR14758878}"
BAM="$WORK/align/${ACC}/sorted_reads.bam"
REGION="${REGION:-chr1}"
BIN_BP="${BIN_BP:-10000}"
MIN_READS="${MIN_READS:-50}"
OUT="$WORK/strandedness"
mkdir -p "$OUT"

[[ -s "$BAM" ]] || { echo "MISSING $BAM (run query_align.slurm)" >&2; exit 1; }

echo "[$(date)] $ACC : per-${BIN_BP}bp R1 strand fractions over $REGION"

# -f 64  : first in pair (R1)
# -F 2308: drop unmapped(4) + secondary(256) + supplementary(2048)
# Primary R1 alignments only, so one vote per fragment.
samtools view -@ 2 -f 64 -F 2308 "$BAM" "$REGION" \
  | awk -v bin="$BIN_BP" -v minr="$MIN_READS" '
    {
        b = int($4 / bin)
        n[b]++
        # bit 16 set => this read is on the reverse strand
        if (and($2, 16) == 0) fwd[b]++
    }
    END {
        for (b in n) if (n[b] >= minr) printf "%d\t%d\t%.6f\n", b, n[b], fwd[b] / n[b]
    }' > "$OUT/${ACC}.${REGION}.bins.tsv"

n_bins=$(wc -l < "$OUT/${ACC}.${REGION}.bins.tsv")
echo "  qualifying bins (>= ${MIN_READS} R1 reads): ${n_bins}"
[[ "$n_bins" -ge 100 ]] || { echo "FAIL: too few bins to judge" >&2; exit 1; }

awk -v n="$n_bins" '
    { f = $3; s += f; ss += f * f; v[NR] = f
      if (f >= 0.40 && f <= 0.60) central++
      if (f <= 0.15 || f >= 0.85) extreme++
      # 10 equal-width buckets, to show the SHAPE and not just a mean
      h[int(f * 10) == 10 ? 9 : int(f * 10)]++ }
    END {
        mean = s / NR
        sd = sqrt(ss / NR - mean * mean)
        printf "\n  mean R1-forward fraction  : %.4f\n", mean
        printf "  sd across bins            : %.4f\n", sd
        printf "  bins in [0.40, 0.60]      : %d / %d  (%.1f %%)\n", central, NR, 100*central/NR
        printf "  bins <= 0.15 or >= 0.85   : %d / %d  (%.1f %%)\n", extreme+0, NR, 100*(extreme+0)/NR
        printf "\n  distribution shape (this is the actual test):\n"
        for (i = 0; i < 10; i++)
            printf "    %.1f-%.1f  %6d  %s\n", i/10, (i+1)/10, h[i]+0, \
                   substr("########################################################", 1, int(60*(h[i]+0)/NR))
        printf "\n  VERDICT: "
        if (central / NR > 0.80 && (extreme+0) / NR < 0.05)
            print "UNSTRANDED -- unimodal and central, consistent with the IU call"
        else if ((extreme+0) / NR > 0.30)
            print "STRANDED -- bimodal at the edges; the IU call and the unstranded L1EM model are WRONG"
        else
            print "AMBIGUOUS -- inspect the histogram before trusting either L1EM driver"
    }' "$OUT/${ACC}.${REGION}.bins.tsv" | tee "$OUT/${ACC}.${REGION}.summary.txt"

echo "[$(date)] done -> $OUT/${ACC}.${REGION}.summary.txt"
