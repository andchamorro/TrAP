#!/bin/bash
# Route N, QUERY side -- the `Salmon` arm of Figure 11, all three DMSO replicates.
#
# Salmon is the cheap method, so it is the one run in triplicate.  That is not an
# accident of budget: blocker 4 of the route (§11.4) is that G2 pairing is
# cross-study -- VM-Cub-1 Illumina RNA-seq exists but not in the ORF1p-RIP deposit --
# and the only available mitigation is to run every usable DMSO replicate and report
# their spread rather than present one library as if it were the cell line.  Doing
# that on the method that costs ~8 SU per library makes the spread measurable
# whatever is decided about the EM arms.
#
# Reads are the seeded depth-matched files, so all three are quantified at
# SRR3997504's 41,368,288 pairs and a difference between them cannot be a depth
# difference.
#
# -l is PINNED, not auto-detected: see query_probe.slurm (B) for why, and read the
# value it measured out of $WORK/probe/lib_types.tsv before submitting.  Pass it in:
#   SALMON_LIBTYPE=ISR bash scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh \
#       scripts/analyses/5race_substitution/route_n_l1rip/query_salmon.slurm
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_salmon
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/rnq_salmon_%A_%a.out
#SBATCH --error=logs/rnq_salmon_%A_%a.err

set -euo pipefail

module load GCC/12.3.0 Salmon/1.10.3

# ONE salmon for both arms, asserted rather than hoped for.  The trap conda env ships
# salmon 2.1.2 and the module tree 1.10.3; the index was built by 1.10.3, and
# AlbertSalmon - Salmon is a PRIMARY endpoint, so a version difference between the two
# arms would be read as a method effect.  (It would probably not even fail loudly: the
# puffer index format is version-sensitive, so the failure mode is either a hard error
# or, worse, different numbers.)  This is the same hazard the library-type pin exists
# for, one layer down.
SALMON_REQUIRED="${SALMON_REQUIRED:-salmon 1.10.3}"
_salmon_seen="$(salmon --version 2>&1 | head -1)"
if [[ "$_salmon_seen" != "$SALMON_REQUIRED" ]]; then
    echo "FAIL: salmon is '$_salmon_seen', the pinned build is '$SALMON_REQUIRED'" >&2
    echo "      (which salmon: $(command -v salmon))" >&2
    exit 1
fi
echo "salmon pinned: $_salmon_seen at $(command -v salmon)"

WORK=/scratch/user/achamorro/analysis/5race_route_n_query
MATCHED="$WORK/fastq_matched"
IDX="$WORK/salmon_ref/only.Index"
THREADS="${SLURM_CPUS_PER_TASK}"

# No default: an unpinned library type is the hazard this flag exists to remove.
SALMON_LIBTYPE="${SALMON_LIBTYPE:?set SALMON_LIBTYPE to the type measured by query_probe.slurm}"
# --validateMappings matches scripts/salmon_pipeline.sh, which produced the published
# HEK293 quant.sf.  (--sketch is the documented fallback if salmon panics extending
# mappings near a short reference's 3' end; it did not on the probe.)
SALMON_FLAGS="${SALMON_FLAGS:---validateMappings}"

ACCS=(SRR14758878 SRR14758879 SRR14758880)
ACC=${ACCS[$SLURM_ARRAY_TASK_ID]}
OUT="$WORK/salmon/$ACC/only"

[[ -s "$IDX/info.json" ]] || { echo "MISSING salmon index $IDX" >&2; exit 1; }
for mate in 1 2; do
    [[ -s "$MATCHED/${ACC}_matched_${mate}.fastq.gz" ]] || { echo "MISSING $ACC mate $mate" >&2; exit 1; }
done

mkdir -p "$OUT"
echo "[$(date)] $ACC : salmon quant -l $SALMON_LIBTYPE $SALMON_FLAGS"
salmon quant -q -i "$IDX" -l "$SALMON_LIBTYPE" \
    -1 "$MATCHED/${ACC}_matched_1.fastq.gz" -2 "$MATCHED/${ACC}_matched_2.fastq.gz" \
    ${SALMON_FLAGS} -o "$OUT" --threads "$THREADS" \
    > "$OUT/salmon.out" 2> "$OUT/salmon.err" \
    || { echo "salmon FAILED" >&2; tail -20 "$OUT/salmon.err" >&2; exit 1; }

# Requested vs observed side by side, so a wrong pin shows up in a log and not in
# the abundances.
fmt=$(sed -n 's/.*"expected_format"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$OUT/lib_format_counts.json" | head -1)
mapped=$(sed -n 's/.*"percent_mapped"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' \
    "$OUT/aux_info/meta_info.json" | head -1)
nonzero=$(awk -F'\t' 'NR>1 && $5>0' "$OUT/quant.sf" | wc -l)
echo "lib_type requested=$SALMON_LIBTYPE observed=${fmt:-unknown}" | tee "$OUT/lib_type.txt"
echo "[$(date)] $ACC : percent_mapped_to_L1=${mapped} targets_with_reads=${nonzero} of $(($(wc -l < "$OUT/quant.sf") - 1))"
echo "[$(date)] done -> $OUT/quant.sf"
