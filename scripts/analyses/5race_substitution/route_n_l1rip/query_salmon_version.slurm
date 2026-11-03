#!/bin/bash
# Route N, QUERY side -- is the LATEST salmon usable here?  Measured, not argued.
#
# THE QUESTION
# The published values were produced by C++ salmon; the pinned build for this
# benchmark is Salmon/1.10.3 (the module that built the index, and the one whose
# target set reproduces the published quant.sf exactly).  The `trap` conda env ships
# salmon 2.1.2 -- the Rust rewrite -- and the question is whether the new one can be
# used without disturbing anything already published.
#
# WHAT THE UPSTREAM MIGRATION GUIDE ESTABLISHES (so it is not re-derived here)
#  * 2.0 uses a NEW INDEX FORMAT and cannot read a C++/pufferfish index.  It detects
#    and rejects one with a clear error, in both directions.  So this arm builds its
#    own index -- from the SAME only.fa, so the TARGETS are identical and only the
#    index and the mapper differ.
#  * `quant.sf` is unchanged and drop-in, so the Figure-11 aggregator needs no change.
#  * `--validateMappings` is accepted-but-ignored in 2.x (selective alignment is the
#    default), so it is simply not passed below rather than passed and warned about.
#  * The one documented behaviour difference that could move numbers: 2.0 defaults
#    `--orphanChainSubThresh` and `--postMergeChainSubThresh` to 0.0 (align every
#    candidate), where C++ used 0.95/0.9.  Upstream reports per-transcript Pearson
#    ~0.999 and says those two flags reproduce C++ mapping counts exactly.
#
# WHY THAT LAST POINT MAKES THIS WORTH 6 SU
# ~0.999 is a claim about ordinary transcriptomes.  This index is 10,666 LINE-1
# elements that are ~99 % identical to each other, which is precisely the regime where
# "align every candidate" versus "prune sub-threshold chains" should matter most.  So
# the parity claim is tested where it is most likely to fail, on our own data:
#   A  1.10.3                                        (already run: the primary arm)
#   B  2.1.2, defaults                               (chain pruning off)
#   C  2.1.2, --orphanChainSubThresh 0.95 --postMergeChainSubThresh 0.9   (C++ parity)
# If B ~ A, the latest salmon is safe here and that is a one-line robustness statement.
# If B != A but C ~ A, the difference is the chain-pruning default and is nameable.
# If C != A, the versions are not interchangeable on repetitive references -- which
# would be worth knowing well beyond this project.
#
# NOTHING PUBLISHED IS TOUCHED.  This writes only under salmon_version/; the primary
# arm's quant.sf is read-only input.  (The read-only-tree discipline is the sibling
# project's ADR-0005 D-3: new output tree, old tree checksummed and never written.)
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_salmonver
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rnq_salmonver_%j.out
#SBATCH --error=logs/rnq_salmonver_%j.err

set -euo pipefail

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_route_n_query
MATCHED="$WORK/fastq_matched"
ONLY_FA="$WORK/salmon_ref/only.fa"
VER="$WORK/salmon_version"
THREADS="${SLURM_CPUS_PER_TASK}"

ACC="${ACC:-SRR14758878}"
SALMON_LIBTYPE="${SALMON_LIBTYPE:?set SALMON_LIBTYPE to the type measured by query_probe.slurm}"
# Absolute path, so which salmon runs is never a PATH question.
SALMON2=/scratch/user/achamorro/.conda/envs/trap/bin/salmon
BASELINE="$WORK/salmon/$ACC/only/quant.sf"       # the 1.10.3 arm, read-only here

[[ -s "$ONLY_FA" ]] || { echo "MISSING $ONLY_FA (run query_build_refs)" >&2; exit 1; }
[[ -s "$BASELINE" ]] || { echo "MISSING the 1.10.3 baseline $BASELINE (run query_salmon)" >&2; exit 1; }
[[ -x "$SALMON2" ]] || { echo "MISSING $SALMON2" >&2; exit 1; }
for mate in 1 2; do
    [[ -s "$MATCHED/${ACC}_matched_${mate}.fastq.gz" ]] || { echo "MISSING $ACC mate $mate" >&2; exit 1; }
done

mkdir -p "$VER"
echo "[$(date)] salmon under test: $("$SALMON2" --version 2>&1 | head -1)"

# ------------------------------------------------------- index (2.x format)
IDX2="$VER/only.Index.v2"
if [[ -d "$IDX2" ]] && "$SALMON2" index --help >/dev/null 2>&1 && [[ -n "$(ls -A "$IDX2" 2>/dev/null)" ]]; then
    echo "[$(date)] [skip] 2.x index exists"
else
    echo "[$(date)] salmon 2.x index from the SAME only.fa (identical targets, --kmerLen 31)"
    "$SALMON2" index -t "$ONLY_FA" -i "$IDX2" --threads "$THREADS" --kmerLen 31
fi

# ------------------------------------------------------- the two 2.x quants
run2() {
    local label="$1"; shift
    local out="$VER/$label"
    if [[ -s "$out/quant.sf" ]]; then echo "[$(date)] [skip] $label"; return 0; fi
    mkdir -p "$out"
    echo "[$(date)] $label : salmon 2.x quant $*"
    "$SALMON2" quant -q -i "$IDX2" -l "$SALMON_LIBTYPE" \
        -1 "$MATCHED/${ACC}_matched_1.fastq.gz" -2 "$MATCHED/${ACC}_matched_2.fastq.gz" \
        "$@" -o "$out" --threads "$THREADS" > "$out/salmon.out" 2> "$out/salmon.err" \
        || { echo "$label FAILED" >&2; tail -20 "$out/salmon.err" >&2; return 1; }
    local fmt mapped
    fmt=$(sed -n 's/.*"expected_format"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$out/lib_format_counts.json" | head -1)
    mapped=$(sed -n 's/.*"percent_mapped"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' \
        "$out/aux_info/meta_info.json" | head -1)
    echo "  observed_libtype=${fmt:-unknown} percent_mapped=${mapped:-NA}"
}

run2 v2_default
run2 v2_cpp_parity --orphanChainSubThresh 0.95 --postMergeChainSubThresh 0.9

# ------------------------------------------------------- compare
echo "[$(date)] comparing the three arms per locus"
/scratch/user/achamorro/.conda/envs/trap/bin/python -I \
    "$REPO/scripts/analyses/5race_substitution/route_n_l1rip/salmon_version_compare.py" \
    "$BASELINE" "$VER/v2_default/quant.sf" "$VER/v2_cpp_parity/quant.sf" \
    "$VER/salmon_version_compare.csv"

echo "[$(date)] done -> $VER/salmon_version_compare.csv"
