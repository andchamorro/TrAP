#!/bin/bash
# Route N, QUERY side -- build the two references the rival methods need.
#
# Neither reference exists on Grace: the HEK293 benchmark's salmon index and L1EM
# install were built in the legacy notebooks workspace and only their OUTPUT tables
# were ever archived.  Rebuilding them is therefore the first real step of Phase 2,
# and the only way the VM-Cub-1 numbers can share an x-axis definition with the
# published ones.
#
# (1) SALMON `only` INDEX -- the L1-only target set, L1EM-named so locus ids join.
#     Reconstructed from L1EM.400.bed, category 1 only (10,666 loci), with each
#     element extended 100 bp UPSTREAM IN TRANSCRIPT ORIENTATION.  That 100 bp is
#     not a guess and not a choice made here: it was MEASURED off the published
#     HEK293 quant.sf, whose 10,643 target names carry exactly ds=-100/de=0 on `+`
#     and ds=0/de=+100 on `-` against their own geneid locus.  The awk in
#     scripts/salmon_pipeline.sh omits it, so following that script would have
#     silently produced a different target set and a drifted comparison.  The build
#     is VALIDATED against the published name set below rather than trusted.
#     The `runon` index is deliberately NOT built: the Figure-11 aggregator reads
#     only `only/quant.sf`, so building it would cost time and prove nothing.
#
# (2) L1EM REFERENCE -- annotation/L1EM.400.fa plus its bwa index, over all 122,626
#     intervals (251.8 Mbp).  run_L1EM.sh HARDCODES this path and ignores any BED
#     passed on the command line, so the reference on disk is the reference used.
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_refs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rnq_refs_%j.out
#SBATCH --error=logs/rnq_refs_%j.err

set -euo pipefail

module load GCC/12.3.0 Salmon/1.10.3 BWA/0.7.17 SAMtools/1.18 BEDTools/2.31.0

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_route_n_query
GENOME="$REPO/data/external/GRCh38.p14.genome.fa"
# L1EM is a third-party checkout, not shipped with TrAP; override for another location.
L1EM_DIR="${L1EM_DIR:-$REPO/.trap/L1EM}"
L1EM_BED="$L1EM_DIR/annotation/L1EM.400.bed"
PUBLISHED_QUANT="$REPO/run/outputs/salmon/SRR3997504/only/quant.sf"
THREADS="${SLURM_CPUS_PER_TASK}"

SALMON_DIR="$WORK/salmon_ref"
mkdir -p "$SALMON_DIR" "$WORK/logs"

for f in "$GENOME" "$L1EM_BED" "$PUBLISHED_QUANT"; do
    [[ -s "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done

# ------------------------------------------------------- (1) salmon `only` index
ONLY_BED="$SALMON_DIR/only.bed"
ONLY_FA="$SALMON_DIR/only.fa"
ONLY_IDX="$SALMON_DIR/only.Index"

if [[ -s "$ONLY_IDX/info.json" ]]; then
    echo "[$(date)] [skip] salmon index already built"
else
    echo "[$(date)] only.bed : category-1 loci, element body + 100 bp 5' extension"
    awk -F'\t' '{
        split($4, a, ".")
        if (a[2] == "1") {
            split(a[3], loc, ":"); split(loc[2], r, "-")
            # 5 prime is the LOW coordinate on +, the HIGH coordinate on -.
            if ($6 == "+") print $1, r[1] - 100, r[2],       $4"_only", 0, $6
            else            print $1, r[1],       r[2] + 100, $4"_only", 0, $6
        }
    }' OFS='\t' "$L1EM_BED" > "$ONLY_BED"
    echo "  $(wc -l < "$ONLY_BED") intervals"

    echo "[$(date)] only.fa : bedtools getfasta -s -name"
    bedtools getfasta -fi "$GENOME" -bed "$ONLY_BED" -s -name -fo "$ONLY_FA"

    # VALIDATE against the published target set before anything is quantified.
    # A silent mismatch here would move the x-axis of every Salmon panel.
    echo "[$(date)] validating target names against the published HEK293 quant.sf"
    grep '^>' "$ONLY_FA" | sed 's/^>//' | sort > "$SALMON_DIR/names.built.txt"
    awk -F'\t' 'NR>1{print $1}' "$PUBLISHED_QUANT" | sort > "$SALMON_DIR/names.published.txt"
    n_built=$(wc -l < "$SALMON_DIR/names.built.txt")
    n_pub=$(wc -l < "$SALMON_DIR/names.published.txt")
    n_common=$(comm -12 "$SALMON_DIR/names.built.txt" "$SALMON_DIR/names.published.txt" | wc -l)
    echo "  built=$n_built published=$n_pub identical=$n_common"
    comm -23 "$SALMON_DIR/names.built.txt" "$SALMON_DIR/names.published.txt" \
        > "$SALMON_DIR/names.built_only.txt"
    comm -13 "$SALMON_DIR/names.built.txt" "$SALMON_DIR/names.published.txt" \
        > "$SALMON_DIR/names.published_only.txt"
    echo "  built-only=$(wc -l < "$SALMON_DIR/names.built_only.txt")" \
         "published-only=$(wc -l < "$SALMON_DIR/names.published_only.txt")"
    head -5 "$SALMON_DIR/names.built_only.txt" | sed 's/^/    + /'
    head -5 "$SALMON_DIR/names.published_only.txt" | sed 's/^/    - /'
    # The published index dropped 23 of the 10,666 category-1 loci; a rebuild that
    # recovers them is fine, one that LOSES published targets is not.
    if [[ "$(wc -l < "$SALMON_DIR/names.published_only.txt")" -gt 0 ]]; then
        echo "WARNING: the rebuild is missing targets the published index had." >&2
        echo "         Phase 2 must not proceed until this is explained." >&2
    fi

    echo "[$(date)] salmon index --kmerLen 31"
    salmon index -t "$ONLY_FA" -i "$ONLY_IDX" --threads "$THREADS" --kmerLen 31
fi

# ------------------------------------------------------- (2) L1EM reference
if [[ -s "$L1EM_DIR/annotation/L1EM.400.fa.bwt" ]]; then
    echo "[$(date)] [skip] L1EM reference already built"
else
    echo "[$(date)] L1EM.400.fa + bwa index (122,626 intervals, 251.8 Mbp)"
    cd "$L1EM_DIR"
    bash generate_L1EM_fasta_and_index.sh "$GENOME"
    ls -la annotation/L1EM.400.fa*
fi

echo "[$(date)] done"
