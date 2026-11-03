#!/bin/bash
# Route N, QUERY side -- Phase 3, the TrAP classifier arm.
#
# Produces the classifier-filtered read pair that BOTH Albert arms consume:
#   AlbertSalmon = salmon on the filtered reads      (run here, step 5)
#   AlbertEM     = L1EM on the filtered alignment    (query_align.slurm READS=filtered
#                                                     then query_l1em.slurm ARM=filtered)
# Steps 1-4 are the recipe in scripts/slurm/62_synthetic_validation.slurm, pointed at a
# real paired library instead of a synthetic grid cell.
#
# MODEL CHOICE, AND AN ASYMMETRY THAT MUST BE DECLARED
# Plan §7b fixes the model at albert.l1hs_l1pa2.v48.k17.salmon.seqlabel with k=17 and
# the salmon tokenizer.  The PUBLISHED HEK293 AlbertSalmon column was produced by the
# older k=18 repeatmasker classifier (run/outputs/quantify_k18_SRR3997504).  That is
# fine for the primary endpoints, which are the paired WITHIN-library contrasts
# (AlbertSalmon - Salmon, AlbertEM - L1EM) and never compare a VM-Cub-1 Albert value to
# a HEK293 Albert value -- but it does mean HEK293's numbers are not a comparator for
# the Albert columns, only for locus counts and dynamic range.  Say so in Methods.
#
# COST CONTROL, with the sister project's measured rate as the starting point
# Sertorius ran this same classifier on 1x A100 at batch 256 with 15 workers / 16 cpus
# and logged a STEADY 2,064 reads/s -- 33,106,699 reads in 16,037 s, whole job 4:04-5:11
# across 12 libraries (/scratch/user/achamorro/Sertorius/logs/trap_classify_19810023_*).
# Their jobs billed at `billing=16`, i.e. GPU work is charged on the CPU count here, not
# a GPU multiplier, so the exposure is cores x elapsed like everything else.
# Two unknowns that rate does not settle, which is why the probe still runs:
#   * whether 41,368,288 PAIRS means 41.4 M or 82.7 M sequences through the model -- a
#     factor of two on the single most expensive step;
#   * whether 7 dataloader workers (8 cpus, the process cap) still keep an A100 fed,
#     where theirs had 15.
# MAX_PAIRS subsets the library before tokenisation, so the SAME script is the probe and
# the full run.  Measure first (MAX_PAIRS=1000000), read the reported pairs/s, then set
# --time from it rather than from a guess.
# --batch-size is raised off the 64 default purely for throughput; inference is
# deterministic in it, so it changes the clock and not the scores.
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_trap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --output=logs/rnq_trap_%j.out
#SBATCH --error=logs/rnq_trap_%j.err

set -euo pipefail

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
module load "${SEQKIT_MODULE:-SeqKit/2.9.0}" 2>/dev/null || echo "[rnq_trap] seqkit not loaded as a module"

# ---------------------------------------------------------------- torch.compile / Triton
# predict.py calls torch.compile(mode="reduce-overhead") on CUDA, which makes Triton
# compile a launcher with the system gcc.  That failed on the first probe (job 20164844,
# 20 min in, after tokenisation had already completed):
#     __triton_launcher.c:2:10: fatal error: cuda.h: No such file or directory
# Diagnosis: a compiler was present (GCCcore/12.3.0/bin/gcc); the CUDA HEADER was not on
# the include path.  Triton resolved its CUDA include dir to $CONDA_PREFIX/targets/... ,
# which does not exist in this env, and its own bundled
# triton/backends/nvidia/include is absent from this triton build.  The real header is in
# the CUDA module: $EBROOTCUDA/include/cuda.h.  gcc honours CPATH, so pointing CPATH at
# it is the whole fix -- set AFTER activate_trap so conda cannot win.
if [[ -n "${EBROOTCUDA:-}" ]]; then
    export CUDA_HOME="$EBROOTCUDA"
    export CPATH="${EBROOTCUDA}/include${CPATH:+:$CPATH}"
fi

# Prove it in two seconds rather than discovering it again twenty minutes in, and fall
# back to eager execution instead of crashing.  torch.compile is a throughput
# optimisation; correctness does not depend on it, so a missing header must not cost a
# whole GPU run.
_cc_probe="${TMPDIR:-/tmp}/cuda_h_probe_$$.c"
printf '#include <cuda.h>\nint main(void){return 0;}\n' > "$_cc_probe"
if gcc -fsyntax-only "$_cc_probe" 2>/dev/null; then
    echo "[rnq_trap] cuda.h resolves -- torch.compile(reduce-overhead) left enabled"
else
    export TORCHDYNAMO_DISABLE=1
    echo "[rnq_trap] WARNING: cuda.h does not resolve; TORCHDYNAMO_DISABLE=1 set." >&2
    echo "[rnq_trap]   Inference runs EAGER -- correct but slower; budget more walltime." >&2
fi
rm -f "$_cc_probe"
# Loaded AFTER activate_trap so the module's salmon wins the PATH over the conda env's.
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

ACC="${ACC:-SRR14758878}"
TAU="${TAU:-0.5}"                 # keep reads with P(NEGATIVE) < TAU
SEQ_MODEL="${SEQ_MODEL:-albert.l1hs_l1pa2.v48.k17.salmon.seqlabel}"
TOK="${TOK:-tokenizer.gencode.v48.k17.salmon}"
K="${K:-17}"
BATCH="${BATCH:-256}"
MAX_PAIRS="${MAX_PAIRS:-0}"       # 0 = whole library
SALMON_LIBTYPE="${SALMON_LIBTYPE:?set SALMON_LIBTYPE to the type measured by query_probe.slurm}"
SALMON_FLAGS="${SALMON_FLAGS:---validateMappings}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

TAG="$ACC"; [[ "$MAX_PAIRS" != "0" ]] && TAG="${ACC}_probe${MAX_PAIRS}"
OUT="$WORK/trap/$TAG"
SALMON_OUT="$WORK/salmon/$TAG/filtered/only"
mkdir -p "$OUT" "$SALMON_OUT"

# --- 0. reads ---------------------------------------------------------------
# The reads are parsed twice (tokenise, then extract), so they are expanded once
# into WORK rather than gunzipped on each pass.
R1="$OUT/reads_1.fq"; R2="$OUT/reads_2.fq"
# Depth of the matched library, read from the manifest the depth-match step wrote rather
# than hardcoded, so a re-matched library cannot silently desynchronise the proportion.
MATCHED_PAIRS=$(awk -F'\t' 'NR==2{print $3}' "$MATCHED/${ACC}.depth_match.tsv")
[[ -n "$MATCHED_PAIRS" ]] || { echo "MISSING $MATCHED/${ACC}.depth_match.tsv" >&2; exit 1; }
for mate in 1 2; do
    src="$MATCHED/${ACC}_matched_${mate}.fastq.gz"
    dst="$OUT/reads_${mate}.fq"
    [[ -s "$src" ]] || { echo "MISSING $src" >&2; exit 1; }
    if [[ -s "$dst" ]]; then echo "[rnq_trap] (0) reads_${mate}.fq exists -- SKIP"; continue; fi
    if [[ "$MAX_PAIRS" == "0" ]]; then
        gzip -dc "$src" > "$dst"
    else
        # A SEEDED RANDOM sample, not `seqkit head`.  Head-of-file on an Illumina run is
        # the flowcell edge and is enriched for low-quality reads.  For raw GPU
        # throughput that would not matter much -- reads are a fixed 150 bp, so the
        # per-read tokenise+forward cost is constant -- but the RETAINED FRACTION at
        # tau is a property of the reads themselves, and measuring it on the worst
        # reads in the file would misreport the one number this probe exists to
        # estimate.  Same seed and proportion on both mates keeps the pair intact,
        # exactly as the depth-match step does.
        frac=$(awk -v t="$MAX_PAIRS" -v n="$MATCHED_PAIRS" 'BEGIN{ if (t>=n) print 1; else printf "%.8f", t/n }')
        seqkit sample -j 2 -s 3469 -p "$frac" "$src" > "$dst"
    fi
done
n_pairs=$(( $(wc -l < "$R1") / 4 ))
echo "[rnq_trap] $ACC tag=$TAG pairs=$n_pairs tau=$TAU model=$SEQ_MODEL batch=$BATCH"

# --- 1. tokenize ------------------------------------------------------------
if [[ "$SKIP_EXISTING" == "1" && -d "$OUT/short_reads/tokenized" ]]; then
    echo "[rnq_trap] (1) tokenized dataset exists -- SKIP"
else
    echo "[rnq_trap] (1) processing-dataset"
    t0=$SECONDS
    python -m trap.modeling.predict processing-dataset \
        --input-file "$R1" --pair-file "$R2" --output-path "$OUT" \
        --k "$K" --pretrained-tokenizer-name "${MODELS_DIR}/${TOK}" \
        --num-workers "${SLURM_CPUS_PER_TASK}" \
        || { echo "[rnq_trap] processing-dataset FAILED"; exit 1; }
    echo "  tokenise: $((SECONDS - t0)) s for $n_pairs pairs"
fi

# --- 2. classify (GPU) ------------------------------------------------------
if [[ "$SKIP_EXISTING" == "1" && -n "$(ls "$OUT"/scores_*.parquet 2>/dev/null)" ]]; then
    echo "[rnq_trap] (2) scores exist -- SKIP"
else
    echo "[rnq_trap] (2) quantify"
    t0=$SECONDS
    python -m trap.modeling.predict quantify \
        --pretrained-model-name "${MODELS_DIR}/${SEQ_MODEL}/final" \
        --output-path "$OUT" --is-tokenized --batch-size "$BATCH" \
        --num-workers "$(( SLURM_CPUS_PER_TASK - 1 ))" \
        || { echo "[rnq_trap] quantify FAILED"; exit 1; }
    dt=$((SECONDS - t0))
    echo "  classify: ${dt} s -> $(awk -v n="$n_pairs" -v d="$dt" 'BEGIN{if(d>0) printf "%.0f", n/d}') pairs/s"
    echo "  extrapolated to 41,368,288 pairs: $(awk -v n="$n_pairs" -v d="$dt" 'BEGIN{if(n>0) printf "%.1f", d*41368288/n/3600}') GPU-h"
fi

# --- 3. filter --------------------------------------------------------------
echo "[rnq_trap] (3) filter_ids (P(NEGATIVE) < $TAU)"
python -m trap.modeling.postprocessing filter-ids \
    --output-path "$OUT" --threshold "$TAU" \
    --fastq "$R1" --output-filtered-ids "$OUT/keep_ids.txt" \
    || { echo "[rnq_trap] filter_ids FAILED"; exit 1; }

# --- 4. extract the kept pairs ---------------------------------------------
# Real Illumina ids have no /1,/2 mate marker (unlike the ART synthetic reads 62_*
# handles), so the id list applies to both mates unchanged.  Both are checked.
# Two normalisations, both defensive rather than assumed:
#  - cut at the first whitespace, because `seqkit grep -f` matches the record NAME and
#    a score id carrying the FASTQ description would match nothing;
#  - strip a trailing /1 or /2, which real Illumina ids do not carry but ART-style
#    synthetic ids (what 62_synthetic_validation.slurm handles) do.
cut -d' ' -f1 "$OUT/keep_ids.txt" | sed 's,/[12]$,,' | sort -u > "$OUT/keep_base.txt"
n_keep=$(wc -l < "$OUT/keep_base.txt")
[[ "$n_keep" -gt 0 ]] || { echo "FAIL: the classifier kept 0 fragments at tau=$TAU" >&2; exit 1; }
echo "[rnq_trap] (4) seqkit grep -- $n_keep of $n_pairs fragments kept ($(awk -v a="$n_keep" -v b="$n_pairs" 'BEGIN{printf "%.2f", 100*a/b}') %)"
printf 'accession\ttag\tpairs_in\tpairs_kept\tretained_pct\ttau\tmodel\n' > "$OUT/filter_summary.tsv"
printf '%s\t%s\t%s\t%s\t%.4f\t%s\t%s\n' "$ACC" "$TAG" "$n_pairs" "$n_keep" \
    "$(awk -v a="$n_keep" -v b="$n_pairs" 'BEGIN{print 100*a/b}')" "$TAU" "$SEQ_MODEL" \
    >> "$OUT/filter_summary.tsv"

seqkit grep -f "$OUT/keep_base.txt" "$R1" -o "$OUT/filtered_1.fq"
seqkit grep -f "$OUT/keep_base.txt" "$R2" -o "$OUT/filtered_2.fq"
k1=$(( $(wc -l < "$OUT/filtered_1.fq") / 4 )); k2=$(( $(wc -l < "$OUT/filtered_2.fq") / 4 ))
echo "  extracted mate1=$k1 mate2=$k2"
[[ "$k1" == "$k2" && "$k1" -gt 0 ]] || { echo "FAIL: filtered mates do not pair" >&2; exit 1; }

# Persist gzipped for the AlbertEM alignment, which reads $WORK/trap/<ACC>/filtered_{1,2}.fq.gz
gzip -c "$OUT/filtered_1.fq" > "$OUT/filtered_1.fq.gz"
gzip -c "$OUT/filtered_2.fq" > "$OUT/filtered_2.fq.gz"

# --- 5. AlbertSalmon --------------------------------------------------------
echo "[rnq_trap] (5) salmon on the filtered reads -> $SALMON_OUT"
salmon quant -q -i "$IDX" -l "$SALMON_LIBTYPE" \
    -1 "$OUT/filtered_1.fq" -2 "$OUT/filtered_2.fq" \
    ${SALMON_FLAGS} -o "$SALMON_OUT" --threads "${SLURM_CPUS_PER_TASK}" \
    > "$SALMON_OUT/salmon.out" 2> "$SALMON_OUT/salmon.err" \
    || { echo "[rnq_trap] salmon FAILED"; tail -20 "$SALMON_OUT/salmon.err" >&2; exit 1; }
fmt=$(sed -n 's/.*"expected_format"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$SALMON_OUT/lib_format_counts.json" | head -1)
echo "lib_type requested=$SALMON_LIBTYPE observed=${fmt:-unknown}" | tee "$SALMON_OUT/lib_type.txt"

# The uncompressed copies are large and regenerable from the .gz.
rm -f "$OUT/filtered_1.fq" "$OUT/filtered_2.fq" "$R1" "$R2"
echo "[rnq_trap] done -> $SALMON_OUT/quant.sf"
