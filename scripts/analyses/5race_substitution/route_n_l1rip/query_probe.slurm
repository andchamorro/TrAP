#!/bin/bash
# Route N, QUERY side -- the cheap probe that must pass before the expensive runs.
#
# Two things are unknown and both are cheap to settle, so neither is assumed:
#
# (A) DOES L1EM EVEN RUN HERE?  The repo's L1EM copy is upstream's, patched for
#     py2/py3 pickling; the env it will actually run in is python 3.12.2 with numpy
#     2.4.6, scipy 1.17.1 and pysam 0.23.3, against an L1EM.yml that pins
#     python 2.7.15 / numpy 1.14.3 / pysam 0.15.0.  Every module compiles and no
#     removed numpy alias is used, but "compiles" is not "produces the right
#     numbers", so the whole chain is run on 200k pairs first.  That size is not
#     arbitrary: scripts/sh/run_l1em.sh records a 200k-pair probe at 101 s, which is
#     the figure the ~4.6 h-at-16-cores extrapolation for a full library rests on,
#     so this probe also re-measures that extrapolation on THIS library before
#     ~100 SU are committed to it.
#
# (B) WHAT LIBRARY TYPE IS THIS?  scripts/salmon_pipeline.sh pins -l ISR and says
#     in as many words that the value must be re-measured per library and then
#     pinned, because `-l A` re-infers per quantification and AlbertSalmon vs Salmon
#     differ only in which reads survive the filter -- a type that resolved
#     differently between those two arms would appear as a method effect.  So
#     salmon runs with `-l A` on 2 M pairs of each replicate here, ONCE, and the
#     observed type is what gets pinned for the real runs.
#
# `head`-style subsetting is deliberate for a smoke test (fastest, deterministic);
# the real runs use the seeded depth-matched files, never these.
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rnq_probe_%j.out
#SBATCH --error=logs/rnq_probe_%j.err

set -euo pipefail

module load GCC/12.3.0 Salmon/1.10.3 BWA/0.7.17 SAMtools/1.18 BEDTools/2.31.0 SeqKit/2.9.0

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_route_n_query
GENOME="$REPO/data/external/GRCh38.p14.genome.fa"
# L1EM is a third-party checkout, not shipped with TrAP; override for another location.
L1EM_DIR="${L1EM_DIR:-$REPO/.trap/L1EM}"
MATCHED="$WORK/fastq_matched"
PROBE="$WORK/probe"
THREADS="${SLURM_CPUS_PER_TASK}"
# The project cap is 8 concurrent processes.  run_L1EM.sh hardcoded threads=16 upstream
# and has been patched to read L1EM_THREADS, so the cap is honoured without forking it.
# $threads there is also the number of pieces the L1 fastq is split into, so it sets the
# fan-out of its parallel sampe pipelines and G_of_R workers -- one knob, all of them.
export L1EM_THREADS="${SLURM_CPUS_PER_TASK}"

# WHICH L1EM DRIVER: the library is UNSTRANDED, measured not assumed.
# salmon -l A on 2 M pairs of each replicate returned IU, and the fragment counts behind
# that verdict are the evidence rather than the verdict itself: ISF vs ISR came out
# 873/868, 931/1047 and 811/890 with strand_mapping_bias 0.471-0.501, i.e. the 49.65-50.35 %
# band the sibling project uses as its own unstranded criterion.  The published HEK293
# query library is ISR (SMARTer Stranded); this is a different kit and a real asymmetry.
# L1EM therefore has to run through run_L1EM_UNSTRANDED.sh, which is not a flag but a
# DIFFERENT MODEL (G_of_R_unstranded.py) and a different output schema: full_counts.txt
# carries only/3prunon/passive instead of the five stranded columns, because sense and
# antisense cannot be separated without strand information.  Running the stranded driver
# on this library would have invented a sense/antisense split out of the EM's
# initialisation -- the exact failure the sibling project's ADR-0005 D-1 refuses to ship.
L1EM_STRAND="${L1EM_STRAND:-unstranded}"
case "$L1EM_STRAND" in
    unstranded) L1EM_DRIVER=run_L1EM_unstranded.sh ;;
    stranded)   L1EM_DRIVER=run_L1EM.sh ;;
    *) echo "L1EM_STRAND must be unstranded|stranded, got '$L1EM_STRAND'" >&2; exit 1 ;;
esac
echo "L1EM driver: $L1EM_DRIVER (L1EM_STRAND=$L1EM_STRAND, L1EM_THREADS=$L1EM_THREADS)"

PRIMARY=SRR14758878
ACCS=(SRR14758878 SRR14758879 SRR14758880)
PROBE_PAIRS=200000
LIBTYPE_PAIRS=2000000

# L1EM calls `python` and `$(which python)`; both must be the trap interpreter.
# A shim with ONLY a python symlink is prepended, rather than activating the conda
# env, so the env's own bwa/samtools (if any) cannot shadow the pinned modules --
# the same reason the blocker-check job called the interpreter by absolute path.
TRAP_PY=/scratch/user/achamorro/.conda/envs/trap/bin/python
SHIM="$WORK/pyshim"
mkdir -p "$SHIM"
ln -sf "$TRAP_PY" "$SHIM/python"
export PATH="$SHIM:$PATH"
echo "python -> $(command -v python) : $(python -c 'import sys,numpy,scipy,pysam; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pysam.__version__)')"

mkdir -p "$PROBE"
[[ -s "$L1EM_DIR/annotation/L1EM.400.fa.bwt" ]] || { echo "MISSING L1EM bwa index (run query_build_refs)" >&2; exit 1; }
[[ -s "$WORK/salmon_ref/only.Index/info.json" ]] || { echo "MISSING salmon index (run query_build_refs)" >&2; exit 1; }
for ACC in "${ACCS[@]}"; do
    [[ -s "$MATCHED/${ACC}_matched_1.fastq.gz" ]] || { echo "MISSING $ACC matched reads" >&2; exit 1; }
done

# ----------------------------------------------------------- (A) L1EM probe
PDIR="$PROBE/l1em_${L1EM_STRAND}_${PRIMARY}_${PROBE_PAIRS}"
if [[ -s "$PDIR/full_counts.txt" ]]; then
    echo "[$(date)] [skip] L1EM probe already done"
else
    mkdir -p "$PDIR"
    echo "[$(date)] probe reads: first ${PROBE_PAIRS} pairs of $PRIMARY"
    for mate in 1 2; do
        seqkit head -n "$PROBE_PAIRS" "$MATCHED/${PRIMARY}_matched_${mate}.fastq.gz" \
            -o "$PROBE/probe_${mate}.fastq.gz"
    done

    echo "[$(date)] bwa mem -> genome BAM (the input L1EM re-extracts from)"
    t0=$SECONDS
    bwa mem -t "$THREADS" "$GENOME" \
        "$PROBE/probe_1.fastq.gz" "$PROBE/probe_2.fastq.gz" 2> "$PROBE/bwa.probe.err" \
        | samtools sort -@ 4 -m 2G -T "$PROBE/tmp.probe" -o "$PROBE/probe.bam" -
    samtools index -@ "$THREADS" "$PROBE/probe.bam"
    echo "  bwa+sort: $((SECONDS - t0)) s"
    samtools flagstat -@ 4 "$PROBE/probe.bam" | head -8

    echo "[$(date)] L1EM on the probe BAM (L1EM_THREADS=${L1EM_THREADS})"
    t0=$SECONDS
    ( cd "$PDIR" && bash "$L1EM_DIR/$L1EM_DRIVER" "$PROBE/probe.bam" "$L1EM_DIR" "$GENOME" ) \
        > "$PDIR/L1EM.out" 2> "$PDIR/L1EM.err" || {
            echo "L1EM FAILED -- tail of stderr:" >&2; tail -40 "$PDIR/L1EM.err" >&2; exit 1; }
    echo "  L1EM: $((SECONDS - t0)) s for ${PROBE_PAIRS} pairs"
    echo "  extrapolated to 41,368,288 pairs: $(( (SECONDS - t0) * 41368288 / PROBE_PAIRS / 3600 )) h"
    echo "  (EM convergence need not scale linearly; this is an extrapolation)"

    wc -l "$PDIR/full_counts.txt"
    cat "$PDIR/baminfo.txt"
    python -I "$REPO/scripts/analyses/5race_substitution/route_n_l1rip/l1hs_l1pa_fpm.py" "$PDIR" "$PDIR/L1HS-L1PA_FPM.txt"
    head -4 "$PDIR/L1HS-L1PA_FPM.txt"
fi

# ----------------------------------------------- (B) library type, all three
echo "[$(date)] salmon -l A on ${LIBTYPE_PAIRS} pairs per replicate (library type + first signal)"
: > "$PROBE/lib_types.tsv"
for ACC in "${ACCS[@]}"; do
    out="$PROBE/salmon_libtype_${ACC}"
    if [[ ! -s "$out/quant.sf" ]]; then
        for mate in 1 2; do
            seqkit head -n "$LIBTYPE_PAIRS" "$MATCHED/${ACC}_matched_${mate}.fastq.gz" \
                -o "$PROBE/lt_${ACC}_${mate}.fastq.gz"
        done
        salmon quant -q -i "$WORK/salmon_ref/only.Index" -l A \
            -1 "$PROBE/lt_${ACC}_1.fastq.gz" -2 "$PROBE/lt_${ACC}_2.fastq.gz" \
            --validateMappings -o "$out" --threads "$THREADS" \
            > "$out.log" 2>&1 || { echo "salmon FAILED for $ACC" >&2; tail -20 "$out.log" >&2; exit 1; }
        rm -f "$PROBE/lt_${ACC}_1.fastq.gz" "$PROBE/lt_${ACC}_2.fastq.gz"
    fi
    fmt=$(sed -n 's/.*"expected_format"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$out/lib_format_counts.json" | head -1)
    mapped=$(sed -n 's/.*"percent_mapped"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' \
        "$out/aux_info/meta_info.json" | head -1)
    nonzero=$(awk -F'\t' 'NR>1 && $5>0' "$out/quant.sf" | wc -l)
    printf '%s\t%s\t%s\t%s\n' "$ACC" "${fmt:-unknown}" "${mapped:-NA}" "$nonzero" \
        | tee -a "$PROBE/lib_types.tsv"
done
echo "accession  observed_libtype  percent_mapped_to_L1  targets_with_reads"
column -t "$PROBE/lib_types.tsv"

echo "[$(date)] done"
