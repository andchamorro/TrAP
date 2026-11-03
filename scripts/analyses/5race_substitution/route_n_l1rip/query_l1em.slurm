#!/bin/bash
# Route N, QUERY side -- the `L1EM` / `AlbertEM` arm of Figure 11.
#
# ARM=plain  -> L1EM    (the baseline, on the depth-matched library's own BAM)
# ARM=filtered -> AlbertEM (the same code on the classifier-filtered BAM)
# The two arms share this script so the ONLY difference between them is which reads
# went in, which is exactly what the R2-2 answer claims.
#
# COST AND SIZING
# scripts/sh/run_l1em.sh records a 200k-pair probe at 101 s, extrapolating to ~4.6 h
# wall at 16 cores for ~33 M pairs; at the 41,368,288 pairs used here that is ~5.7 h,
# and the extrapolation is explicitly NOT an observation -- EM convergence need not
# stay linear the way bwa mem's read-proportional cost does.  query_probe.slurm
# re-measures the probe on THIS library first; read its reported extrapolation before
# submitting, and raise --time rather than discovering the cap at hour 24.
# run_L1EM.sh is run at L1EM_THREADS=8 (upstream hardcodes 16; patched to read the
# variable) because the project cap is 8 concurrent processes.  That roughly DOUBLES the
# wall clock against the 16-core extrapolation -- the repo's own note already puts a
# full-depth library at ~9 h on 8 cores -- while the SU cost is similar, since the charge
# is cores x elapsed.
#
# SIZED FROM FULL-LIBRARY MEASUREMENTS IN THE SISTER PROJECT, not from a probe.
#
# Sertorius ran L1EM against the SAME L1EM.400 reference on 12 libraries of
# 29.8-40.5 M proper pairs -- i.e. this library's scale -- at 16 cores:
#     L1EM step alone      1023-1797 s   (17-30 min)
#     whole per-sample job 1:00:34-1:32:59
# (/scratch/user/achamorro/Sertorius/logs/l1em400_full_*, jobs 20022827 / 20025057;
#  depths from each run's own baminfo.txt.)  At 8 cores that is ~35-60 min for the L1EM
# step, which is all this job does -- alignment is query_align.slurm.  So --time=04:00:00
# is 4-7x headroom, and the exposure if this hangs is ~32 SU rather than the ~384 SU a
# 48 h request would have allowed.
#
# WHY THE PROBE'S OWN EXTRAPOLATION WAS WRONG:
# query_probe.slurm measured 316 s for 200,000 pairs and extrapolated ~18 h.  That is off
# by ~20x, and linear extrapolation is the wrong model here: a large part of L1EM's cost
# is FIXED per split, not per read.  STEP 3 runs `bwa aln` once per split file per mate --
# 2 x $threads invocations -- and each one loads the 251.8 Mbp L1EM.400 bwa index from
# disk before doing any work.  At probe scale (~1.6k L1 read pairs per split) those index
# loads ARE the runtime; at full scale they are amortised over ~200x more reads.  A probe
# can confirm the stack runs and can measure throughput for a read-proportional step; it
# cannot size a step dominated by fixed overhead.  Prefer a real full-library measurement
# whenever one exists.
# Memory is held at 8 GB/core (Grace's per-core share) so the job stays core-bound in the
# charge: L1EM's peak is per-chunk, bounded by reads_per_pickle, not by library size, and
# the 200k probe peaked under 1 GB.
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_l1em
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rnq_l1em_%j.out
#SBATCH --error=logs/rnq_l1em_%j.err

set -euo pipefail

module load GCC/12.3.0 BWA/0.7.17 SAMtools/1.18

REPO=/scratch/user/achamorro/Python/TrAP
WORK=/scratch/user/achamorro/analysis/5race_route_n_query
GENOME="$REPO/data/external/GRCh38.p14.genome.fa"
# L1EM is a third-party checkout, not shipped with TrAP; override for another location.
L1EM_DIR="${L1EM_DIR:-$REPO/.trap/L1EM}"
# 8, not upstream's hardcoded 16: the project cap is 8 concurrent processes.
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

ACC="${ACC:-SRR14758878}"
ARM="${ARM:-plain}"
case "$ARM" in
    plain)    BAM="$WORK/align/${ACC}/sorted_reads.bam";          OUT="$WORK/l1em/${ACC}" ;;
    filtered) BAM="$WORK/align/${ACC}_filtered/sorted_reads.bam"; OUT="$WORK/mlem/${ACC}" ;;
    *) echo "ARM must be plain|filtered, got '$ARM'" >&2; exit 1 ;;
esac

# L1EM calls both `python` and `$(which python)`; a shim with ONLY a python symlink is
# prepended so the conda env cannot shadow the pinned bwa/samtools modules.
TRAP_PY=/scratch/user/achamorro/.conda/envs/trap/bin/python
SHIM="$WORK/pyshim"; mkdir -p "$SHIM"; ln -sf "$TRAP_PY" "$SHIM/python"
export PATH="$SHIM:$PATH"
echo "python -> $(command -v python) : $(python -c 'import sys,numpy,scipy,pysam; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pysam.__version__)')"

[[ -s "$BAM" ]] || { echo "MISSING $BAM (run query_align.slurm with READS=$ARM)" >&2; exit 1; }
[[ -s "$L1EM_DIR/annotation/L1EM.400.fa.bwt" ]] || { echo "MISSING L1EM bwa index" >&2; exit 1; }

if [[ -s "$OUT/full_counts.txt" ]]; then
    echo "[$(date)] [skip] $OUT/full_counts.txt exists"
else
    mkdir -p "$OUT"
    echo "[$(date)] $ACC ($ARM) : L1EM on $BAM"
    t0=$SECONDS
    ( cd "$OUT" && bash "$L1EM_DIR/$L1EM_DRIVER" "$BAM" "$L1EM_DIR" "$GENOME" ) \
        > "$OUT/L1EM.out" 2> "$OUT/L1EM.err" || {
            echo "L1EM FAILED -- tail of stderr:" >&2; tail -40 "$OUT/L1EM.err" >&2; exit 1; }
    echo "  L1EM wall: $(( (SECONDS - t0) / 60 )) min"
    # The intermediates are large and regenerable; the pickles are what further
    # analysis needs, and run_L1EM.sh already copied those up a level.
    rm -rf "$OUT/split_fqs" "$OUT/G_of_R" "$OUT/idL1reads"
fi

cat "$OUT/baminfo.txt"
python -I "$REPO/scripts/analyses/5race_substitution/route_n_l1rip/l1hs_l1pa_fpm.py" "$OUT" "$OUT/L1HS-L1PA_FPM.txt"
echo "[$(date)] done -> $OUT/L1HS-L1PA_FPM.txt"
