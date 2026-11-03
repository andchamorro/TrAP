#!/bin/bash
# Route N, QUERY side -- convert the prefetched archives to FASTQ and depth-match them.
#
# This job does NO networking.  The transfer happened on the login node with
# `prefetch` (query_prefetch.sh); what is left is CPU work, and the login node is
# cgroup-limited to one cpu, so it belongs here.  The split is the point:
#   network  -> login node, SRA Tools, zero SU
#   cpu      -> this job
# (`nproc` prints 1 both here and on the login node, because Grace exports
#  OMP_NUM_THREADS=1 and GNU nproc honours it.  That is NOT a cpu limit: `nproc --all`
#  is 48 and scontrol confirms this job holds NumCPUs=8.  Never size off `nproc` on this
#  cluster, and note that any OpenMP-threaded tool runs single-threaded unless told
#  otherwise -- which is why every thread count below is passed explicitly.)
#
# THREE PROCESS-COUNT RULES ARE RESPECTED, deliberately:
#   * the project cap is 8 concurrent processes, so this is ONE job over the three
#     accessions in sequence -- not an array of three, which at 6 dump threads each
#     would have run 18;
#   * fasterq-dump gets 6 threads;
#   * the depth-match runs one mate at a time, `seqkit -j 2 | pigz -p 5` = 7.
# Sequential accessions also bound peak disk at ~37 GB instead of ~111 GB.
#
# WHY vdb-validate RUNS HERE AND NOT AT DOWNLOAD TIME
# `prefetch --verify yes` already checks the archive as it lands.  `vdb-validate` is the
# deeper, per-column checksum pass over the whole archive, and it is CPU-bound, so it
# would have crawled on a one-cpu login node.  It is the specialised equivalent of the
# md5 check that caught the corrupt wget resumes -- and it is strictly better, because
# it validates the archive's internal structure rather than just a whole-file digest.
#
# The charge account is NOT inlined here: it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=rnq_dump
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rnq_dump_%j.out
#SBATCH --error=logs/rnq_dump_%j.err

set -euo pipefail

module load GCC/13.3.0 SeqKit/2.9.0 pigz/2.8

SRA_BIN=/scratch/user/achamorro/.local/sratoolkit.3.4.1-ubuntu64/bin
export PATH="$SRA_BIN:$PATH"

WORK=/scratch/user/achamorro/analysis/5race_route_n_query
CACHE="$WORK/sra_cache"
DUMP="$WORK/fastq_dump"            # untrusted: derived from downloaded archives
MATCHED="$WORK/fastq_matched"
mkdir -p "$DUMP" "$MATCHED"

# HEK293 published query library depth -- the subsample target.  Depth is matched
# because per-locus detection is depth-dependent, so an unmatched comparison
# would confound protocol with sequencing effort.
TARGET_PAIRS=41368288
SEED=3469
DUMP_THREADS=6

ACCS=(SRR14758878 SRR14758879 SRR14758880)
# ENA read_count, checked against what fasterq-dump actually writes.
PAIRS=(43490739   51040895     54483025)

echo "[$(date)] fasterq-dump $(fasterq-dump --version 2>&1 | tr -d '\n'); $(nproc) cpus visible"

for i in "${!ACCS[@]}"; do
    ACC=${ACCS[$i]}
    WANT=${PAIRS[$i]}
    SRA="$CACHE/$ACC/$ACC.sra"
    R1="$DUMP/${ACC}_1.fastq"
    R2="$DUMP/${ACC}_2.fastq"
    M1="$MATCHED/${ACC}_matched_1.fastq.gz"
    M2="$MATCHED/${ACC}_matched_2.fastq.gz"

    if [[ -s "$M1" && -s "$M2" && -s "$MATCHED/${ACC}.depth_match.tsv" ]]; then
        echo "[$(date)] [skip] $ACC already depth-matched"; continue
    fi
    [[ -s "$SRA" ]] || { echo "MISSING $SRA -- run query_prefetch.sh on the login node" >&2; exit 1; }

    # ------------------------------------------------------------ validate
    echo "[$(date)] $ACC : vdb-validate"
    vdb-validate "$CACHE/$ACC" 2>&1 | sed 's/^/    /'

    # ------------------------------------------------------------ dump
    if [[ -s "$R1" && -s "$R2" ]]; then
        echo "[$(date)] $ACC : [skip] FASTQ already dumped"
    else
        echo "[$(date)] $ACC : fasterq-dump --split-files --threads $DUMP_THREADS"
        t0=$SECONDS
        fasterq-dump "$SRA" --split-files --threads "$DUMP_THREADS" \
            --outdir "$DUMP" --temp "$DUMP/tmp.$ACC" --force
        rm -rf "$DUMP/tmp.$ACC"
        echo "    dump: $((SECONDS - t0)) s"
    fi
    for f in "$R1" "$R2"; do
        [[ -s "$f" ]] || { echo "fasterq-dump produced no $f" >&2; exit 1; }
    done

    # ------------------------------------------------------------ count check
    # Line counting is exact and cheap next to the dump itself, and it is the check
    # that a truncated archive or a half-written dump would fail.
    n1=$(( $(wc -l < "$R1") / 4 ))
    n2=$(( $(wc -l < "$R2") / 4 ))
    echo "[$(date)] $ACC : dumped mate1=$n1 mate2=$n2 (ENA read_count=$WANT)"
    [[ "$n1" == "$n2" ]] || { echo "FAIL: $ACC mate counts differ -- pairing broken" >&2; exit 1; }
    [[ "$n1" == "$WANT" ]] || {
        echo "FAIL: $ACC dumped $n1 spots, ENA declares $WANT." >&2
        echo "      A mismatch here means the archive is not the run the manifest describes;" >&2
        echo "      do not depth-match it." >&2
        exit 1; }

    # ------------------------------------------------------------ depth-match
    FRAC=$(awk -v t="$TARGET_PAIRS" -v n="$n1" 'BEGIN{ if (t>=n) print 1; else printf "%.8f", t/n }')
    echo "[$(date)] $ACC : depth-match $n1 -> $TARGET_PAIRS pairs (p=$FRAC, seed=$SEED)"
    # One mate at a time: `seqkit -j 2 | pigz -p 5` is 7 processes, under the cap of 8.
    # Identical seed and proportion on both mates select the same records, because the
    # two files hold them in the same order -- verified below rather than assumed.
    for mate in 1 2; do
        src="$DUMP/${ACC}_${mate}.fastq"
        out="$MATCHED/${ACC}_matched_${mate}.fastq.gz"
        [[ -s "$out" ]] && { echo "    [skip] $(basename "$out")"; continue; }
        if [[ "$FRAC" == "1" ]]; then
            pigz -p 5 -c "$src" > "$out.part"
        else
            seqkit sample -j 2 -s "$SEED" -p "$FRAC" "$src" | pigz -p 5 -c > "$out.part"
        fi
        mv "$out.part" "$out"
    done

    # ------------------------------------------------------------ pairing check
    k1=$(seqkit stats -T "$M1" | awk 'NR==2{print $4}')
    k2=$(seqkit stats -T "$M2" | awk 'NR==2{print $4}')
    h1=$(seqkit head -n 1 "$M1" | head -1 | cut -d' ' -f1)
    h2=$(seqkit head -n 1 "$M2" | head -1 | cut -d' ' -f1)
    echo "[$(date)] $ACC : matched mate1=$k1 mate2=$k2 target=$TARGET_PAIRS; first id $h1 / $h2"
    [[ "$k1" == "$k2" ]] || { echo "FAIL: matched mate counts differ" >&2; exit 1; }
    [[ "$h1" == "$h2" ]] || { echo "FAIL: matched mate order diverged" >&2; exit 1; }
    printf 'accession\tarchive_pairs\tmatched_pairs\tproportion\tseed\n'  > "$MATCHED/${ACC}.depth_match.tsv"
    printf '%s\t%s\t%s\t%s\t%s\n' "$ACC" "$n1" "$k1" "$FRAC" "$SEED" >> "$MATCHED/${ACC}.depth_match.tsv"

    # The uncompressed dump is ~30 GB and fully regenerable from the verified archive.
    rm -f "$R1" "$R2"
    echo "[$(date)] $ACC done"
done

echo
echo "=== depth-match summary ==="
cat "$MATCHED"/*.depth_match.tsv | awk 'NR==1 || !/^accession/'
echo "[$(date)] all accessions done"
