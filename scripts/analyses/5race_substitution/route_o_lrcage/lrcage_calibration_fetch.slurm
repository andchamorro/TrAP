#!/bin/bash
# Route O (long-read CAGE calibration arm) -- fetch the three Sequel II runs.
#
# PRJNA936447 (PMC10760525), NCI-H1299.  Three long-read libraries, and the
# three are the whole point of this route:
#   SRR23517790  LRCAGE  DMSO   cap-trapped, vehicle        3,479,899 reads  ~1351 nt
#   SRR23517787  LRCAGE  DACSB  cap-trapped, decitabine+SB939  3,314,336  ~1375 nt
#   SRR23517789  LRhex   DMSO   RANDOM-PRIMED, vehicle      4,253,780 reads  ~1013 nt
# LRhex is the internal negative control: same cells, same platform, same read
# length, no cap selection.  It is what makes this route self-calibrating --
# unlike the Iso-Seq arm, there is no 5'RACE library in NCI-H1299 to compare
# against, so the decisive contrast has to come from inside the deposit.
# DACSB is the epigenetic-activation arm: rather than find a line with more active
# L1 loci, the same line is driven into that state, which keeps the contrast internal.
#
# The matched SHORT-read nanoCAGE partners are deliberately NOT fetched here --
# they are the query side (TrAP/Salmon/L1EM input), not the reference arm:
#   SRR23517788 nanoCAGE DMSO (33,516,249 reads), SRR23517786 nanoCAGE DACSB (51,861,480).
#
# SRA Toolkit has no Lmod module on Grace; use the local 3.4.1 install.
# WebProxy is required for outbound HTTPS from a compute node.
# The charge account is NOT inlined here.  It is sensitive, so it is read from
# the git-ignored config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_o_lrcage/submit_lrcage_calibration.sh.
#SBATCH --job-name=lrcage_fetch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/lrcage_fetch_%A_%a.out
#SBATCH --error=logs/lrcage_fetch_%A_%a.err

set -euo pipefail

module load WebProxy
export PATH=/scratch/user/achamorro/.local/sratoolkit.3.4.1-ubuntu64/bin:$PATH

SRA_DIR=/scratch/user/achamorro/data/SRA/PRJNA936447
mkdir -p "$SRA_DIR"/{sra_cache,fastqs,tmp}
cd "$SRA_DIR"

ACCS=(SRR23517790 SRR23517787 SRR23517789)      # LRCAGE DMSO, LRCAGE DACSB, LRhex DMSO
EXPECTED=(3479899 3314336 4253780)              # ENA read_count, checked below
ACC=${ACCS[$SLURM_ARRAY_TASK_ID]}
WANT=${EXPECTED[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] $ACC : prefetch -> $SRA_DIR/sra_cache"
prefetch -p -f no --max-size 50G -O sra_cache "$ACC"

echo "[$(date)] $ACC : vdb-validate"
vdb-validate "sra_cache/$ACC" 2>&1 | sed "s/^/  /"

echo "[$(date)] $ACC : fasterq-dump (single-end, PacBio CCS)"
fasterq-dump "sra_cache/$ACC" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --concatenate-reads \
    --include-technical \
    -O fastqs -t tmp

gzip -f "fastqs/${ACC}.fastq"

# Integrity, not just presence.  `head` on a pipe would SIGPIPE `zcat` and,
# under `pipefail`, fail the job after all the real work had succeeded -- that
# bug cost a dependency chain on the Iso-Seq arm.
gzip -t "fastqs/${ACC}.fastq.gz"
READS=$(zcat "fastqs/${ACC}.fastq.gz" | awk 'END{print NR/4}')
echo "$ACC reads=$READS expected=$WANT"
[[ "$READS" == "$WANT" ]] || { echo "ERROR: read count != ENA" >&2; exit 1; }

# Protocol check BEFORE any compute is spent: the Iso-Seq arm was rejected partly
# because its "full-length" reads turned out to be raw HiFi, not FLNC.  A
# cap-trapped LRCAGE library should be long and tightly distributed around ~1.3 kb;
# a bimodal or very short distribution means the deposit is not what GEO claims.
echo "[$(date)] $ACC : read-length distribution"
zcat "fastqs/${ACC}.fastq.gz" \
    | awk 'NR%4==2 {n++; L+=length($0); if(length($0)<500)s++; else if(length($0)<2000)m++; else l++}
           END {printf "  reads=%d mean=%.0f  <500nt=%.1f%%  500-2000nt=%.1f%%  >=2000nt=%.1f%%\n",
                       n, L/n, 100*s/n, 100*m/n, 100*l/n}'

echo "[$(date)] $ACC : done"
ls -la "fastqs/${ACC}.fastq.gz"
