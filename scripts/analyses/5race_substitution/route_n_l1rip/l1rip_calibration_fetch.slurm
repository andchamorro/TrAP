#!/bin/bash
# Route N (L1-ORF1p RIP nanopore calibration arm) -- fetch the two GridION runs.
#
# PRJNA657989 (Whongsiri et al., Int J Mol Sci 2020;21:9433).  VM-Cub-1 urothelial
# carcinoma, RNA immunoprecipitated
# with an L1-ORF1p antibody, then nanopore full-length RNA-seq:
#   SRR12474736  Library_8VmCub1ORF1    6,312,840 reads  ~739 nt mean
#   SRR12474735  Library_10VmCub1ORF1   6,974,505 reads  ~793 nt mean
# The two libraries are biological replicates of the same pull-down, so they give
# a replicate-concordance check this route would otherwise lack.
#
# WHY THIS ROUTE EXISTS
# It is the one candidate that is L1-TARGETED.  The Iso-Seq arm failed because its
# L1 reads were passengers inside abundant host transcripts; an ORF1p pull-down
# removes that failure mode by construction rather than by filtering.  The cost is
# that it is not 5'-anchored -- ORF1p binds its own transcript anywhere along it --
# so "full-length, 5'-complete" has to be MEASURED per read, which is what
# l1rip_fulllength.py does.  ENA metadata says only `RNA-Seq / RANDOM`, which is
# why no library_selection query in the search of record could ever have found it.
#
# SRA Toolkit has no Lmod module on Grace; use the local 3.4.1 install.
# WebProxy is required for outbound HTTPS from a compute node.
# The charge account is NOT inlined here; it is read from the git-ignored
# config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_n_l1rip/submit_l1rip_calibration.sh.
#SBATCH --job-name=l1rip_fetch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/l1rip_fetch_%A_%a.out
#SBATCH --error=logs/l1rip_fetch_%A_%a.err

set -euo pipefail

module load WebProxy
export PATH=/scratch/user/achamorro/.local/sratoolkit.3.4.1-ubuntu64/bin:$PATH

SRA_DIR=/scratch/user/achamorro/data/SRA/PRJNA657989
mkdir -p "$SRA_DIR"/{sra_cache,fastqs,tmp}
cd "$SRA_DIR"

ACCS=(SRR12474736 SRR12474735)       # Library_8, Library_10
EXPECTED=(6312840 6974505)           # ENA read_count, checked below
ACC=${ACCS[$SLURM_ARRAY_TASK_ID]}
WANT=${EXPECTED[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] $ACC : prefetch -> $SRA_DIR/sra_cache"
prefetch -p -f no --max-size 50G -O sra_cache "$ACC"

echo "[$(date)] $ACC : vdb-validate"
vdb-validate "sra_cache/$ACC" 2>&1 | sed "s/^/  /"

echo "[$(date)] $ACC : fasterq-dump (single-end, ONT)"
fasterq-dump "sra_cache/$ACC" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --concatenate-reads \
    --include-technical \
    -O fastqs -t tmp

gzip -f "fastqs/${ACC}.fastq"

# Integrity, not just presence.  `head` on a pipe would SIGPIPE `zcat` and, under
# `pipefail`, fail the job after all the real work had succeeded.
gzip -t "fastqs/${ACC}.fastq.gz"
READS=$(zcat "fastqs/${ACC}.fastq.gz" | awk 'END{print NR/4}')
echo "$ACC reads=$READS expected=$WANT"
[[ "$READS" == "$WANT" ]] || { echo "ERROR: read count != ENA" >&2; exit 1; }

# Protocol check BEFORE spending compute.  A full-length L1 mRNA is ~6 kb; the
# deposit's ~790 nt mean says most reads CANNOT span one, so the route's claim is
# about which loci are transcribed, not about recovering intact transcripts.
# Recording the tail is what makes that statement quantitative rather than assumed.
echo "[$(date)] $ACC : read-length distribution"
zcat "fastqs/${ACC}.fastq.gz" \
    | awk 'NR%4==2 {n++; L+=length($0)
                    if(length($0)>=6000) ge6++; else if(length($0)>=2000) ge2++;
                    else if(length($0)>=500) ge05++; else lt05++}
           END {printf "  reads=%d mean=%.0f  <500nt=%.1f%%  500-2000=%.1f%%  2000-6000=%.1f%%  >=6000nt=%.2f%% (%d reads)\n",
                       n, L/n, 100*lt05/n, 100*ge05/n, 100*ge2/n, 100*ge6/n, ge6}'

echo "[$(date)] $ACC : done"
ls -la "fastqs/${ACC}.fastq.gz"
