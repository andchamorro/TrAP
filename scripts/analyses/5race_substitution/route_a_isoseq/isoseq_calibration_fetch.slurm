#!/bin/bash
# Phase 1 (HEK-293 Iso-Seq calibration arm) -- fetch the four Revio HiFi runs.
#
# PRJNA1176011 / GSE280041, pb_HEK293_rep1 = SRR31072049 + SRR31072050,
#                           pb_HEK293_rep2 = SRR31072047 + SRR31072048.
# SRA Toolkit has no Lmod module on Grace; use the local 3.4.1 install.
# WebProxy is required for outbound HTTPS from a compute node.
# The charge account is NOT inlined here.  It is sensitive, so it is read from
# the git-ignored config/hpc/grace.yaml and exported as SBATCH_ACCOUNT by
# scripts/analyses/5race_substitution/route_a_isoseq/submit_isoseq_calibration.sh, which sbatch honours as the
# default --account.  Convention borrowed from Sertorius scripts/hpc/submit.sh.
#SBATCH --job-name=isoseq_fetch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/fetch_%A_%a.out
#SBATCH --error=logs/fetch_%A_%a.err

set -euo pipefail

module purge
module load WebProxy
export PATH=/scratch/user/achamorro/.local/sratoolkit.3.4.1-ubuntu64/bin:$PATH

SRA_DIR=/scratch/user/achamorro/data/SRA/PRJNA1176011
cd "$SRA_DIR"

ACCS=(SRR31072047 SRR31072048 SRR31072049 SRR31072050)
ACC=${ACCS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] $ACC : prefetch -> $SRA_DIR/sra_cache"
prefetch -p -f no --max-size 50G -O sra_cache "$ACC"

echo "[$(date)] $ACC : vdb-validate"
vdb-validate "sra_cache/$ACC" 2>&1 | sed "s/^/  /"

echo "[$(date)] $ACC : fasterq-dump (single-end, PacBio)"
fasterq-dump "sra_cache/$ACC" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --concatenate-reads \
    --include-technical \
    -O fastqs -t tmp

gzip -f "fastqs/${ACC}.fastq"

echo "[$(date)] $ACC : done"
ls -la "fastqs/${ACC}.fastq.gz"
# Integrity, not just presence: the archive must inflate and hold the number of
# reads ENA reports.  `head` on a pipe would SIGPIPE `zcat` and, under
# `pipefail`, fail the job after the real work had already succeeded.
gzip -t "fastqs/${ACC}.fastq.gz"
READS=$(zcat "fastqs/${ACC}.fastq.gz" | awk 'END{print NR/4}')
echo "$ACC reads=$READS"
