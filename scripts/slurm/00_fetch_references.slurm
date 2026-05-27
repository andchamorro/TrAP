#!/bin/bash
#SBATCH --job-name=trap_00_fetch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 0 — fetch GENCODE v48 transcripts + GRCh38.p14 and build STAR/BWA
# indexes (plan §6.1 / §9.1). Run from the repo root: `sbatch scripts/slurm/00_fetch_references.slurm`.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules
module load ${BWA_MODULE}
command -v bwa &>/dev/null || { echo "[00_fetch] ERROR: bwa not in PATH after loading ${BWA_MODULE}" >&2; exit 1; }
activate_trap

bash "${REPO_ROOT}/scripts/data/fetch_references.sh"
