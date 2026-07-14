#!/bin/bash
#SBATCH --job-name=trap_22_build_seqlabel
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Track B2 — SLURM wrapper for the sequence-anchored build (compute node: heavy
# ART + STAR). Mirrors 20_dataset.slurm step (a) but runs build_dataset_seqlabel.sh
# (strict .out overlap labelling + minimap2 cross-check) instead of build_dataset.sh.
#
# Outputs the *seqlabel.5x FASTQ (additive — never touches the option-A FASTQ).
# After this, run preprocessing with the EXISTING 20_dataset.slurm reusing these
# FASTQ (SKIP_BUILD=1) into a NEW *.seqlabel processing name — see B2-cont below.
#
#   sbatch scripts/slurm/22_build_seqlabel.slurm                       # full build (~few h)
# CREDITS: SUs = cores × actual_runtime, so a fast dry-run is cheap — request fewer
# cores + time for it (and the chrom check now fails in seconds on a misconfig):
#   POS_COV=1 NEG_COV=1 sbatch --time=02:00:00 --cpus-per-task=16 --mem=64G \
#     scripts/slurm/22_build_seqlabel.slurm
#
# Knobs: POS_COV / NEG_COV (per-stream ART coverage), STRICT_FRAC (default 0.5),
# MAX_DIV (default 100; e.g. 10 for young-only L1), CROSSCHECK_LIB (B0 library).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules
activate_trap
# minimap2 for the cross-check phase (optional — the build skips it if absent).
module load minimap2/2.29 2>/dev/null || module load GCCcore/13.2.0 minimap2/2.29 2>/dev/null \
    || echo "[build_seqlabel] minimap2 module not loaded — cross-check will skip"

bash "${REPO_ROOT}/scripts/build_dataset_seqlabel.sh"

echo "[build_seqlabel] done. B2-cont (preprocess) — reuse 20_dataset.slurm, e.g.:"
echo "  SKIP_BUILD=1 SKIP_MASKING=1 \\"
echo "    L1_R1=data/external/l1hs_l1pa2_negative.seqlabel.5x_R1.fq \\"
echo "    L1_R2=data/external/l1hs_l1pa2_negative.seqlabel.5x_R2.fq \\"
echo "    PROCESSING_NAME=gencode.v48.k17.salmon.seqlabel/l1hs_l1pa2 \\"
echo "    RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/20_dataset.slurm"
