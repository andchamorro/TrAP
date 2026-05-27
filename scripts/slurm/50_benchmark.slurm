#!/bin/bash
#SBATCH --job-name=trap_50_benchmark
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# --gres and --partition are injected by submit_pipeline.sh from config/hpc/grace.yaml
# To submit directly: sbatch --gres=gpu:a100:2 --partition=gpu scripts/slurm/50_benchmark.slurm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 5 — streaming quantify performance benchmark + manifest (plan §6.6).
# This is the user-priority gate before the full experimental re-run (§8 floors).
# Point BENCH_R1/BENCH_R2 at a release-size FASTQ; defaults to the tiny CI fixture.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

BENCH_R1="${BENCH_R1:-${REPO_ROOT}/tests/fixtures/tiny_pair_R1.fq.gz}"
BENCH_R2="${BENCH_R2:-${REPO_ROOT}/tests/fixtures/tiny_pair_R2.fq.gz}"
BENCH_OUT="${BENCH_OUT:-${REPO_ROOT}/reports/bench/v48.k17}"

# quantify.py is a single-command Typer app (invoked without a sub-command name).
accelerate launch $(accelerate_args) "${REPO_ROOT}/trap/modeling/quantify.py" \
    --pretrained-model-name "${MODEL_CLS}" \
    --r1 "${BENCH_R1}" \
    --r2 "${BENCH_R2}" \
    --output-path "${BENCH_OUT}" \
    --k "${K}" \
    --batch-size 64 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --seed "${SEED}"

echo "[50_benchmark] scores + manifest -> ${BENCH_OUT}"
