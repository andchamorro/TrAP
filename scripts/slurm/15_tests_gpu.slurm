#!/bin/bash
#SBATCH --job-name=trap_15_tests_gpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 15 gpu — slow tests: training loops, HPO, and inference (GPU required).
# Covers:
#   TestClassificationTrainingLoop  — multi-epoch ALBERT classifier
#   TestMaskingTrainingLoop         — multi-epoch ALBERT MLM
#   TestHyperparameterSearch        — 2-trial Optuna search (train_workflow)
#   TestEndToEndSearch              — 2-trial Optuna search (tune)
#   TestSalmonTokenizerTokenization — 1000-read tokenisation pipeline
#   test_dataloader_inference_*     — DataLoader → model → Parquet
#
# TrainingArguments auto-detects CUDA and enables bf16/fp16 (same path as
# production).  On GPU the full suite finishes in < 15 min.
#
# --gres and --partition are injected by submit_tests.sh from grace.yaml.
# To submit directly:
#   sbatch --gres=gpu:a100:1 --partition=gpu scripts/slurm/15_tests_gpu.slurm
#
# Pair: 15_tests_cpu.slurm runs unit + integration (not slow) on CPU.
# Orchestrator: bash scripts/slurm/submit_tests.sh  (submits both in parallel)

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"

load_gpu_modules
activate_trap

cd "${REPO_ROOT}"

TEST_PATH="${TEST_PATH:-tests}"
TEST_MARK="${TEST_MARK:-slow}"
JUNIT="logs/pytest_gpu_${SLURM_JOB_ID:-local}.xml"

mark_args=()
[[ -n "${TEST_MARK}" ]] && mark_args+=(-m "${TEST_MARK}")

cov_args=()
if python -c "import pytest_cov" 2>/dev/null; then
    cov_args+=(--cov=trap --cov-report=xml:"logs/coverage_gpu_${SLURM_JOB_ID:-local}.xml")
fi

echo "[15_tests_gpu] pytest ${TEST_PATH} -m '${TEST_MARK}' (size=${TRAP_TEST_DATASET_SIZE:-tiny})"
python -m pytest "${TEST_PATH}" "${mark_args[@]}" \
    -ra -v \
    --junitxml="${JUNIT}" \
    "${cov_args[@]}"

echo "[15_tests_gpu] done -> ${JUNIT}"
