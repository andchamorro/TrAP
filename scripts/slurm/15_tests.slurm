#!/bin/bash
#SBATCH --job-name=trap_15_tests
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Single-node full test suite (GPU required).  Runs all tests — unit,
# integration, and slow (training loops + HPO + inference) — in one job.
#
# Preferred workflow: use submit_tests.sh to run the CPU and GPU suites in
# parallel (faster wall-clock, better resource fit):
#   bash scripts/slurm/submit_tests.sh --run-config config/runs/salmon.yaml
#
# Use this script when you want a single job id to wait on, e.g.:
#   sbatch --gres=gpu:a100:1 --partition=gpu scripts/slurm/15_tests.slurm
#   TEST_MARK="not slow" sbatch scripts/slurm/15_tests.slurm  # skip training loops
#   TEST_MARK=unit sbatch scripts/slurm/15_tests.slurm        # structural only
#
# --gres and --partition are injected when called from submit_pipeline.sh.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"

load_gpu_modules
activate_trap

cd "${REPO_ROOT}"

TEST_PATH="${TEST_PATH:-tests}"
TEST_MARK="${TEST_MARK:-}"
JUNIT="logs/pytest_${SLURM_JOB_ID:-local}.xml"

mark_args=()
[[ -n "${TEST_MARK}" ]] && mark_args+=(-m "${TEST_MARK}")

cov_args=()
if python -c "import pytest_cov" 2>/dev/null; then
    cov_args+=(--cov=trap --cov-report=xml:"logs/coverage_${SLURM_JOB_ID:-local}.xml")
fi

echo "[15_tests] pytest ${TEST_PATH} -m '${TEST_MARK:-all}' (size=${TRAP_TEST_DATASET_SIZE:-tiny})"
python -m pytest "${TEST_PATH}" "${mark_args[@]}" \
    -ra -v \
    --junitxml="${JUNIT}" \
    "${cov_args[@]}"

echo "[15_tests] done -> ${JUNIT}"
