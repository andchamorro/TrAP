#!/bin/bash
#SBATCH --job-name=trap_15_tests_cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 15 cpu — unit + integration tests (no training loops, no GPU required).
# Runs the full suite minus @pytest.mark.slow.  Completes in < 10 min on CPU.
#
# Pair: 15_tests_gpu.slurm runs the slow (training / HPO / inference) tests.
# Orchestrator: bash scripts/slurm/submit_tests.sh  (submits both in parallel)
#
#   sbatch scripts/slurm/15_tests_cpu.slurm
#   TEST_MARK=unit sbatch scripts/slurm/15_tests_cpu.slurm        # structural only
#   TEST_PATH=tests/utils sbatch scripts/slurm/15_tests_cpu.slurm  # single directory

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"

load_cpu_modules
activate_trap

cd "${REPO_ROOT}"

TEST_PATH="${TEST_PATH:-tests}"
TEST_MARK="${TEST_MARK:-not slow}"
JUNIT="logs/pytest_cpu_${SLURM_JOB_ID:-local}.xml"

mark_args=()
[[ -n "${TEST_MARK}" ]] && mark_args+=(-m "${TEST_MARK}")

cov_args=()
if python -c "import pytest_cov" 2>/dev/null; then
    cov_args+=(--cov=trap --cov-report=xml:"logs/coverage_cpu_${SLURM_JOB_ID:-local}.xml")
fi

echo "[15_tests_cpu] pytest ${TEST_PATH} -m '${TEST_MARK}' (size=${TRAP_TEST_DATASET_SIZE:-tiny})"
python -m pytest "${TEST_PATH}" "${mark_args[@]}" \
    -ra -v \
    --junitxml="${JUNIT}" \
    "${cov_args[@]}"

echo "[15_tests_cpu] done -> ${JUNIT}"
