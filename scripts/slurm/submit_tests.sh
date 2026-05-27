#!/bin/bash
# Submit the TrAP test suite as two parallel SLURM jobs:
#   15_tests_cpu — unit + integration (not slow), CPU node, < 10 min
#   15_tests_gpu — training loops + HPO + inference (slow), GPU node, < 15 min
#
# The two jobs are independent and run in parallel — no afterok chain.
#
# Usage:
#   bash scripts/slurm/submit_tests.sh                                        # both (default)
#   bash scripts/slurm/submit_tests.sh --cpu-only                             # 15 cpu only
#   bash scripts/slurm/submit_tests.sh --gpu-only                             # 15 gpu only
#   bash scripts/slurm/submit_tests.sh --run-config config/runs/salmon.yaml
#   bash scripts/slurm/submit_tests.sh --dry-run
#   bash scripts/slurm/submit_tests.sh --account 123456789
#   bash scripts/slurm/submit_tests.sh --gres=gpu:a40:1                       # override GPU type
#
# mail-user, account, --gres, and --partition are read from config/hpc/grace.yaml
# (or $HPC_CONFIG).  Any unrecognised flag is passed through to every sbatch call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Load SLURM config from YAML ------------------------------------------
_HPC_CONFIG="${HPC_CONFIG:-}"
if [[ -z "${_HPC_CONFIG}" ]]; then
    [[ -f "${REPO_ROOT}/config/hpc/grace.yaml" ]] \
        && _HPC_CONFIG="${REPO_ROOT}/config/hpc/grace.yaml" \
        || _HPC_CONFIG="${REPO_ROOT}/config/hpc/default.yaml"
fi
_PY="$(command -v python3 2>/dev/null || echo '')"
if [[ -n "${_PY}" && -f "${REPO_ROOT}/scripts/slurm/hpc_config.py" && -f "${_HPC_CONFIG}" ]]; then
    eval "$("${_PY}" "${REPO_ROOT}/scripts/slurm/hpc_config.py" --vars "${_HPC_CONFIG}" 2>/dev/null)" || true
fi
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-END,FAIL}"
SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpu}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-}"

DRY_RUN=0
ACCOUNT_OVERRIDE=""
DO_CPU=0
DO_GPU=0
RUN_CONFIG="${RUN_CONFIG:-}"
EXTRA=()

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --account)    ACCOUNT_OVERRIDE="$2"; shift 2 ;;
        --run-config) RUN_CONFIG="$2"; shift 2 ;;
        --cpu-only)   DO_CPU=1; shift ;;
        --gpu-only)   DO_GPU=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            EXTRA+=("$1"); shift ;;
    esac
done

# Default: run both.
if [[ "$DO_CPU" == "0" && "$DO_GPU" == "0" ]]; then
    DO_CPU=1; DO_GPU=1
fi

[[ -n "${RUN_CONFIG}" ]] && export RUN_CONFIG
[[ -n "${ACCOUNT_OVERRIDE}" ]] && SLURM_ACCOUNT="${ACCOUNT_OVERRIDE}"

mkdir -p logs

# Build per-class sbatch arg arrays.
common_args=()
[[ -n "${SLURM_ACCOUNT}"   ]] && common_args+=("--account=${SLURM_ACCOUNT}")
[[ -n "${SLURM_MAIL_USER}" ]] && common_args+=("--mail-user=${SLURM_MAIL_USER}" "--mail-type=${SLURM_MAIL_TYPE}")
[[ ${#EXTRA[@]} -gt 0      ]] && common_args+=("${EXTRA[@]}")

cpu_args=("${common_args[@]+"${common_args[@]}"}")

gpu_args=("${common_args[@]+"${common_args[@]}"}")
[[ -n "${SLURM_GPU_GRES}"       ]] && gpu_args+=("--gres=${SLURM_GPU_GRES}")
[[ -n "${SLURM_PARTITION_GPU}"  ]] && gpu_args+=("--partition=${SLURM_PARTITION_GPU}")

submit_one() {
    local script="scripts/slurm/${1}"
    shift
    local args=("$@")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch --parsable ${args[*]:-} ${script}" >&2
        echo "<jobid_${script}>"
    else
        local jobid
        jobid=$(sbatch --parsable "${args[@]+"${args[@]}"}" "${script}")
        echo "submitted ${script} as job ${jobid}" >&2
        echo "${jobid}"
    fi
}

[[ "$DO_CPU" == "1" ]] && submit_one 15_tests_cpu.slurm "${cpu_args[@]+"${cpu_args[@]}"}" > /dev/null
[[ "$DO_GPU" == "1" ]] && submit_one 15_tests_gpu.slurm "${gpu_args[@]+"${gpu_args[@]}"}" > /dev/null
