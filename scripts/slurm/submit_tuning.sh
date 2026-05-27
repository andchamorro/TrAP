#!/bin/bash
# Submit the OPTIONAL Optuna tuning sweep: a job array of workers on a shared
# study, then a single tune_finalize (afterok) that writes the
# config/training/*.tuned.json the 30/40 stages can consume. Run from anywhere:
#
#   bash scripts/slurm/submit_tuning.sh                                      # both MLM + cls
#   bash scripts/slurm/submit_tuning.sh --classification                      # cls only
#   bash scripts/slurm/submit_tuning.sh --mlm                                 # MLM only
#   bash scripts/slurm/submit_tuning.sh --run-config config/runs/salmon.yaml  # salmon tokenizer
#   bash scripts/slurm/submit_tuning.sh --dry-run                             # print chain only
#   bash scripts/slurm/submit_tuning.sh --account 123456789                   # override account
#   bash scripts/slurm/submit_tuning.sh --gres=gpu:a40:2                      # override GPU type
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
DO_CLS=0
DO_MLM=0
RUN_CONFIG="${RUN_CONFIG:-}"
EXTRA=()

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --account) ACCOUNT_OVERRIDE="$2"; shift 2 ;;
        --run-config) RUN_CONFIG="$2"; shift 2 ;;
        --classification|--cls) DO_CLS=1; shift ;;
        --mlm) DO_MLM=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# Export RUN_CONFIG so every SLURM job inherits it via _common.sh.
[[ -n "${RUN_CONFIG}" ]] && export RUN_CONFIG

[[ -n "${ACCOUNT_OVERRIDE}" ]] && SLURM_ACCOUNT="${ACCOUNT_OVERRIDE}"

# Default: run both sweeps.
if [[ "$DO_CLS" == "0" && "$DO_MLM" == "0" ]]; then
    DO_CLS=1; DO_MLM=1
fi

mkdir -p logs

# HPO runs ONE GPU per array task: each trial is single-process (no DDP), which
# Trainer.hyperparameter_search requires once HyperbandPruner is in play (DDP +
# pruning desyncs the ranks -> "DDP expects same model across all ranks" / NCCL
# timeout). Parallelism comes from the array + shared Optuna journal, not extra
# GPUs per trial. Derive a 1-GPU gres from the configured (training) gres, or
# honor an explicit TUNE_GPU_GRES.
TUNE_GPU_GRES="${TUNE_GPU_GRES:-}"
if [[ -z "${TUNE_GPU_GRES}" && -n "${SLURM_GPU_GRES}" ]]; then
    TUNE_GPU_GRES="${SLURM_GPU_GRES%:*}:1"
fi

# Build shared GPU args (sweep workers are all GPU jobs)
gpu_args=()
[[ -n "${SLURM_ACCOUNT}" ]] && gpu_args+=("--account=${SLURM_ACCOUNT}")
[[ -n "${SLURM_MAIL_USER}" ]] && gpu_args+=("--mail-user=${SLURM_MAIL_USER}" "--mail-type=${SLURM_MAIL_TYPE}")
[[ -n "${TUNE_GPU_GRES}" ]] && gpu_args+=("--gres=${TUNE_GPU_GRES}")
[[ -n "${SLURM_PARTITION_GPU}" ]] && gpu_args+=("--partition=${SLURM_PARTITION_GPU}")
[[ ${#EXTRA[@]} -gt 0 ]] && gpu_args+=("${EXTRA[@]}")

# CPU args for finalize job (no gres/partition)
cpu_args=()
[[ -n "${SLURM_ACCOUNT}" ]] && cpu_args+=("--account=${SLURM_ACCOUNT}")
[[ -n "${SLURM_MAIL_USER}" ]] && cpu_args+=("--mail-user=${SLURM_MAIL_USER}" "--mail-type=${SLURM_MAIL_TYPE}")
[[ ${#EXTRA[@]} -gt 0 ]] && cpu_args+=("${EXTRA[@]}")

submit_one() {
    local name="$1"
    local script="scripts/slurm/${name}"
    shift
    local args=("$@")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch --parsable ${args[*]:-} ${script}" >&2
        echo "<jobid_${name}>"
    else
        local jobid
        jobid=$(sbatch --parsable ${args[@]+"${args[@]}"} "${script}")
        echo "submitted ${script} as job ${jobid}" >&2
        echo "${jobid}"
    fi
}

# Submit each sweep and its own finalize (afterok the sweep's array).
# MLM: 25_tune_mlm → 25_tune_mlm_finalize
# CLS: 35_tune_classification → 35_tune_classification_finalize

if [[ "$DO_MLM" == "1" ]]; then
    mlm_id="$(submit_one 25_tune_mlm.slurm "${gpu_args[@]+"${gpu_args[@]}"}")"
    mlm_fin_args=("${cpu_args[@]+"${cpu_args[@]}"}" "--dependency=afterok:${mlm_id}")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch --parsable ${mlm_fin_args[*]} scripts/slurm/25_tune_mlm_finalize.slurm  (after: ${mlm_id})"
    else
        mlm_fin_id=$(sbatch --parsable "${mlm_fin_args[@]}" scripts/slurm/25_tune_mlm_finalize.slurm)
        echo "submitted 25_tune_mlm_finalize.slurm as job ${mlm_fin_id} (after: ${mlm_id})"
    fi
fi

if [[ "$DO_CLS" == "1" ]]; then
    cls_id="$(submit_one 35_tune_classification.slurm "${gpu_args[@]+"${gpu_args[@]}"}")"
    cls_fin_args=("${cpu_args[@]+"${cpu_args[@]}"}" "--dependency=afterok:${cls_id}")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch --parsable ${cls_fin_args[*]} scripts/slurm/35_tune_classification_finalize.slurm  (after: ${cls_id})"
    else
        cls_fin_id=$(sbatch --parsable "${cls_fin_args[@]}" scripts/slurm/35_tune_classification_finalize.slurm)
        echo "submitted 35_tune_classification_finalize.slurm as job ${cls_fin_id} (after: ${cls_id})"
    fi
fi
