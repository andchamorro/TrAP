#!/bin/bash
# Submit the TrAP Phase-3 training reproduction pipeline as an
# `sbatch --dependency=afterok` chain. Run from anywhere:
#
#   bash scripts/slurm/submit_pipeline.sh                              # BPE (default)
#   bash scripts/slurm/submit_pipeline.sh --run-config config/runs/spm.yaml
#   bash scripts/slurm/submit_pipeline.sh --dry-run                   # print the chain only
#   bash scripts/slurm/submit_pipeline.sh --from 10_tokenizer --to 40_classification
#   bash scripts/slurm/submit_pipeline.sh --mlm-tune                  # 25_tune_mlm → finalize
#   bash scripts/slurm/submit_pipeline.sh --cls-tune                  # 35_tune_cls → finalize
#   bash scripts/slurm/submit_pipeline.sh --account 123456789         # override account
#   bash scripts/slurm/submit_pipeline.sh --gres=gpu:a40:2            # override GPU type
#   bash scripts/slurm/submit_pipeline.sh -q                          # suppress submission logs
#
# Run configs (config/runs/*.yaml) set TOKENIZER_ALGORITHM, TOKENIZER_NAME,
# processing names, and model names.  Multiple algorithms can run in parallel:
#   bash scripts/slurm/submit_pipeline.sh --run-config config/runs/bpe.yaml
#   bash scripts/slurm/submit_pipeline.sh --run-config config/runs/spm.yaml
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

# GPU stages that need --gres and --partition injected
GPU_STAGES=("15_tests.slurm" "25_tune_mlm.slurm" "30_mlm_pretrain.slurm" "35_tune_classification.slurm" "40_classification.slurm" "50_benchmark.slurm")

STAGES=(
    "00_fetch_references.slurm"
    "10_tokenizer.slurm"
    "20_dataset.slurm"
    "21_dataset_diagnosis.slurm"
    "24_mlm_smoke.slurm"
    "30_mlm_pretrain.slurm"
    "34_classification_smoke.slurm"
    "40_classification.slurm"
    "50_benchmark.slurm"
)

DRY_RUN=0
QUIET=0
FROM=""
TO=""
ACCOUNT_OVERRIDE=""
RUN_CONFIG="${RUN_CONFIG:-}"
MLM_TUNE=0
CLS_TUNE=0
EXTRA=()

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -q|--quiet) QUIET=1; shift ;;
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --mlm-tune) MLM_TUNE=1; shift ;;
        --cls-tune) CLS_TUNE=1; shift ;;
        --account) ACCOUNT_OVERRIDE="$2"; shift 2 ;;
        --run-config) RUN_CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# --mlm-tune / --cls-tune delegate to submit_tuning.sh which has the correct
# job-array → afterok → finalize wiring.  Pass through all relevant flags and exit.
if [[ "$MLM_TUNE" == "1" || "$CLS_TUNE" == "1" ]]; then
    tune_args=()
    [[ "$MLM_TUNE" == "1" ]]        && tune_args+=("--mlm")
    [[ "$CLS_TUNE" == "1" ]]        && tune_args+=("--classification")
    [[ "$DRY_RUN" == "1" ]]         && tune_args+=("--dry-run")
    [[ -n "${RUN_CONFIG}" ]]        && tune_args+=("--run-config" "${RUN_CONFIG}")
    [[ -n "${ACCOUNT_OVERRIDE}" ]]  && tune_args+=("--account" "${ACCOUNT_OVERRIDE}")
    [[ ${#EXTRA[@]} -gt 0 ]]        && tune_args+=("${EXTRA[@]}")
    exec bash "${SCRIPT_DIR}/submit_tuning.sh" "${tune_args[@]}"
fi

# Export so every sbatch job inherits it (SLURM passes the submission env by default)
[[ -n "${RUN_CONFIG}" ]] && export RUN_CONFIG

# Eval the run config now so SLURM_TIME_N / SLURM_CPUS_N / SLURM_MEM_N are
# available in this shell for injecting as sbatch flags below.
if [[ -n "${RUN_CONFIG}" ]]; then
    _rc_path="${RUN_CONFIG}"
    [[ "${_rc_path:0:1}" != "/" ]] && _rc_path="${REPO_ROOT}/${_rc_path}"
    _rc_py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo '')"
    _rc_cfg="${REPO_ROOT}/scripts/slurm/hpc_config.py"
    if [[ -n "${_rc_py}" && -f "${_rc_cfg}" && -f "${_rc_path}" ]]; then
        eval "$("${_rc_py}" "${_rc_cfg}" --run-vars "${_rc_path}" 2>/dev/null)" || true
        [[ "$QUIET" == "0" ]] && echo "[submit] run config: ${_rc_path}"
    fi
fi

# CLI --account overrides YAML
[[ -n "${ACCOUNT_OVERRIDE}" ]] && SLURM_ACCOUNT="${ACCOUNT_OVERRIDE}"

idx_of() {
    local q="$1" i
    for i in "${!STAGES[@]}"; do
        local s="${STAGES[$i]}"
        if [[ "$s" == "$q" || "${s%.slurm}" == "$q" || "$s" == "$q"* ]]; then
            echo "$i"; return 0
        fi
    done
    echo "-1"
}

is_gpu_stage() {
    local s="$1"
    for gs in "${GPU_STAGES[@]}"; do [[ "$s" == "$gs" ]] && return 0; done
    return 1
}

start=0
end=$(( ${#STAGES[@]} - 1 ))
[[ -n "$FROM" ]] && start=$(idx_of "$FROM")
[[ -n "$TO" ]] && end=$(idx_of "$TO")
if [[ "$start" -lt 0 || "$end" -lt 0 || "$start" -gt "$end" ]]; then
    echo "error: bad --from/--to range (from='${FROM}' to='${TO}')" >&2
    exit 1
fi

mkdir -p logs

prev=""
for (( i = start; i <= end; i++ )); do
    stage="${STAGES[$i]}"
    script="scripts/slurm/${stage}"
    # Extract the numeric stage prefix, stripping leading zeros so it matches
    # the YAML key ("00_fetch.slurm" → base "00" → num 0, "10_tok.slurm" → 10)
    _stage_base="${stage%.slurm}"
    _stage_num=$(( 10#${_stage_base%%_*} ))

    args=()
    [[ -n "${SLURM_ACCOUNT}" ]] && args+=("--account=${SLURM_ACCOUNT}")
    [[ -n "${SLURM_MAIL_USER}" ]] && args+=("--mail-user=${SLURM_MAIL_USER}" "--mail-type=${SLURM_MAIL_TYPE}")
    if is_gpu_stage "${stage}"; then
        [[ -n "${SLURM_GPU_GRES}" ]] && args+=("--gres=${SLURM_GPU_GRES}")
        [[ -n "${SLURM_PARTITION_GPU}" ]] && args+=("--partition=${SLURM_PARTITION_GPU}")
    fi
    # Per-stage resource overrides from the run config (SLURM_TIME_N etc.)
    _t="SLURM_TIME_${_stage_num}"; [[ -n "${!_t:-}" ]] && args+=("--time=${!_t}")
    _c="SLURM_CPUS_${_stage_num}"; [[ -n "${!_c:-}" ]] && args+=("--cpus-per-task=${!_c}")
    _m="SLURM_MEM_${_stage_num}";  [[ -n "${!_m:-}" ]] && args+=("--mem=${!_m}")
    [[ -n "$prev" ]] && args+=("--dependency=afterok:${prev}")
    [[ ${#EXTRA[@]} -gt 0 ]] && args+=("${EXTRA[@]}")

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] sbatch ${args[*]:-} ${script}  (after: ${prev:-none})"
        prev="<jobid_${i}>"
    elif [[ "$QUIET" == "1" ]]; then
        # --parsable suppresses HPC accounting preamble; output is just the job ID.
        prev=$(sbatch --parsable ${args[@]+"${args[@]}"} "${script}" 2>/dev/null \
               | tail -1 | cut -d: -f1 | tr -d '[:space:]')
    else
        # No --parsable: sbatch prints the HPC accounting block (Project Account,
        # Account Balance, Requested SUs) followed by "Submitted batch job JOBID".
        # Capture and echo everything, then extract the job ID from the last line.
        sbatch_out=$(sbatch ${args[@]+"${args[@]}"} "${script}")
        echo "${sbatch_out}"
        jobid=$(echo "${sbatch_out}" | grep -E 'Submitted batch job' | grep -oE '[0-9]+$')
        echo "submitted ${stage} as job ${jobid} (after: ${prev:-none})"
        prev="${jobid}"
    fi
done
