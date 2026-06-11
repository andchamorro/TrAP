#!/bin/bash
# Shared environment for the TrAP Phase-3 training reproduction pipeline.
# SOURCED (not executed) by every scripts/slurm/*.slurm stage:
#   source "$(dirname "$0")/_common.sh"
#
# Module versions and required commands are read from config/hpc/grace.yaml
# (or $HPC_CONFIG).  Every value is still overridable from the submission env:
#   ANACONDA_MODULE=Anaconda3/2024.10 sbatch scripts/slurm/30_mlm_pretrain.slurm
#
# To target a different HPC site, copy config/hpc/grace.yaml, adjust the
# module versions, and point HPC_CONFIG to the new file.

set -euo pipefail

# --- Repo root (this file lives in scripts/slurm/) -------------------------
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_COMMON_DIR}/../.." && pwd)"
export REPO_ROOT

# --- HPC config YAML -------------------------------------------------------
# Look for a site-specific config first; fall back to generic defaults.
if [[ -z "${HPC_CONFIG:-}" ]]; then
    if [[ -f "${REPO_ROOT}/config/hpc/grace.yaml" ]]; then
        HPC_CONFIG="${REPO_ROOT}/config/hpc/grace.yaml"
    else
        HPC_CONFIG="${REPO_ROOT}/config/hpc/default.yaml"
    fi
fi

# Load module names + conda env from the YAML config.
# Skips variables already set in the environment (env var overrides take priority).
# Falls back silently to hardcoded defaults if Python or the config is missing.
_load_hpc_config() {
    local py config_script
    py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo '')"
    config_script="${REPO_ROOT}/scripts/slurm/hpc_config.py"
    if [[ -z "${py}" || ! -f "${config_script}" || ! -f "${HPC_CONFIG}" ]]; then
        echo "[_common] WARNING: hpc_config.py or ${HPC_CONFIG} not found; using hardcoded defaults" >&2
        return
    fi
    local vars
    vars="$("${py}" "${config_script}" --vars "${HPC_CONFIG}")" || {
        echo "[_common] WARNING: failed to parse ${HPC_CONFIG}; using hardcoded defaults" >&2
        return
    }
    eval "${vars}"
}

_load_hpc_config

# --- Run config YAML (config/runs/*.yaml) ----------------------------------
# Loaded after HPC config so it can override K, VOCAB, TOKENIZER_NAME, etc.
# Priority: env var > run config YAML > _common.sh hardcoded defaults below.

RUN_CONFIG="${RUN_CONFIG:-}"

_load_run_config() {
    local py config_script run_cfg vars
    py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo '')"
    config_script="${REPO_ROOT}/scripts/slurm/hpc_config.py"
    run_cfg="${1:-${RUN_CONFIG}}"
    # Resolve relative path against repo root
    if [[ -n "${run_cfg}" && "${run_cfg:0:1}" != "/" ]]; then
        run_cfg="${REPO_ROOT}/${run_cfg}"
    fi
    if [[ -z "${run_cfg}" ]]; then
        return 0  # no run config specified; use hardcoded defaults
    fi
    if [[ ! -f "${run_cfg}" ]]; then
        echo "[_common] WARNING: RUN_CONFIG not found: ${run_cfg}" >&2
        return
    fi
    if [[ -z "${py}" || ! -f "${config_script}" ]]; then
        echo "[_common] WARNING: cannot load run config (missing python or hpc_config.py)" >&2
        return
    fi
    vars="$("${py}" "${config_script}" --run-vars "${run_cfg}")" || {
        echo "[_common] WARNING: failed to parse ${run_cfg}; using defaults" >&2
        return
    }
    eval "${vars}"
    # SentencePiece is deprecated: its metaspace/Whitespace pre-tokenizer mismatch
    # fragments each k-mer into ~16 char-level pieces, silently destroying the
    # tokenization. Use the Salmon canonical k-mer tokenizer (config/runs/salmon.yaml).
    if [[ "${TOKENIZER_ALGORITHM:-}" == "spm" || "${TOKENIZER_NAME:-}" == *.spm ]]; then
        echo "[_common] ERROR: SentencePiece (spm) tokenizer is DEPRECATED and disabled." >&2
        echo "  Run config '${run_cfg}' selects algorithm=spm / a .spm tokenizer." >&2
        echo "  Reason: metaspace pre-tokenizer mismatch fragments k-mers (~16 char" >&2
        echo "          pieces each), silently corrupting the tokenized dataset." >&2
        echo "  Fix: use config/runs/salmon.yaml (canonical k-mer tokenizer)." >&2
        exit 1
    fi
    echo "[_common] run config loaded: ${run_cfg}"
}

_load_run_config


# Hardcoded fallbacks — active only when _load_hpc_config fails or skips a var.
# Keep these in sync with config/hpc/grace.yaml as a last-resort safety net.
ANACONDA_MODULE="${ANACONDA_MODULE:-Anaconda3/2025.12-2}"
GCC_MODULE="${GCC_MODULE:-GCC/13.3.0}"
JELLYFISH_MODULE="${JELLYFISH_MODULE:-Jellyfish/2.3.1}"   # loaded after GCC_MODULE
SEQKIT_MODULE="${SEQKIT_MODULE:-SeqKit/2.9.0}"
CUDA_MODULES="${CUDA_MODULES:-GCCcore/13.3.0 CUDA/12.6.0 NCCL/2.22.3-CUDA-12.6.0}"
BWA_MODULE="${BWA_MODULE:-GCC/13.3.0 BWA/0.7.18}"
BIO_MODULES="${BIO_MODULES:-GCCcore/13.3.0 GCC/13.3.0 STAR/2.7.11b SAMtools/1.21 BEDTools/2.31.1}"
CONDA_ENV="${CONDA_ENV:-trap}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
# SLURM submission settings (injected by submit_pipeline.sh / submit_tuning.sh)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-END,FAIL}"
SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpu}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-}"

# --- Module loaders --------------------------------------------------------

load_cpu_modules() {
    module purge
    module load ${GCC_MODULE} ${ANACONDA_MODULE}
    check_required_commands base
}

load_gpu_modules() {
    module purge
    module load ${ANACONDA_MODULE}
    module load ${CUDA_MODULES}
    module load WebProxy 2>/dev/null || true   # internet from compute nodes (if needed)
    check_required_commands cuda
}

load_bio_modules() {
    module purge
    module load ${ANACONDA_MODULE}
    module load ${BIO_MODULES}
    check_required_commands bio
}

activate_trap() {
    # conda's hook trips `set -u`; guard it.
    set +u
    eval "$(conda shell.bash hook)"
    # module purge resets PATH but leaves CONDA_PREFIX set; deactivate first so
    # the subsequent activate is always a fresh re-entry rather than a no-op.
    conda deactivate 2>/dev/null || true
    conda activate "${CONDA_ENV}"
    set -u
    # Fail loudly if Python is not coming from the expected env.
    local py_path
    py_path="$(command -v python)"
    if [[ "${py_path}" != *"/envs/${CONDA_ENV}/"* && "${py_path}" != *"/${CONDA_ENV}/bin/"* ]]; then
        echo "[_common] ERROR: conda activate '${CONDA_ENV}' did not take effect" \
             "(python: ${py_path})" >&2
        return 1
    fi
    echo "[_common] ✓ conda env '${CONDA_ENV}' active (python: ${py_path})"
}

# --- Safety check: verify required commands are in PATH --------------------
# Called automatically by load_*_modules() after loading modules.
# Exits 1 with a clear message + fix hint if any command is missing.
check_required_commands() {
    local group="${1:-}"
    [[ -z "${group}" ]] && return 0

    local py config_script
    py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo '')"
    config_script="${REPO_ROOT}/scripts/slurm/hpc_config.py"
    if [[ -z "${py}" || ! -f "${config_script}" || ! -f "${HPC_CONFIG}" ]]; then
        return 0  # can't check without the helper; fail open
    fi

    local cmds
    cmds="$("${py}" "${config_script}" --check-cmds "${group}" "${HPC_CONFIG}" 2>/dev/null)" \
        || return 0

    local missing=()
    for cmd in ${cmds}; do
        command -v "${cmd}" &>/dev/null || missing+=("${cmd}")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "" >&2
        echo "[_common] ERROR: required commands not in PATH after loading '${group}' modules:" >&2
        printf '  missing: %s\n' "${missing[@]}" >&2
        echo "" >&2
        echo "  Fix: update the module version in ${HPC_CONFIG}" >&2
        echo "  Hint: run 'module spider <tool>' on Grace to find the available version." >&2
        echo "" >&2
        return 1
    fi
    echo "[_common] ✓ ${group} commands OK"
}

# --- HuggingFace: fully offline on compute nodes --------------------------
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Keep TMPDIR on fast node-local scratch.  Under SLURM on Grace, TMPDIR is
# already set to a per-job /tmp/job.* directory backed by node-local NVMe (see
# the HF_DATASETS_CACHE note below) — that is exactly where we want HuggingFace
# to write the intermediate shard files Dataset.map(num_proc>1) produces, so we
# keep it.  The ${TMPDIR:-...} fallback only fires for non-SLURM/interactive
# runs where TMPDIR is unset: there we use REPO_ROOT/../.tmp rather than the
# system /tmp (which may be small or RAM-backed off the compute nodes).
export TMPDIR="${TMPDIR:-$(dirname "${REPO_ROOT}")/.tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || true

# Keep the HuggingFace *datasets* cache (Dataset.from_generator output + the
# intermediate shards Dataset.map writes) on node-local NVMe — on Grace,
# SLURM sets TMPDIR=/tmp/job.* which is a local NVMe xfs disk (~1.5 TB), not a
# RAM tmpfs.  HF_HOME lives on Lustre (parallel FS), which is ~100x slower for
# the many small writes map produces; routing only the datasets cache to NVMe
# is the main stage-20 I/O win.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets}"
mkdir -p "${HF_DATASETS_CACHE}" 2>/dev/null || true

# --- Canonical Phase-3 parameters -----------------------------------------
# (mirror config/datasets/l1hs_l1pa2_v48_k17.yaml — keep both in sync)
export K="${K:-17}"
# VOCAB is a LABEL ONLY for the legacy SPM/BPE tokenizer training arg — it does
# NOT set the model's effective vocab. The Salmon tokenizer (current default)
# ignores it: its vocab is 5 specials + n_hash buckets (= 65541 at n_hash=65536),
# and train.py overrides albert_config.vocab_size from len(tokenizer) at runtime.
# Do not read this as the model vocab; the per-run manifest records the real one.
export VOCAB="${VOCAB:-32000}"
export MAX_POSITION="${MAX_POSITION:-1280}"
# Classification uses short paired reads (150bp, k=17 → 271 real tokens).
# Padding to MAX_POSITION (1280) wastes 79% of each example and gives ALBERT
# 22x more attention FLOPs than needed (O(n²): 1280² vs 272²).
# MAX_POSITION_CLS caps padding for the classification tokenised dataset only;
# the ALBERT model still supports up to MAX_POSITION positions at inference.
# Formula: (read_len - k + 1) * 2 + 3 special tokens, rounded up to mult of 8.
#   (150-17+1)*2 + 3 = 271 → 272; add one mult-of-8 margin → 280.
export MAX_POSITION_CLS="${MAX_POSITION_CLS:-280}"
export SEED="${SEED:-3469}"

# Inputs / outputs (relative to REPO_ROOT unless absolute)
export DATA_EXTERNAL="${DATA_EXTERNAL:-${REPO_ROOT}/data/external}"
export PROCESSED_DIR="${PROCESSED_DIR:-${REPO_ROOT}/data/processed}"
export MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
export GENCODE_FASTA="${GENCODE_FASTA:-${DATA_EXTERNAL}/gencode.v48.transcripts.fa.gz}"
export L1_CORPUS="${L1_CORPUS:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_rm.LINE1.gencode.v48.fa}"
export REPEATMASKER_GFF="${REPEATMASKER_GFF:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_rm.gff}"

# Names
export TOKENIZER_NAME="${TOKENIZER_NAME:-tokenizer.gencode.v48.k17.32k}"
export MLM_PROCESSING_NAME="${MLM_PROCESSING_NAME:-gencode.v48.k17.32k}"
export PROCESSING_NAME="${PROCESSING_NAME:-gencode.v48.k17.32k/l1hs_l1pa2}"
export MODEL_MLM="${MODEL_MLM:-albert.gencode.v48.k17.32k}"
export MODEL_CLS="${MODEL_CLS:-albert.l1hs_l1pa2.v48.k17.32k}"

# L1 paired FASTQ produced by the ART->STAR->bedtools dataset stage
export L1_R1="${L1_R1:-${DATA_EXTERNAL}/l1hs_l1pa2_negative.5x_R1.fq}"
export L1_R2="${L1_R2:-${DATA_EXTERNAL}/l1hs_l1pa2_negative.5x_R2.fq}"

# Config files
export ALBERT_CONFIG="${ALBERT_CONFIG:-${REPO_ROOT}/config/albert_config_k17_v48.json}"
export MLM_TRAINER_CONFIG="${MLM_TRAINER_CONFIG:-${REPO_ROOT}/config/training/mlm.json}"
export CLS_TRAINER_CONFIG="${CLS_TRAINER_CONFIG:-${REPO_ROOT}/config/training/classification_final.json}"
export TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-${REPO_ROOT}/config/datasets/l1hs_l1pa2_v48_k17.yaml}"

# Resolve any relative paths from the run config against REPO_ROOT.
# Paths set by the run config YAML are relative to the repo root; they bypass
# the ${VAR:-${REPO_ROOT}/...} defaults above and must be made absolute here.
_make_absolute() {
    local var="$1" val
    eval "val=\"\${${var}:-}\""
    if [[ -n "${val}" && "${val:0:1}" != "/" ]]; then
        eval "export ${var}=\"${REPO_ROOT}/${val}\""
    fi
}
_make_absolute GENCODE_FASTA
_make_absolute L1_CORPUS
_make_absolute REPEATMASKER_GFF
_make_absolute L1_R1
_make_absolute L1_R2
_make_absolute ALBERT_CONFIG
_make_absolute MLM_TRAINER_CONFIG
_make_absolute CLS_TRAINER_CONFIG
_make_absolute TOKENIZER_CONFIG

# Derived paths used by more than one stage
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODELS_DIR}/${TOKENIZER_NAME}}"
export PREPROCESSING_OUT="${PREPROCESSING_OUT:-${PROCESSED_DIR}/${PROCESSING_NAME}}"
export MLM_PREPROCESSING_OUT="${MLM_PREPROCESSING_OUT:-${PROCESSED_DIR}/${MLM_PROCESSING_NAME}}"

# --- Optional: Optuna hyperparameter tuning --------------------------------
export TUNE_DIR="${TUNE_DIR:-${SCRATCH:-$HOME}/trap_hpo}"
export STUDY_CLS="${STUDY_CLS:-cls.${MLM_PROCESSING_NAME//\//_}}"
export STUDY_MLM="${STUDY_MLM:-mlm.${MLM_PROCESSING_NAME//\//_}}"
export TUNE_JOURNAL_CLS="${TUNE_JOURNAL_CLS:-${TUNE_DIR}/${STUDY_CLS}.journal}"
export TUNE_JOURNAL_MLM="${TUNE_JOURNAL_MLM:-${TUNE_DIR}/${STUDY_MLM}.journal}"
export TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-2}"
export CLS_SEARCH_CONFIG="${CLS_SEARCH_CONFIG:-${REPO_ROOT}/config/tuning/classification_optuna.yaml}"
export MLM_SEARCH_CONFIG="${MLM_SEARCH_CONFIG:-${REPO_ROOT}/config/tuning/mlm_optuna.yaml}"

# --- GPU / accelerate ------------------------------------------------------
accelerate_args() {
    echo "--multi_gpu --num_machines 1 --num_processes=${GPUS_PER_NODE}" \
         "--mixed_precision bf16 --dynamo_backend no"
}

# SINGLE-process (one GPU, NO DDP) launch — used by the Optuna HPO stages
# (25_tune_mlm / 35_tune_classification). Trainer.hyperparameter_search is not
# DDP-safe with Optuna pruning: when HyperbandPruner kills a trial, the train
# loop exits on rank 0 while rank 1 is mid-collective, so the next trial's DDP
# re-init aborts ("DDP expects same model across all ranks, but Rank 0 has N
# params, while rank 1 has inconsistent M params") or hangs to the 30-min NCCL
# watchdog timeout. HPO parallelism comes from the job ARRAY sharing one Optuna
# journal, not from DDP within a trial. Final training (30/40) still uses DDP.
accelerate_args_single() {
    echo "--num_machines 1 --num_processes 1 --mixed_precision bf16 --dynamo_backend no"
}

echo "[_common] REPO_ROOT=${REPO_ROOT} ENV=${CONDA_ENV} K=${K} VOCAB=${VOCAB} HPC_CONFIG=${HPC_CONFIG}"
