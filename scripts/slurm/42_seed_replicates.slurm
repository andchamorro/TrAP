#!/bin/bash
#SBATCH --job-name=trap_42_seed_replicates
# The run of record early-stopped at epoch 9 in 18.4 h (~2.0 h/epoch). The config
# allows 12 epochs with early_stopping_patience=2, so a seed that never triggers
# early stopping needs ~24.5 h; 36 h covers that with margin without asking for
# stage 40's full 48 h, which schedules worse.
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Stage 42 — seed replicates for the classification model (Reviewer 3, Comment 5,
# "whether results are stable across random seeds or multiple training runs").
#
# Submit in batches of 3 concurrent tasks (GPUs are scarce; %3 throttles the
# array rather than queueing 15 two-GPU jobs at once):
#
#   sbatch --array=0-14%3 scripts/slurm/42_seed_replicates.slurm
#
# Do NOT pass RUN_CONFIG. `scripts/slurm/hpc_config.py --run-vars` emits
# `export MODEL_CLS=...` UNCONDITIONALLY, so a run config silently overrides the
# per-seed model name and every task would write into the same directory — the
# directory of the model of record. The bare defaults in _common.sh already track
# the winning model, and the guards below fail closed if anything drifts.
#
# DISK. This is a stability exploration, not a model we ship, so checkpoint
# retention is cut twice over: the per-seed config drops save_total_limit from 5
# to 1, and the checkpoints are deleted once `final/` exists. Only `final/`
# (63 MB) is needed to score the held-out split. 999 MB/run -> ~63 MB/run, so the
# 15 replicates cost ~1 GB rather than ~15 GB. save_total_limit cannot go to
# save_strategy="no": load_best_model_at_end=true needs checkpoints to restore
# from, and dropping it would change which model each seed selects and make the
# replicates incomparable to the run of record. Set KEEP_CHECKPOINTS=1 to retain.
#
# NAMING CONVENTION. Replicates extend the existing dotted scheme with a
# `.seed<N>` suffix, alongside the established `.distilled` / `.e12` variants:
#
#   albert.l1hs_l1pa2.v48.k17.salmon.seqlabel          <- model of record, NEVER written here
#   albert.l1hs_l1pa2.v48.k17.salmon.seqlabel.seed3470 <- replicate, array index 0
#   ...                                          .seed3484 <- replicate, array index 14
#
# Seeds are 3470 + array index, chosen to be contiguous, documented and distinct
# from the 3469 of the run of record. ~1 GB per replicate (final + 5 checkpoints).
#
# NOTE for whoever reads the outputs: this config sets
# dataloader_persistent_workers=true, which is the setting behind the stage-40
# eval-cache defect. Each replicate's own eval_results.json therefore reports the
# PER-EPOCH split, not the held-out split. Score the held-out split separately:
#   python -m trap.analysis heldout-eval --model <name> --out results/seeds/<name>

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

PROTECTED_MODEL="albert.l1hs_l1pa2.v48.k17.salmon.seqlabel"
EXPECTED_DATASET="gencode.v48.k17.salmon.seqlabel/l1hs_l1pa2"

SEED=$(( 3470 + ${SLURM_ARRAY_TASK_ID:-0} ))
MODEL_CLS="${PROTECTED_MODEL}.seed${SEED}"      # set AFTER sourcing, so nothing overrides it
export MODEL_CLS

# --- guards: fail closed rather than overwrite anything -------------------
if [[ "${MODEL_CLS}" == "${PROTECTED_MODEL}" ]]; then
    echo "[42] ERROR: refusing to write the model of record (${PROTECTED_MODEL})." >&2
    exit 1
fi
if [[ "${PROCESSING_NAME}" != "${EXPECTED_DATASET}" ]]; then
    echo "[42] ERROR: PROCESSING_NAME is '${PROCESSING_NAME}', expected '${EXPECTED_DATASET}'." >&2
    echo "        A seed replicate must train on the same build as the model of record;" >&2
    echo "        salmon.yaml names the other (~89% NEGATIVE) build. Do not pass RUN_CONFIG." >&2
    exit 1
fi
if [[ -e "${MODELS_DIR}/${MODEL_CLS}/final" && "${FORCE_OVERWRITE:-0}" != "1" ]]; then
    echo "[42] ERROR: ${MODELS_DIR}/${MODEL_CLS}/final already exists." >&2
    echo "        Refusing to overwrite a trained model. Set FORCE_OVERWRITE=1 to override." >&2
    exit 1
fi

# Per-seed trainer config. Each task writes its own uniquely named file, so array
# tasks cannot race; /run/ is gitignored and persists for inspection.
SEED_CFG_DIR="${REPO_ROOT}/run/seed_configs"
mkdir -p "${SEED_CFG_DIR}"
CLS_TRAINER_CONFIG="${SEED_CFG_DIR}/classification_final.seed${SEED}.json"
python -I - "${CLS_TRAINER_CONFIG}" "${SEED}" <<'PY'
import json, sys
from pathlib import Path
out, seed = Path(sys.argv[1]), int(sys.argv[2])
base = json.loads(Path("config/training/classification_final.json").read_text())
assert base["seed"] == 3469, f"base config seed changed: {base['seed']}"
base["seed"] = seed
# Exploration run: keep the smallest checkpoint set that still lets
# load_best_model_at_end restore the selected epoch. Training dynamics and the
# selected model are unchanged; only the retention of superseded checkpoints is.
base["save_total_limit"] = 1
out.write_text(json.dumps(base, indent=4) + "\n")
print(f"[42] wrote {out} (seed={seed}, save_total_limit=1, else identical)")
PY
export CLS_TRAINER_CONFIG

echo "[42] array task ${SLURM_ARRAY_TASK_ID:-0}  seed=${SEED}"
echo "[42] MODEL_CLS=${MODEL_CLS}"
echo "[42] PROCESSING_NAME=${PROCESSING_NAME}"
echo "[42] CLS_TRAINER_CONFIG=${CLS_TRAINER_CONFIG}"

# _common.sh sets `set -euo pipefail`, so a bare failure would exit before the
# cleanup below ever runs. Capture the status explicitly instead.
TRAIN_RC=0
accelerate launch $(accelerate_args) "${REPO_ROOT}/trap/modeling/train.py" classification \
    "${MODEL_CLS}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --trainer-config-path "${CLS_TRAINER_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --do-eval || TRAIN_RC=$?

MODEL_DIR="${MODELS_DIR}/${MODEL_CLS}"
if [[ "${TRAIN_RC}" -ne 0 ]]; then
    echo "[42] training exited ${TRAIN_RC}; keeping checkpoints for diagnosis." >&2
    exit "${TRAIN_RC}"
fi

# Drop the checkpoints now that final/ is written. Scoped deliberately: only
# under this replicate's own directory, only after a clean exit, only once
# final/ exists, and never for the protected model.
if [[ "${KEEP_CHECKPOINTS:-0}" == "1" ]]; then
    echo "[42] KEEP_CHECKPOINTS=1, retaining checkpoints."
elif [[ "${MODEL_CLS}" != "${PROTECTED_MODEL}" && -d "${MODEL_DIR}/final" ]]; then
    before=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1 || true)
    find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
    after=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1 || true)
    echo "[42] checkpoints removed: ${before} -> ${after} (final/ kept)"
else
    echo "[42] WARNING: final/ missing or protected name; leaving ${MODEL_DIR} untouched." >&2
fi

echo "[42] model -> ${MODEL_DIR}/final"
echo "[42] score it with: python -m trap.analysis heldout-eval --model ${MODEL_CLS} --out results/seeds/${MODEL_CLS}"
