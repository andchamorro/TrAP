#!/bin/bash
#SBATCH --job-name=trap_34_classification_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Stage 34 — classification pre-flight smoke GATE. Fine-tunes a few hundred
# steps on a 1k-row debug subset (single GPU) from a RANDOM-INIT classifier
# (Track A: no MLM pre-training) and FAILS the job if the loss does not drop
# below the uniform-random baseline ln(num_labels) — i.e. the classifier is not
# learning (wrong LR, vocab/embedding mismatch, or label/range mismatch).
#
# Track A wiring: the random-init AlbertForSequenceClassification is built from
# ALBERT_CONFIG (vocab 65541), the salmon tokenizer from TOKENIZER_PATH, and the
# class-weighted loss in classification.smoke.json. No MLM checkpoint required.
# Wired before 35_tune_classification / 40_classification.
#
# Direct run:
#   RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/34_classification_smoke.slurm
# Tunables (env): CLS_SMOKE_CONFIG, CLS_SMOKE_MARGIN (default 0.2 nats).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

SMOKE_MODEL="smoke.${MODEL_CLS}"
SMOKE_CONFIG="${CLS_SMOKE_CONFIG:-${REPO_ROOT}/config/training/classification.smoke.json}"
SMOKE_OUT="${MODELS_DIR}/${SMOKE_MODEL}"
MARGIN="${CLS_SMOKE_MARGIN:-0.2}"

rm -rf "${SMOKE_OUT}"

# Track A: random-init classifier. Build from ALBERT_CONFIG + the salmon
# tokenizer (no --pretrained-model-path, so model_init falls back to from-config
# random init). For the optional P3 MLM-init ablation, add
# --pretrained-model-path "${MODEL_MLM}".
python "${REPO_ROOT}/trap/modeling/train.py" classification \
    "${SMOKE_MODEL}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --trainer-config-path "${SMOKE_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --debug || { echo "[34_classification_smoke] FAIL: training crashed"; exit 1; }

python - "${SMOKE_OUT}" "${MARGIN}" <<'PY'
import json, math, sys, pathlib

out, margin = pathlib.Path(sys.argv[1]), float(sys.argv[2])
metrics = json.loads((out / "all_results.json").read_text())
cfg = json.loads((out / "final" / "config.json").read_text())
num_labels = cfg.get("num_labels") or len(cfg.get("id2label", {})) or 2
loss = metrics.get("train_loss")
uniform = math.log(num_labels)
print(
    f"[34_classification_smoke] num_labels={num_labels} "
    f"uniform=ln(num_labels)={uniform:.3f} train_loss={loss} "
    f"threshold<{uniform - margin:.3f} (margin={margin})"
)
if loss is None or not (loss < uniform - margin):
    print(
        "[34_classification_smoke] FAIL: loss did not drop below the uniform "
        "baseline — the classifier is NOT learning. Before launching "
        "40_classification, check the pretrained backbone is loaded (not random), "
        "the learning rate, and that labels are in range."
    )
    sys.exit(1)
print("[34_classification_smoke] PASS: classification training reduces loss.")
PY
status=$?

rm -rf "${SMOKE_OUT}"
exit $status
