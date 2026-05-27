#!/bin/bash
#SBATCH --job-name=trap_24_mlm_smoke
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
# Stage 24 — MLM pre-flight smoke GATE. Trains a few hundred steps on a 1k-row
# debug subset (single GPU, no DDP — so the loss is read without the
# multi-process x2 artifact) and FAILS the job if the loss does not drop well
# below the uniform-random baseline ln(vocab).
#
# Wired before 25_tune_mlm / 30_mlm_pretrain so a config that cannot learn
# (e.g. a learning rate too low for from-scratch pretraining, bf16 update
# underflow, or a data/label-alignment bug) never reaches a multi-day run.
#
# Direct run:
#   RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/24_mlm_smoke.slurm
# Tunables (env): MLM_SMOKE_CONFIG, MLM_SMOKE_MARGIN (default 1.0 nats).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

SMOKE_MODEL="smoke.${MODEL_MLM}"
SMOKE_CONFIG="${MLM_SMOKE_CONFIG:-${REPO_ROOT}/config/training/mlm.smoke.json}"
SMOKE_OUT="${MODELS_DIR}/${SMOKE_MODEL}"
MARGIN="${MLM_SMOKE_MARGIN:-1.0}"

rm -rf "${SMOKE_OUT}"

# Single-process (plain python, not accelerate) keeps the reported loss free of
# the DDP x2 aggregation artifact, so the threshold check below is exact.
python "${REPO_ROOT}/trap/modeling/train.py" masking \
    "${SMOKE_MODEL}" \
    --pretrained-tokenizer-path "${TOKENIZER_PATH}" \
    --albert-config-path "${ALBERT_CONFIG}" \
    --trainer-config-path "${SMOKE_CONFIG}" \
    --k "${K}" \
    --preprocessing-name "${MLM_PROCESSING_NAME}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --debug || { echo "[24_mlm_smoke] FAIL: training crashed"; exit 1; }

python - "${SMOKE_OUT}" "${MARGIN}" <<'PY'
import json, math, sys, pathlib

out, margin = pathlib.Path(sys.argv[1]), float(sys.argv[2])
metrics = json.loads((out / "all_results.json").read_text())
cfg = json.loads((out / "final" / "config.json").read_text())
vocab = cfg["vocab_size"]
loss = metrics.get("final_train_loss", metrics.get("train_loss"))
uniform = math.log(vocab)
print(
    f"[24_mlm_smoke] vocab={vocab} uniform=ln(vocab)={uniform:.3f} "
    f"final_loss={loss} threshold<{uniform - margin:.3f} (margin={margin})"
)
if loss is None or not (loss < uniform - margin):
    print(
        "[24_mlm_smoke] FAIL: loss did not drop below the uniform baseline — the "
        "model is NOT learning. Before launching 30_mlm_pretrain, check: "
        "(1) learning rate (from-scratch needs ~1e-4..5e-4, not fine-tuning 5e-5), "
        "(2) bf16 update underflow (try fp32), (3) data/label alignment."
    )
    sys.exit(1)
print("[24_mlm_smoke] PASS: MLM training reduces loss below the uniform baseline.")
PY
status=$?

rm -rf "${SMOKE_OUT}"
exit $status
