#!/bin/bash
#SBATCH --job-name=trap_64_baseline_compare
#SBATCH --time=03:00:00
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
# Track B4 — head-to-head: coordinate-label model (option A) vs sequence-label model
# (option B), judged on a COMMON SEQUENCE-DEFINED ground truth at deployment priors.
#
# The decisive comparison is the two cross-evaluations on the *seqlabel* test split
# (sequence truth): does the coord-trained model detect actual L1 sequence, vs the
# seqlabel model? Plus each model on its NATIVE test split for reference.
#
# Per combo: boundary_calibration dump-logits (GPU) -> quantification scorecard (CPU,
# resampled to deployment priors), which now reports BOTH the filter purity/yield
# frontier AND 3-class per-class/macro-F1.
#
#   sbatch scripts/slurm/64_baseline_compare.slurm

export TRAP_SPM_EXPERIMENTAL=1
export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
mkdir -p results

COORD_MODEL="${COORD_MODEL:-albert.l1hs_l1pa2.v48.k17.salmon.lr5e5.e12}"   # option-A deploy
SEQ_MODEL="${SEQ_MODEL:-albert.l1hs_l1pa2.v48.k17.salmon.seqlabel}"        # option-B
TOK="${TOK:-tokenizer.gencode.v48.k17.salmon}"                            # same tokenizer
SEQ_PREP="${SEQ_PREP:-gencode.v48.k17.salmon.seqlabel/l1hs_l1pa2}"        # sequence-truth split
COORD_PREP="${COORD_PREP:-gencode.v48.k17.salmon/l1hs_l1pa2}"             # coordinate-truth split
PAD="${PAD:-280}"
SPLIT="${SPLIT:-test}"
MAX_READS="${MAX_READS:-400000}"

# dump_and_score <tag> <model_cls> <prep>
dump_and_score() {
    local tag="$1" model="$2" prep="$3"
    echo "[b4] === ${tag}: dump-logits ${model} over ${prep}:${SPLIT} ==="
    python -m trap.analysis.boundary_calibration dump-logits \
        --model-path "${MODELS_DIR}/${model}/final" --tokenizer-path "${MODELS_DIR}/${TOK}" \
        --preprocessing-name "${prep}" --split "${SPLIT}" \
        --max-position-embeddings "${PAD}" --max-reads "${MAX_READS}" \
        --out "results/b4_logits.${tag}.npz" \
        || { echo "[b4] ${tag} dump FAILED"; return 1; }
    python -m trap.analysis.quantification scorecard \
        --logits "results/b4_logits.${tag}.npz" --out "results/b4_quant.${tag}.json" \
        || echo "[b4] ${tag} scorecard FAILED"
}

# DECISIVE: both models on the sequence-defined ground truth.
dump_and_score seqmodel_on_seqtruth   "${SEQ_MODEL}"   "${SEQ_PREP}"
dump_and_score coordmodel_on_seqtruth "${COORD_MODEL}" "${SEQ_PREP}"
# REFERENCE: each model on its own native test split.
dump_and_score coordmodel_on_coordtruth "${COORD_MODEL}" "${COORD_PREP}"

echo "[b4] === comparison (deployment priors) ==="
python - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob("results/b4_quant.*.json")):
    d = json.load(f and open(f))
    tag = f.split("b4_quant.")[1].rsplit(".json", 1)[0]
    m = next(k for k in d if isinstance(d[k], dict) and "classification" in d[k])
    cls = d[m]["classification"]; fil = d["filter"][m]
    pc = cls["per_class"]
    rows.append((tag, cls["macro_f1"], pc["L1HS"]["f1"], pc["L1PA"]["f1"], pc["NEGATIVE"]["f1"],
                 fil["max_purity"], fil["frontier"].get("yield_at_p90")))
print(f"{'combo':<26} {'macroF1':>7} {'L1HS':>6} {'L1PA':>6} {'NEG':>6} {'maxPur':>7} {'yld@P90':>7}")
for t, mf, hs, pa, ng, mp, y90 in rows:
    y = "  — " if y90 is None or y90 != y90 else f"{y90:.3f}"
    print(f"{t:<26} {mf:>7.3f} {hs:>6.3f} {pa:>6.3f} {ng:>6.3f} {mp:>7.3f} {y:>7}")
PY
echo "[b4] done -> results/b4_quant.*.json"
