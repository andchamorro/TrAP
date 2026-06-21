#!/bin/bash
#SBATCH --job-name=trap_spm_gate2_mlm_learnability
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
# Gate 2 — does the raw-read SPM tokenisation carry learnable MLM signal?
#
# The Salmon tokenizer feature-hashes k-mers through a non-invertible avalanche,
# so a masked token is unpredictable from its neighbours and MLM loss collapses
# to H(unigram) (see scripts/slurm/legacy/mlm/README.md). Content-preserving SPM
# pieces keep neighbour sequence, so masked-piece prediction SHOULD beat that
# floor. This runs the two-check probe (input-dependence + overfit-capacity) on
# the real SPM tokenizer + GENCODE corpus, twice: random-mask and fixed-mask.
#
# Reads the SPM tokenizer + GENCODE paths from config/runs/spm.yaml via _common.sh
# (TOKENIZER_PATH, GENCODE_FASTA, ALBERT_CONFIG). Defaults below make it runnable
# with no env, but each is overridable:
#   sbatch scripts/slurm/spm_gate2_mlm_learnability.slurm
#   N_SEQS=32 STEPS=600 sbatch scripts/slurm/spm_gate2_mlm_learnability.slurm
#
# PASS  = final loss drops well below the unigram floor (ln #unique masked tokens)
#         -> MLM is learnable on this tokenisation; an SPM MLM pre-train is worth it.
# FAIL  = loss stalls at the floor -> no context signal (back to Track A).

# Set before sourcing _common.sh: the run-config guard checks TRAP_SPM_EXPERIMENTAL
# at source time, and _load_run_config needs RUN_CONFIG.
export TRAP_SPM_EXPERIMENTAL="${TRAP_SPM_EXPERIMENTAL:-1}"
export RUN_CONFIG="${RUN_CONFIG:-config/runs/spm.yaml}"
# Gate-2 output is logger.info; TRAP_VERBOSITY defaults to 'off', which silently
# drops it (the bug that hid the original MLM diagnostics). Force it visible.
export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap

N_DATA="${N_DATA:-2000}"
N_SEQS="${N_SEQS:-16}"
STEPS="${STEPS:-400}"
LR="${LR:-1e-3}"

echo "[gate2] tokenizer=${TOKENIZER_PATH} corpus=${GENCODE_FASTA} albert=${ALBERT_CONFIG}"
echo "[gate2] n_data=${N_DATA} n_seqs=${N_SEQS} steps=${STEPS} lr=${LR}"

_probe() {
    python "${REPO_ROOT}/scripts/diagnose_mlm_learnability.py" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --corpus "${GENCODE_FASTA}" --file-format fasta \
        --albert-config-path "${ALBERT_CONFIG}" \
        --n-data "${N_DATA}" --n-seqs "${N_SEQS}" --steps "${STEPS}" --lr "${LR}" "$@"
}

echo "[gate2] === random-mask probe ==="
_probe

echo "[gate2] === fixed-mask probe (memorisation; isolates optimisation from mask draw) ==="
_probe --fixed-mask

echo "[gate2] done — read the two [verdict] lines above."
