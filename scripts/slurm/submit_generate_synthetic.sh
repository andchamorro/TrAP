#!/bin/bash
# Submit the synthetic-dataset generation as a prep job → per-cell ARRAY (afterok).
# The prep job builds the shared inputs once (chr1 transcripts, full-length L1 elements,
# salmon index); the array then generates one grid cell (insert L1 → ART) per task,
# sized to POWERS × DELPROBS. Resumable (cells with existing reads are skipped).
#
#   bash scripts/slurm/submit_generate_synthetic.sh                 # full grid
#   bash scripts/slurm/submit_generate_synthetic.sh --dry-run       # print the chain only
#   POWERS="8" DELPROBS="0.025 0.100" THROTTLE=4 bash scripts/slurm/submit_generate_synthetic.sh
#
# Any extra flags are passed through to every sbatch call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SLURM="scripts/slurm/generate_synthetic.slurm"

DRY_RUN=0
passthru=()
for a in "$@"; do
    case "${a}" in
        --dry-run) DRY_RUN=1 ;;
        *) passthru+=("${a}") ;;
    esac
done

export POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
export DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
read -ra _P <<< "${POWERS}"; read -ra _D <<< "${DELPROBS}"
N=$(( ${#_P[@]} * ${#_D[@]} ))
THROTTLE="${THROTTLE:-10}"

# 1. prep — build shared inputs once (PREP_ONLY exits before the grid).
prep_args=(--parsable --job-name=trap_gensynth_prep --array=0
           --export=ALL,PREP_ONLY=1 "${passthru[@]+"${passthru[@]}"}")
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] sbatch ${prep_args[*]} ${SLURM}"
    prep="<prep_jobid>"
else
    prep="$(sbatch "${prep_args[@]}" "${SLURM}")"
fi
echo "prep job: ${prep}"

# 2. array — one cell per task, afterok the prep.
arr_args=(--parsable --dependency="afterok:${prep}" --array="0-$((N - 1))%${THROTTLE}"
          --export=ALL "${passthru[@]+"${passthru[@]}"}")
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] sbatch ${arr_args[*]} ${SLURM}"
else
    arr="$(sbatch "${arr_args[@]}" "${SLURM}")"
    echo "array job: ${arr}"
fi
echo "grid: ${N} cells (POWERS='${POWERS}' × DELPROBS='${DELPROBS}'), --array=0-$((N - 1))%${THROTTLE} afterok:${prep}"
