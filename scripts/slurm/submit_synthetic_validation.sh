#!/bin/bash
# Submit the synthetic e2e validation as a per-cell SLURM array (one grid cell per
# task: classify -> filter -> salmon). Sized to POWERS x DELPROBS. Each cell is
# independent (no shared prep), so there's no prep job — just the array.
#
#   bash scripts/slurm/submit_synthetic_validation.sh                # full grid
#   bash scripts/slurm/submit_synthetic_validation.sh --dry-run      # print the sbatch line
#   POWERS="8" DELPROBS="0.025 0.100" THROTTLE=2 bash scripts/slurm/submit_synthetic_validation.sh
#   L1_SOURCE=l1base TAU=0.5 bash scripts/slurm/submit_synthetic_validation.sh
#
# THROTTLE caps concurrent GPU tasks (default 4). Extra flags pass through to sbatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SLURM="scripts/slurm/62_synthetic_validation.slurm"

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
THROTTLE="${THROTTLE:-4}"

args=(--parsable --array="0-$((N - 1))%${THROTTLE}" --export=ALL "${passthru[@]+"${passthru[@]}"}")
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] sbatch ${args[*]} ${SLURM}"
else
    jobid="$(sbatch "${args[@]}" "${SLURM}")"
    echo "array job: ${jobid}"
fi
echo "grid: ${N} cells (POWERS='${POWERS}' × DELPROBS='${DELPROBS}'), --array=0-$((N - 1))%${THROTTLE}"
