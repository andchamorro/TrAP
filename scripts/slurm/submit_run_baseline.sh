#!/bin/bash
# Submit a baseline quantification method as a per-cell SLURM array, sized to POWERS x
# DELPROBS. METHOD=star chains a prep job (STAR index build, PREP_ONLY=1) → align array
# (afterok); other methods submit the array directly.
#
#   METHOD=salmon bash scripts/slurm/submit_run_baseline.sh
#   METHOD=star   bash scripts/slurm/submit_run_baseline.sh --dry-run
#   METHOD=l1em L1EM_PATH=/sw/L1EM THROTTLE=4 bash scripts/slurm/submit_run_baseline.sh
#
# THROTTLE caps concurrent tasks (default 6). Extra flags pass through to sbatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SLURM="scripts/slurm/63_run_baseline.slurm"
METHOD="${METHOD:?set METHOD=salmon|star|l1em|tetranscripts|htseq|albertem}"
export METHOD

DRY_RUN=0; passthru=()
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
THROTTLE="${THROTTLE:-6}"

# Per-method resource profile — salmon/htseq are tiny (observed: salmon ~10 s CPU, <5 MB);
# STAR/L1EM are heavy. sbatch CLI overrides the static #SBATCH in 63_run_baseline.slurm.
# Override any with CPUS=/MEM=/TIME=.
d_prep_mem=""   # star: the index build (prep) needs more RAM than the per-cell align tasks
case "${METHOD}" in
    salmon)        d_cpus=2; d_mem=4G;  d_time=00:20:00 ;;
    htseq)         d_cpus=2; d_mem=8G;  d_time=01:00:00 ;;
    tetranscripts) d_cpus=4; d_mem=16G; d_time=02:00:00 ;;
    star)          d_cpus=8; d_mem=40G; d_time=04:00:00; d_prep_mem=96G ;;  # index build OOM'd at 40G
    l1em|albertem) d_cpus=8; d_mem=32G; d_time=06:00:00 ;;
    *)             d_cpus=4; d_mem=16G; d_time=04:00:00 ;;
esac
CPUS="${CPUS:-$d_cpus}"; MEM="${MEM:-$d_mem}"; TIME="${TIME:-$d_time}"
PREP_MEM="${PREP_MEM:-${d_prep_mem:-$MEM}}"
export THREADS="${THREADS:-$CPUS}"   # method scripts thread to the allocation
res=(--cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}")
prep_res=(--cpus-per-task="${CPUS}" --mem="${PREP_MEM}" --time="${TIME}")

submit() {  # echoes jobid (or the dry-run line)
    if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] sbatch $* ${SLURM}"; else sbatch "$@" "${SLURM}"; fi
}

dep=""
if [[ "${METHOD}" == "star" ]]; then
    # prep job: build the STAR index once (single task), before the align array.
    prep="$(submit "${prep_res[@]}" --parsable --job-name=trap_baseline_prep --array=0 --export=ALL,PREP_ONLY=1 "${passthru[@]+"${passthru[@]}"}")"
    echo "prep (STAR index): ${prep}"
    [[ "${DRY_RUN}" == "1" ]] && prep="<prep_jobid>"
    dep="--dependency=afterok:${prep}"
fi

arr_args=("${res[@]}" --parsable --array="0-$((N - 1))%${THROTTLE}" --export=ALL,PREP_ONLY=0 "${passthru[@]+"${passthru[@]}"}")
[[ -n "${dep}" ]] && arr_args=("${dep}" "${arr_args[@]}")
jobid="$(submit "${arr_args[@]}")"
echo "array job (${METHOD}): ${jobid}"
echo "grid: ${N} cells (POWERS='${POWERS}' × DELPROBS='${DELPROBS}'), --array=0-$((N - 1))%${THROTTLE}, ${CPUS} cpu / ${MEM} / ${TIME}"
