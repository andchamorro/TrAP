#!/bin/bash
# Baseline method: Salmon (no classifier filter) — salmon quant on the RAW synthetic
# reads against the L1 index, for every grid cell. This is the "Salmon" column of the
# abundance comparison (contrast: "AlbertSalmon" = classifier-filtered → salmon).
#
# Refactor of notebooks/scripts/run_salmon.sh: source-suffixed dataset dir, the
# generator's l1_synthetic.Index (quant targets == ground-truth elements), --sketch
# (selective alignment panics on the short L1 refs), resumable, no update_tpm noise hack.
#
#   bash scripts/sh/run_salmon.sh                          # full grid
#   POWERS="8" DELPROBS="0.025" bash scripts/sh/run_salmon.sh
#   L1_SOURCE=rm bash scripts/sh/run_salmon.sh
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"
CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
L1_INDEX="${L1_INDEX:-${OUTPUT_DIR}/l1_synthetic.Index}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
THREADS="${THREADS:-8}"
SALMON_FLAGS="${SALMON_FLAGS:---sketch}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

command -v salmon >/dev/null 2>&1 || { echo "[salmon] ERROR: salmon not on PATH" >&2; exit 1; }
[[ -f "${L1_INDEX}/info.json" ]] || { echo "[salmon] ERROR: missing index ${L1_INDEX} (run generate_synthetic_dataset.sh)" >&2; exit 1; }

# Resolve a mate's reads, preferring uncompressed .fq.
resolve_reads() {
    local base="$1" mate="$2"
    local plain="${OUTPUT_DIR}/art/${base}${mate}.fq" gz="${OUTPUT_DIR}/art/${base}${mate}.fq.gz"
    if [[ -s "${plain}" ]]; then printf '%s' "${plain}"
    elif [[ -s "${gz}" ]]; then printf '%s' "${gz}"; fi
}

outroot="${OUTPUT_DIR}/salmon/unfiltered"; mkdir -p "${outroot}"
n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    out="${outroot}/${base}"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${out}/quant.sf" ]]; then
        echo "[salmon] ${base}: SKIP (quant.sf exists)"; n_skip=$((n_skip + 1)); continue
    fi
    r1="$(resolve_reads "${base}" 1)"; r2="$(resolve_reads "${base}" 2)"
    [[ -n "${r1}" && -n "${r2}" ]] || { echo "[salmon] WARN: missing reads for ${base} — skipping" >&2; continue; }
    mkdir -p "${out}"
    echo "[salmon] ${base}: salmon quant (${SALMON_FLAGS})"
    salmon quant -q -i "${L1_INDEX}" -l A -1 "${r1}" -2 "${r2}" ${SALMON_FLAGS} \
        -o "${out}" --threads "${THREADS}" > "${out}/salmon.out" 2> "${out}/salmon.err" \
        || { echo "[salmon] ${base}: FAILED (see ${out}/salmon.err)" >&2; exit 1; }
    n_done=$((n_done + 1))
  done
done
echo "[salmon] done: ${n_done} quantified, ${n_skip} skipped → ${outroot}"
