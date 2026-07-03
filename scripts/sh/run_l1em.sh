#!/bin/bash
# Baseline method: L1EM — EM quantification of L1 expression from the STAR BAM.
# Refactor of notebooks/scripts/run_l1em.sh: source-suffixed dataset dir, configurable
# external L1EM repo, resumable. Requires scripts/sh/align_star.sh to have run first.
#
#   L1EM_PATH=/path/to/L1EM bash scripts/sh/run_l1em.sh
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"; CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
GENOME_FA="${GENOME_FA:-data/external/GRCh38.p14.genome.fa}"   # decompressed by align_star.sh / the generator
L1EM_PATH="${L1EM_PATH:-L1EM}"                                 # external L1EM repo (kept unchanged)
L1EM_BED="${L1EM_BED:-data/ref/l1base/hsflil1_8438.bed}"       # reference resource lives under data/ref/
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

[[ -f "${L1EM_PATH}/run_L1EM.sh" ]] || { echo "[l1em] ERROR: L1EM repo not found (set L1EM_PATH=; looked in ${L1EM_PATH})" >&2; exit 1; }
for f in "${GENOME_FA}" "${L1EM_BED}"; do
    [[ -f "${f}" ]] || { echo "[l1em] ERROR: missing ${f}" >&2; exit 1; }
done
abs_l1em="$(realpath "${L1EM_PATH}")"; abs_genome="$(realpath "${GENOME_FA}")"; abs_bed="$(realpath "${L1EM_BED}")"

n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    bam="${OUTPUT_DIR}/star/${base}/Aligned.sortedByCoord.out.bam"
    out="${OUTPUT_DIR}/L1EM/${base}"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${out}/full_counts.txt" ]]; then
        echo "[l1em] ${base}: SKIP (counts exist)"; n_skip=$((n_skip + 1)); continue
    fi
    [[ -s "${bam}" ]] || { echo "[l1em] WARN: no BAM for ${base} (run align_star.sh) — skipping" >&2; continue; }
    mkdir -p "${out}"
    echo "[l1em] ${base}: L1EM"
    ( cd "${out}" && bash "${abs_l1em}/run_L1EM.sh" "$(realpath "${bam}")" "${abs_l1em}" \
        "${abs_genome}" "${abs_bed}" "$(pwd)" ) > "${out}/L1EM.out" 2> "${out}/L1EM.err" \
        || { echo "[l1em] ${base}: FAILED (see ${out}/L1EM.err)" >&2; exit 1; }
    rm -rf "${out}/split_fqs" "${out}/G_of_R" "${out}/idL1reads"
    n_done=$((n_done + 1))
  done
done
if [[ "${n_done}" -eq 0 && "${n_skip}" -eq 0 ]]; then
    echo "[l1em] ERROR: nothing processed — no input BAMs under ${OUTPUT_DIR}/star (run align_star.sh first?)." >&2
    echo "[l1em]   wrong dataset? for the background experiment pass SIM_MODEL=insert (→ .l1base.insert)." >&2
    exit 1
fi
echo "[l1em] done: ${n_done} quantified, ${n_skip} skipped → ${OUTPUT_DIR}/L1EM"
