#!/bin/bash
# Baseline method: AlbertEM — the seqlabel classifier filter followed by the EM algorithm
# (ML-L1EM). Refactor of notebooks/scripts/draft/run_mlem_mut.sh (run_MLL1EM.sh).
# Requires: align_star.sh (the BAM) and the classifier-filtered reads that
# synthetic_validation.slurm persists to <OUTPUT_DIR>/filtered_seqlabel/<base>/{1,2}.fq[.gz].
#
#   L1EM_PATH=/path/to/L1EM bash scripts/sh/run_albertem.sh
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"; CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
GENOME_FA="${GENOME_FA:-data/external/GRCh38.p14.genome.fa}"   # decompressed by align_star.sh / the generator
L1EM_PATH="${L1EM_PATH:-L1EM}"                                 # external L1EM repo (kept unchanged)
L1EM_BED="${L1EM_BED:-data/ref/l1base/hsflil1_8438.bed}"       # reference resource lives under data/ref/
FILTERED_DIR="${FILTERED_DIR:-${OUTPUT_DIR}/filtered_seqlabel}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

[[ -f "${L1EM_PATH}/run_MLL1EM.sh" ]] || { echo "[albertem] ERROR: run_MLL1EM.sh not in L1EM repo (set L1EM_PATH=; looked in ${L1EM_PATH})" >&2; exit 1; }
for f in "${GENOME_FA}" "${L1EM_BED}"; do
    [[ -f "${f}" ]] || { echo "[albertem] ERROR: missing ${f}" >&2; exit 1; }
done
abs_l1em="$(realpath "${L1EM_PATH}")"; abs_genome="$(realpath "${GENOME_FA}")"; abs_bed="$(realpath "${L1EM_BED}")"

# Copy a persisted (possibly gzipped) filtered mate into the run dir as plain .fq.
stage_reads() {
    local src_base="$1" dst="$2"
    if [[ -s "${src_base}.fq" ]]; then cp "${src_base}.fq" "${dst}"
    elif [[ -s "${src_base}.fq.gz" ]]; then gzip -dc "${src_base}.fq.gz" > "${dst}"
    else return 1; fi
}

n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    bam="${OUTPUT_DIR}/star/${base}/Aligned.sortedByCoord.out.bam"
    out="${OUTPUT_DIR}/MLEM/${base}"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${out}/full_counts.txt" ]]; then
        echo "[albertem] ${base}: SKIP (counts exist)"; n_skip=$((n_skip + 1)); continue
    fi
    [[ -s "${bam}" ]] || { echo "[albertem] WARN: no BAM for ${base} (run align_star.sh) — skipping" >&2; continue; }
    mkdir -p "${out}/filtered"
    if ! stage_reads "${FILTERED_DIR}/${base}/1" "${out}/filtered/filtered_r1.fq" || \
       ! stage_reads "${FILTERED_DIR}/${base}/2" "${out}/filtered/filtered_r2.fq"; then
        echo "[albertem] WARN: no filtered reads for ${base} in ${FILTERED_DIR} (run synthetic_validation) — skipping" >&2
        continue
    fi
    echo "[albertem] ${base}: ML-L1EM"
    ( cd "${out}" && bash "${abs_l1em}/run_MLL1EM.sh" "$(realpath "${bam}")" "${abs_l1em}" \
        "${abs_genome}" "${abs_bed}" "$(pwd)" ) > "${out}/MLEM.out" 2> "${out}/MLEM.err" \
        || { echo "[albertem] ${base}: FAILED (see ${out}/MLEM.err)" >&2; exit 1; }
    rm -rf "${out}/split_fqs" "${out}/G_of_R" "${out}/idL1reads"
    n_done=$((n_done + 1))
  done
done
if [[ "${n_done}" -eq 0 && "${n_skip}" -eq 0 ]]; then
    echo "[albertem] ERROR: nothing processed — no input BAMs / filtered reads under ${OUTPUT_DIR}." >&2
    echo "[albertem]   wrong dataset? for the background experiment pass SIM_MODEL=insert (→ .l1base.insert)." >&2
    exit 1
fi
echo "[albertem] done: ${n_done} quantified, ${n_skip} skipped → ${OUTPUT_DIR}/MLEM"
