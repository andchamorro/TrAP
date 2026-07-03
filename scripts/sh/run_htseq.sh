#!/bin/bash
# Baseline method: HTSeq (htseq-count) — union-mode feature counts from the STAR BAM.
# Refactor of the 4.08 HTSeq step (processing_htseq reads htseq_counts.csv).
# Requires scripts/sh/align_star.sh first.
#
# HTSEQ_GTF should annotate the L1 elements so counts join the per-element ground truth.
#
#   HTSEQ_GTF=/path/L1.gtf bash scripts/sh/run_htseq.sh
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"; CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
HTSEQ_GTF="${HTSEQ_GTF:-data/ref/l1base/hsflil1_8438.te.gtf}"
FEATURE_TYPE="${FEATURE_TYPE:-exon}"; ID_ATTR="${ID_ATTR:-gene_id}"; STRANDED="${STRANDED:-no}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

command -v htseq-count >/dev/null 2>&1 || { echo "[htseq] ERROR: htseq-count not on PATH" >&2; exit 1; }
[[ -f "${HTSEQ_GTF}" ]] || { echo "[htseq] ERROR: missing ${HTSEQ_GTF} (set HTSEQ_GTF)" >&2; exit 1; }

n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    bam="${OUTPUT_DIR}/star/${base}/Aligned.sortedByCoord.out.bam"
    out="${OUTPUT_DIR}/HTseq/${base}"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${out}/htseq_counts.csv" ]]; then
        echo "[htseq] ${base}: SKIP (counts exist)"; n_skip=$((n_skip + 1)); continue
    fi
    [[ -s "${bam}" ]] || { echo "[htseq] WARN: no BAM for ${base} (run align_star.sh) — skipping" >&2; continue; }
    mkdir -p "${out}"
    echo "[htseq] ${base}: htseq-count"
    htseq-count -f bam -r pos -s "${STRANDED}" -t "${FEATURE_TYPE}" -i "${ID_ATTR}" \
        "${bam}" "${HTSEQ_GTF}" > "${out}/htseq_counts.csv" 2> "${out}/htseq.err" \
        || { echo "[htseq] ${base}: FAILED (see ${out}/htseq.err)" >&2; exit 1; }
    n_done=$((n_done + 1))
  done
done
if [[ "${n_done}" -eq 0 && "${n_skip}" -eq 0 ]]; then
    echo "[htseq] ERROR: nothing processed — no input BAMs under ${OUTPUT_DIR}/star (run align_star.sh first?)." >&2
    echo "[htseq]   wrong dataset? for the background experiment pass SIM_MODEL=insert (→ .l1base.insert)." >&2
    exit 1
fi
echo "[htseq] done: ${n_done} quantified, ${n_skip} skipped → ${OUTPUT_DIR}/HTseq"
