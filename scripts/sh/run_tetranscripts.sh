#!/bin/bash
# Baseline method: TEtranscripts (TEcount) — EM-based TE/L1 counts from the STAR BAM.
# Refactor of the 4.08 TEtranscripts step (processing_tecount reads <project>.cntTable).
# Requires scripts/sh/align_star.sh first.
#
# TE_GTF must annotate the L1 elements as features so counts join the per-element ground
# truth (the legacy used a hsflil1 UID-based TE GTF; processing_tecount keeps 'UID-*').
#
#   TE_GTF=/path/L1.te.gtf GENE_GTF=/path/genes.gtf bash scripts/sh/run_tetranscripts.sh
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"; CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}}"
GENE_GTF="${GENE_GTF:-data/external/gencode.v48.annotation.gtf}"
TE_GTF="${TE_GTF:-data/ref/l1base/hsflil1_8438.te.gtf}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
TE_MODE="${TE_MODE:-multi}"; SKIP_EXISTING="${SKIP_EXISTING:-1}"

command -v TEcount >/dev/null 2>&1 || { echo "[te] ERROR: TEcount not on PATH (TEtranscripts)" >&2; exit 1; }
for f in "${GENE_GTF}" "${TE_GTF}"; do
    [[ -f "${f}" ]] || { echo "[te] ERROR: missing ${f} (set GENE_GTF/TE_GTF)" >&2; exit 1; }
done

n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    project="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}"
    bam="${OUTPUT_DIR}/star/${base}/Aligned.sortedByCoord.out.bam"
    out="${OUTPUT_DIR}/TEtranscripts/${base}"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${out}/${project}.cntTable" ]]; then
        echo "[te] ${base}: SKIP (cntTable exists)"; n_skip=$((n_skip + 1)); continue
    fi
    [[ -s "${bam}" ]] || { echo "[te] WARN: no BAM for ${base} (run align_star.sh) — skipping" >&2; continue; }
    mkdir -p "${out}"
    echo "[te] ${base}: TEcount (${TE_MODE})"
    TEcount --sortByPos -b "${bam}" --GTF "${GENE_GTF}" --TE "${TE_GTF}" \
        --mode "${TE_MODE}" --project "${project}" --outdir "${out}" \
        > "${out}/TEcount.out" 2> "${out}/TEcount.err" \
        || { echo "[te] ${base}: FAILED (see ${out}/TEcount.err)" >&2; exit 1; }
    n_done=$((n_done + 1))
  done
done
echo "[te] done: ${n_done} quantified, ${n_skip} skipped → ${OUTPUT_DIR}/TEtranscripts"
