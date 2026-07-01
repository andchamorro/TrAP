#!/bin/bash
# STAR-align the synthetic reads to the genome → one sorted BAM per grid cell.
# This BAM is the shared input for the alignment-based baselines (L1EM, TEtranscripts,
# HTseq, AlbertEM). Run this before those.
#
# PREP_ONLY=1 builds only the (heavy, ~30 GB RAM) STAR genome index, which the per-cell
# workers then reuse. Idempotent + resumable.
#
#   PREP_ONLY=1 bash scripts/sh/align_star.sh        # build the STAR index once
#   bash scripts/sh/align_star.sh                    # align the grid
set -euo pipefail

L1_SOURCE="${L1_SOURCE:-l1base}"; CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}}"
GENOME_FA="${GENOME_FA:-data/ref/GRCh38.p14.genome.fa}"
GENCODE_GTF="${GENCODE_GTF:-data/external/gencode.v48.annotation.gtf.gz}"
STAR_INDEX="${STAR_INDEX:-data/ref/star/GRCh38.p14.genome.StarIndex}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
ART_LEN="${ART_LEN:-150}"; THREADS="${THREADS:-8}"; SKIP_EXISTING="${SKIP_EXISTING:-1}"

for t in STAR samtools; do command -v "$t" >/dev/null 2>&1 || { echo "[star] ERROR: $t not on PATH" >&2; exit 1; }; done

# --- genome index (one-time, heavy) ----------------------------------------
if [[ ! -s "${STAR_INDEX}/SA" || "${FORCE_PREP:-0}" == "1" ]]; then
    [[ -s "${GENOME_FA}" ]] || { echo "[star] ERROR: missing genome ${GENOME_FA}" >&2; exit 1; }
    echo "[star] (prep) building STAR index → ${STAR_INDEX}"
    mkdir -p "${STAR_INDEX}"
    gtf_arg=()
    if [[ -f "${GENCODE_GTF}" ]]; then
        gtf="${GENCODE_GTF}"
        [[ "${gtf}" == *.gz ]] && { gtf="${STAR_INDEX}/annotation.gtf"; [[ -s "${gtf}" ]] || zcat -f "${GENCODE_GTF}" > "${gtf}"; }
        gtf_arg=(--sjdbGTFfile "${gtf}" --sjdbOverhang "$((ART_LEN - 1))")
    fi
    STAR --runMode genomeGenerate --genomeDir "${STAR_INDEX}" \
         --genomeFastaFiles "${GENOME_FA}" "${gtf_arg[@]}" --runThreadN "${THREADS}"
else
    echo "[star] (prep) STAR index exists → ${STAR_INDEX}"
fi

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
    echo "[star] PREP_ONLY — index ready; skipping alignment"; exit 0
fi

# --- align the grid --------------------------------------------------------
resolve_reads() {
    local base="$1" mate="$2"
    local plain="${OUTPUT_DIR}/art/${base}${mate}.fq" gz="${OUTPUT_DIR}/art/${base}${mate}.fq.gz"
    if [[ -s "${plain}" ]]; then printf '%s' "${plain}"
    elif [[ -s "${gz}" ]]; then printf '%s' "${gz}"; fi
}

n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    base="GRCh38.p14.${CHR}.insert_level_${power}_delprob_${dp}.pair.${FCOV}x"
    out="${OUTPUT_DIR}/star/${base}"
    bam="${out}/Aligned.sortedByCoord.out.bam"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${bam}" ]]; then
        echo "[star] ${base}: SKIP (BAM exists)"; n_skip=$((n_skip + 1)); continue
    fi
    r1="$(resolve_reads "${base}" 1)"; r2="$(resolve_reads "${base}" 2)"
    [[ -n "${r1}" && -n "${r2}" ]] || { echo "[star] WARN: missing reads for ${base} — skipping" >&2; continue; }
    mkdir -p "${out}"
    readcmd=(); [[ "${r1}" == *.gz ]] && readcmd=(--readFilesCommand zcat)
    echo "[star] ${base}: aligning"
    STAR --genomeDir "${STAR_INDEX}" --readFilesIn "${r1}" "${r2}" "${readcmd[@]}" \
         --outSAMtype BAM SortedByCoordinate --outFileNamePrefix "${out}/" \
         --outSAMprimaryFlag AllBestScore --outFilterMultimapNmax 1000 \
         --runThreadN "${THREADS}" > "${out}/STAR.out" 2> "${out}/STAR.err"
    samtools index "${bam}"
    n_done=$((n_done + 1))
  done
done
echo "[star] done: ${n_done} aligned, ${n_skip} skipped → ${OUTPUT_DIR}/star"
