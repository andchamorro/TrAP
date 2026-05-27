#!/bin/bash
# Build the labelled paired-end L1 FASTQ for the TrAP classification dataset
# (plan §6.2). Refactor of the old scripts/generate_dataset.sh: parameterised
# paths (no /path/to placeholders), ART coverage -f 5 (fix S1), STAR multimap
# 100 (S5), and the now-present scripts/divide_gff.py (S4).
#
# Pipeline:  ART (simulate) -> STAR (align) -> samtools (sort by name)
#            -> divide_gff.py (per-subfamily GFF) -> bedtools pairtobed (label)
#            -> samtools fastq -> concat positives + NEGATIVE -> L1_R1 / L1_R2
#
# Requires (load_bio_modules): art_illumina, STAR, samtools, bedtools, python.
#
# NOTE — least-certain stage. One thing to validate on real data before the
# full re-do:
#   1. scripts/divide_gff.py is a reconstruction (subfamily-name parsing).
# (Previously noted: _extract_transcript_id may need a strip of the ART
# read-number suffix.  Not needed — GENCODE headers use '|' as field
# separators, so split('|')[0] already returns a clean Ensembl transcript ID.)
set -euo pipefail

# --- Inputs (override via env) ---------------------------------------------
ART_INPUT="${ART_INPUT:-${GENCODE_FASTA:?set GENCODE_FASTA or ART_INPUT}}"
REF_DIR="${REF_DIR:-data/external/star_index}"
REPEATMASKER_GFF="${REPEATMASKER_GFF:?set REPEATMASKER_GFF (RepeatMasker LINE-1 GFF)}"
WORK="${WORK:-data/external/dataset_build}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

# ART parameters (config/datasets/l1hs_l1pa2_v48_k17.yaml)
ART_COV="${ART_COV:-5}"
ART_LEN="${ART_LEN:-150}"
ART_FRAG_MEAN="${ART_FRAG_MEAN:-500}"
ART_FRAG_SD="${ART_FRAG_SD:-10}"
ART_SS="${ART_SS:-MSv3}"

# Final outputs consumed by 20_dataset.slurm preprocessing
L1_R1="${L1_R1:-data/external/l1hs_l1pa2_negative.5x_R1.fq}"
L1_R2="${L1_R2:-data/external/l1hs_l1pa2_negative.5x_R2.fq}"

mkdir -p "${WORK}"/{art,star,gffs,bam,fastq}

# --- 1. Simulate paired-end reads (ART) ------------------------------------
# ART does not support gzip input; decompress to a plain FASTA first.
_ART_INPUT="${ART_INPUT}"
if [[ "${ART_INPUT}" == *.gz || "${ART_INPUT}" == *.bgz ]]; then
    _ART_INPUT="${WORK}/art/$(basename "${ART_INPUT%.gz}")"
    _ART_INPUT="${_ART_INPUT%.bgz}"
    if [[ ! -s "${_ART_INPUT}" ]]; then
        echo "[build] decompressing $(basename "${ART_INPUT}") for ART"
        gunzip -c "${ART_INPUT}" > "${_ART_INPUT}"
    fi
fi
echo "[build] ART simulate (-f ${ART_COV}) from ${_ART_INPUT}"
art_illumina -sam -na -i "${_ART_INPUT}" -p \
    -l "${ART_LEN}" -f "${ART_COV}" -m "${ART_FRAG_MEAN}" -s "${ART_FRAG_SD}" \
    -ss "${ART_SS}" -o "${WORK}/art/reads_R"

# --- 2. Align (STAR, multimap 100) -----------------------------------------
echo "[build] STAR align (multimap 100)"
# Default --readNameSeparator is '/', which trims the ART '/1','/2' mate suffix
# so both mates share one QNAME and STAR pairs them correctly. (The previous
# '--readNameSeparator space' kept the suffixes, giving the two mates different
# names and corrupting mate ids in samtools fastq downstream.)
STAR --runThreadN "${THREADS}" \
    --genomeDir "${REF_DIR}" \
    --readFilesIn "${WORK}/art/reads_R1.fq" "${WORK}/art/reads_R2.fq" \
    --readFilesCommand cat \
    --outSAMunmapped Within KeepPairs \
    --outSAMtype BAM Unsorted \
    --winAnchorMultimapNmax 100 \
    --outFilterMultimapNmax 100 \
    --outFileNamePrefix "${WORK}/star/"

echo "[build] samtools sort by name"
samtools sort -@"${THREADS}" -n -o "${WORK}/star/aligned_byname.bam" "${WORK}/star/Aligned.out.bam"

# --- 3. Split RepeatMasker GFF into per-subfamily files ---------------------
echo "[build] divide_gff (LINE-1 subfamilies)"
python "$(dirname "$0")/divide_gff.py" "${REPEATMASKER_GFF}" "${WORK}/gffs" --pattern '^L1'

# --- 4. Label reads per subfamily (bedtools pairtobed) ---------------------
# Number of concurrent bedtools+samtools jobs. Each job uses 1 bedtools thread
# + SAMTOOLS_THREADS_PARALLEL samtools threads. Default: floor(THREADS / 4),
# capped at 32 — keeps all cores busy without I/O saturation on Lustre.
PARALLEL_LABEL="${PARALLEL_LABEL:-$(( THREADS / 4 < 32 ? THREADS / 4 : 32 ))}"
SAMTOOLS_THREADS_PARALLEL="${SAMTOOLS_THREADS_PARALLEL:-2}"

: > "${L1_R1}"
: > "${L1_R2}"

# Each subfamily job writes to its own per-subfamily output files and never
# touches the shared L1_R1/L1_R2 outputs (avoids concurrent append races).
# After all parallel jobs complete, per-subfamily files are concatenated in
# deterministic GFF sort order to guarantee R1[i] == mate of R2[i].
_label_one() {
    local gff="$1"
    local target bam r1 r2
    target="$(basename "${gff}" .gff)"
    bam="${WORK}/bam/${target}.bam"
    r1="${WORK}/fastq/${target}_R1.fq"
    r2="${WORK}/fastq/${target}_R2.fq"
    bedtools pairtobed -type either \
        -abam "${WORK}/star/aligned_byname.bam" -b "${gff}" > "${bam}"
    # Low thread count per job: we are running N_PARALLEL in parallel.
    samtools collate -@"${SAMTOOLS_THREADS_PARALLEL}" -u -O "${bam}" \
        | samtools fastq -f 1 -F 268 \
            -1 "${r1}" -2 "${r2}" -0 /dev/null -s /dev/null
    # Write labeled reads to per-subfamily files (not the shared output).
    awk -v t="${target}" 'NR%4==1{print $0 "|" t; next} {print}' "${r1}" \
        > "${WORK}/fastq/${target}_R1.labeled.fq"
    awk -v t="${target}" 'NR%4==1{print $0 "|" t; next} {print}' "${r2}" \
        > "${WORK}/fastq/${target}_R2.labeled.fq"
    echo "[build] labelled subfamily ${target}"
}
export -f _label_one
export WORK SAMTOOLS_THREADS_PARALLEL

# Background-job pool — portable bash (no GNU parallel required).
_bg_pids=()
_bg_failed=0
_bg_throttle() {
    while (( ${#_bg_pids[@]} >= PARALLEL_LABEL )); do
        local new=() pid
        for pid in "${_bg_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                new+=("${pid}")
            else
                wait "${pid}" || (( ++_bg_failed )) || true
            fi
        done
        _bg_pids=("${new[@]}")
        # 'if' form avoids returning non-zero when condition is false,
        # which would trigger set -e on the caller side.
        if (( ${#_bg_pids[@]} >= PARALLEL_LABEL )); then sleep 0.2; fi
    done
    return 0
}
_bg_wait_all() {
    local pid
    for pid in "${_bg_pids[@]}"; do
        wait "${pid}" || (( ++_bg_failed ))
    done
    _bg_pids=()
}

# TODO: Is implementable using gpu parallel or xargs
echo "[build] labelling ${PARALLEL_LABEL} subfamilies in parallel (THREADS=${THREADS})"
for gff in "${WORK}"/gffs/*.gff; do
    [[ -e "$gff" ]] || { echo "[build] no subfamily GFFs found" >&2; exit 1; }
    _label_one "${gff}" &
    _bg_pids+=($!)
    _bg_throttle
done
_bg_wait_all
(( _bg_failed > 0 )) && { echo "[build] ERROR: ${_bg_failed} label job(s) failed" >&2; exit 1; }

# Deterministic in-order concatenation (matches the GFF sort order so R1/R2 are synced).
for gff in "${WORK}"/gffs/*.gff; do
    target="$(basename "${gff}" .gff)"
    cat "${WORK}/fastq/${target}_R1.labeled.fq" >> "${L1_R1}"
    cat "${WORK}/fastq/${target}_R2.labeled.fq" >> "${L1_R2}"
done

# --- 5. NEGATIVE: read pairs overlapping no LINE-1 subfamily ---------------
# Run sequentially after the parallel loop to avoid I/O contention with NEGATIVE
# (the largest BAM by far — 14M reads — it gets the full THREADS for collate).
echo "[build] NEGATIVE (no LINE-1 overlap)"
NEG_BAM="${WORK}/bam/NEGATIVE.bam"
_neg_r1="${WORK}/fastq/NEGATIVE_R1.fq"
_neg_r2="${WORK}/fastq/NEGATIVE_R2.fq"
bedtools pairtobed -type neither \
    -abam "${WORK}/star/aligned_byname.bam" -b "${REPEATMASKER_GFF}" > "${NEG_BAM}"
samtools collate -@"${THREADS}" -u -O "${NEG_BAM}" \
    | samtools fastq -f 1 -F 268 \
        -1 "${_neg_r1}" -2 "${_neg_r2}" -0 /dev/null -s /dev/null
awk -v t="NEGATIVE" 'NR%4==1{print $0 "|" t; next} {print}' "${_neg_r1}" >> "${L1_R1}"
awk -v t="NEGATIVE" 'NR%4==1{print $0 "|" t; next} {print}' "${_neg_r2}" >> "${L1_R2}"

echo "[build_dataset] done:"
echo "  R1 -> ${L1_R1}"
echo "  R2 -> ${L1_R2}"
