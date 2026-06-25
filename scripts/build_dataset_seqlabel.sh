#!/bin/bash
# Track B2 — SEQUENCE-anchored build fork of build_dataset.sh. Same ART→STAR, but
# the positive class is defined by STRICT read-fraction overlap with the LINE-1
# RepeatMasker .out (a read is L1 iff >= STRICT_FRAC of it lies in an L1 instance),
# instead of `pairtobed -type either` (any 1bp overlap). Subfamily comes from the
# annotation; reads with no L1 overlap → NEGATIVE; partial (0<overlap<frac) → DROP.
#
# Then a minimap2 CROSS-CHECK phase (relabel_by_sequence) independently re-derives
# the label by aligning the read SEQUENCE to an L1 consensus reference, and reports
# agreement with the .out labels — catching STAR-multimap artifacts.
#
# Additive: writes *seqlabel* outputs only (never clobbers option-A FASTQ).
#   module load GCCcore/13.3.0 STAR/2.7.11b SAMtools/1.21 BEDTools/2.31.1 minimap2/2.29
#   LINE1_OUT=data/external/GCF_000001405.40_GRCh38.p14_rm.LINE1.out.gz \
#     bash scripts/build_dataset_seqlabel.sh
set -euo pipefail

ART_INPUT="${ART_INPUT:-${L1_CORPUS:-${GENCODE_FASTA:?set GENCODE_FASTA/L1_CORPUS/ART_INPUT}}}"
REF_DIR="${REF_DIR:-data/external/star_index}"
LINE1_OUT="${LINE1_OUT:?set LINE1_OUT (GRCh38 LINE-1 RepeatMasker .out(.gz))}"
WORK="${WORK:-data/external/dataset_build_seqlabel}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
STRICT_FRAC="${STRICT_FRAC:-0.5}"      # min fraction of the READ inside an L1 instance
MAX_DIV="${MAX_DIV:-100}"              # %div cap (e.g. 10 → young-only L1 positives)
ART_COV="${ART_COV:-5}"; ART_LEN="${ART_LEN:-150}"
ART_FRAG_MEAN="${ART_FRAG_MEAN:-500}"; ART_FRAG_SD="${ART_FRAG_SD:-10}"; ART_SS="${ART_SS:-MSv3}"

# *seqlabel* outputs — never overwrite the option-A FASTQ.
L1_R1="${L1_R1:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R1.fq}"
L1_R2="${L1_R2:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R2.fq}"
CROSSCHECK_LIB="${CROSSCHECK_LIB:-data/external/l1_subfamily_consensus.fa}"  # falls back to L19088.1.fa

mkdir -p "${WORK}"/{art,star,bed,fastq}

# --- 1. ART simulate -------------------------------------------------------
_ART_INPUT="${ART_INPUT}"
if [[ "${ART_INPUT}" == *.gz || "${ART_INPUT}" == *.bgz ]]; then
    _ART_INPUT="${WORK}/art/$(basename "${ART_INPUT%.gz}")"; _ART_INPUT="${_ART_INPUT%.bgz}"
    [[ -s "${_ART_INPUT}" ]] || gunzip -c "${ART_INPUT}" > "${_ART_INPUT}"
fi
echo "[seqlabel] ART simulate (-f ${ART_COV}) from ${_ART_INPUT}"
art_illumina -sam -na -i "${_ART_INPUT}" -p -l "${ART_LEN}" -f "${ART_COV}" \
    -m "${ART_FRAG_MEAN}" -s "${ART_FRAG_SD}" -ss "${ART_SS}" -o "${WORK}/art/reads_R"

# --- 2. STAR align ---------------------------------------------------------
echo "[seqlabel] STAR align (multimap 100)"
STAR --runThreadN "${THREADS}" --genomeDir "${REF_DIR}" \
    --readFilesIn "${WORK}/art/reads_R1.fq" "${WORK}/art/reads_R2.fq" --readFilesCommand cat \
    --outSAMunmapped Within KeepPairs --outSAMtype BAM Unsorted \
    --winAnchorMultimapNmax 100 --outFilterMultimapNmax 100 --outFileNamePrefix "${WORK}/star/"
BAM="${WORK}/star/Aligned.out.bam"

# --- 3. LINE-1 .out → BED (optionally young-only via MAX_DIV) ---------------
echo "[seqlabel] RepeatMasker .out → L1 BED (max_div=${MAX_DIV})"
python -m trap.utils.rmout to-bed --rmout "${LINE1_OUT}" --out "${WORK}/bed/l1.bed" --max-div "${MAX_DIV}"
sort -k1,1 -k2,2n "${WORK}/bed/l1.bed" > "${WORK}/bed/l1.sorted.bed"

# --- 4. Strict (>=frac) + no-overlap (-v) intersects ------------------------
echo "[seqlabel] bedtools intersect: strict -f ${STRICT_FRAC} (+wb family) and -v (NEGATIVE)"
bedtools intersect -abam "${BAM}" -b "${WORK}/bed/l1.sorted.bed" -f "${STRICT_FRAC}" -bed -wb \
    | awk 'BEGIN{OFS="\t"} {print $4, $(NF-2)}' > "${WORK}/bed/strict_hits.tsv"
bedtools intersect -abam "${BAM}" -b "${WORK}/bed/l1.sorted.bed" -v -bed \
    | awk '{print $4}' | sort -u > "${WORK}/bed/negative_qnames.txt"

# --- 5. Resolve per-fragment labels ----------------------------------------
python -m trap.utils.rmout label-fragments \
    --strict-hits "${WORK}/bed/strict_hits.tsv" \
    --negative-qnames "${WORK}/bed/negative_qnames.txt" \
    --out "${WORK}/bed/qname_label.tsv"

# --- 6. Emit labelled paired FASTQ -----------------------------------------
echo "[seqlabel] samtools fastq (primary reads) → emit labelled pairs"
samtools collate -@"${THREADS}" -u -O "${BAM}" \
    | samtools fastq -f 1 -F 268 -1 "${WORK}/fastq/all_R1.fq" -2 "${WORK}/fastq/all_R2.fq" -0 /dev/null -s /dev/null
python -m trap.utils.rmout emit-labeled \
    --r1-in "${WORK}/fastq/all_R1.fq" --r2-in "${WORK}/fastq/all_R2.fq" \
    --labels "${WORK}/bed/qname_label.tsv" --r1-out "${L1_R1}" --r2-out "${L1_R2}"
echo "[seqlabel] R1 -> ${L1_R1}"; echo "[seqlabel] R2 -> ${L1_R2}"

# --- 7. minimap2 CROSS-CHECK (independent sequence label vs the .out label) --
[[ -f "${CROSSCHECK_LIB}" ]] || CROSSCHECK_LIB="data/external/L19088.1.fa"
if command -v minimap2 >/dev/null && [[ -f "${CROSSCHECK_LIB}" ]]; then
    echo "[seqlabel] === minimap2 cross-check vs ${CROSSCHECK_LIB} ==="
    # relabel_by_sequence reads the .out label from the read id (old) and aligns the
    # read to the L1 reference (new); the old-vs-new confusion = agreement.
    python -m trap.analysis.relabel_by_sequence run \
        --library "${CROSSCHECK_LIB}" --r1 "${L1_R1}" --r2 "${L1_R2}" \
        --max-reads "${CROSSCHECK_N:-50000}" \
        --out "results/seqlabel_crosscheck.tsv" \
        || echo "[seqlabel] cross-check FAILED (FASTQ still written)"
else
    echo "[seqlabel] cross-check SKIPPED (need minimap2 + an L1 reference; load minimap2/2.29)"
fi
echo "[seqlabel] done."
