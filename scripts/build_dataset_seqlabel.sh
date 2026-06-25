#!/bin/bash
# Track B2 — SEQUENCE-anchored build fork of build_dataset.sh, TWO ART streams so
# the class balance is controllable:
#   POSITIVES  ← ART(L1_CORPUS, POS_COV)  → STAR → strict overlap (-f STRICT_FRAC)
#                with the LINE-1 .out → keep reads that are >= STRICT_FRAC L1,
#                labelled by the annotation subfamily (rest DROP).
#   NEGATIVE   ← ART(NEG_INPUT, NEG_COV)  → STAR → no L1 overlap (-v) → NEGATIVE.
#
# Positive label = STRICT read-fraction overlap (a sequence-content proxy, since STAR
# aligned at high identity), NOT `pairtobed -type either` (any 1bp overlap). A final
# minimap2 CROSS-CHECK phase (relabel_by_sequence) re-derives the label from the read
# SEQUENCE and reports agreement (catches STAR-multimap artifacts).
#
# Additive: writes *seqlabel* outputs only (never clobbers option-A FASTQ).
#   module load GCCcore/13.3.0 STAR/2.7.11b SAMtools/1.21 BEDTools/2.31.1 minimap2/2.29
#   LINE1_OUT=data/external/GCF_000001405.40_GRCh38.p14_rm.LINE1.out.gz \
#     POS_COV=8 NEG_COV=2 bash scripts/build_dataset_seqlabel.sh
set -euo pipefail

# Pull the path vars (L1_CORPUS, GENCODE_FASTA, DATA_EXTERNAL, ...) from _common.sh
# when run standalone; a slurm wrapper (20_dataset-style) that already sourced it
# sets L1_CORPUS, so this is skipped there. _common.sh is source-safe (exports only).
_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/_common.sh"
[[ -z "${L1_CORPUS:-}" && -f "${_COMMON}" ]] && source "${_COMMON}"

POS_INPUT="${POS_INPUT:-${L1_CORPUS:?set L1_CORPUS or POS_INPUT (L1-overlapping transcripts)}}"
NEG_INPUT="${NEG_INPUT:-${GENCODE_FASTA:?set GENCODE_FASTA or NEG_INPUT (full transcriptome)}}"
REF_DIR="${REF_DIR:-${STAR_INDEX:-data/external/star_index}}"
LINE1_OUT="${LINE1_OUT:-${DATA_EXTERNAL:-data/external}/GCF_000001405.40_GRCh38.p14_rm.LINE1.out.gz}"
[[ -f "${LINE1_OUT}" ]] || { echo "[seqlabel] ERROR: LINE1_OUT not found: ${LINE1_OUT}" >&2; exit 1; }
WORK="${WORK:-data/external/dataset_build_seqlabel}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
STRICT_FRAC="${STRICT_FRAC:-0.5}"     # min fraction of the READ inside an L1 instance
MAX_DIV="${MAX_DIV:-100}"             # %div cap (e.g. 10 → young-only L1 positives)
POS_COV="${POS_COV:-8}"              # ART coverage for the (small) L1 corpus
NEG_COV="${NEG_COV:-2}"              # ART coverage for the (large) transcriptome
ART_LEN="${ART_LEN:-150}"; ART_FRAG_MEAN="${ART_FRAG_MEAN:-500}"
ART_FRAG_SD="${ART_FRAG_SD:-10}"; ART_SS="${ART_SS:-MSv3}"

L1_R1="${L1_R1:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R1.fq}"
L1_R2="${L1_R2:-data/external/l1hs_l1pa2_negative.seqlabel.5x_R2.fq}"
CROSSCHECK_LIB="${CROSSCHECK_LIB:-data/external/l1_subfamily_consensus.fa}"

mkdir -p "${WORK}"/{bed,pos,neg,fastq}

# --- shared L1 BED (built once) --------------------------------------------
echo "[seqlabel] RepeatMasker .out → L1 BED (max_div=${MAX_DIV})"
python -m trap.utils.rmout to-bed --rmout "${LINE1_OUT}" --out "${WORK}/bed/l1.bed" --max-div "${MAX_DIV}"
sort -k1,1 -k2,2n "${WORK}/bed/l1.bed" > "${WORK}/bed/l1.sorted.bed"
L1BED="${WORK}/bed/l1.sorted.bed"
: > "${WORK}/bed/empty.txt"

# _simulate_align <input_fasta> <cov> <subdir>  → echoes the BAM path
_simulate_align() {
    local in="$1" cov="$2" sub="$3" d="${WORK}/$3"
    mkdir -p "${d}"
    local fa="${in}"
    if [[ "${in}" == *.gz || "${in}" == *.bgz ]]; then
        fa="${d}/$(basename "${in%.gz}")"; fa="${fa%.bgz}"
        [[ -s "${fa}" ]] || gunzip -c "${in}" > "${fa}"
    fi
    echo "[seqlabel] (${sub}) ART -f ${cov} from $(basename "${fa}")" >&2
    art_illumina -sam -na -i "${fa}" -p -l "${ART_LEN}" -f "${cov}" \
        -m "${ART_FRAG_MEAN}" -s "${ART_FRAG_SD}" -ss "${ART_SS}" -o "${d}/reads_R" >&2
    echo "[seqlabel] (${sub}) STAR align" >&2
    STAR --runThreadN "${THREADS}" --genomeDir "${REF_DIR}" \
        --readFilesIn "${d}/reads_R1.fq" "${d}/reads_R2.fq" --readFilesCommand cat \
        --outSAMunmapped Within KeepPairs --outSAMtype BAM Unsorted \
        --winAnchorMultimapNmax 100 --outFilterMultimapNmax 100 --outFileNamePrefix "${d}/" >&2
    echo "${d}/Aligned.out.bam"
}

# _emit <bam> <subdir> <r1_out> <r2_out> <mode:pos|neg>
_emit() {
    local bam="$1" d="${WORK}/$2" r1o="$3" r2o="$4" mode="$5"
    local strict="${WORK}/bed/empty.txt" neg="${WORK}/bed/empty.txt"
    if [[ "${mode}" == "pos" ]]; then
        strict="${d}/strict_hits.tsv"
        bedtools intersect -abam "${bam}" -b "${L1BED}" -f "${STRICT_FRAC}" -bed -wb \
            | awk 'BEGIN{OFS="\t"} {print $4, $(NF-2)}' > "${strict}"
    else
        neg="${d}/negative_qnames.txt"
        bedtools intersect -abam "${bam}" -b "${L1BED}" -v -bed \
            | awk '{print $4}' | sort -u > "${neg}"
    fi
    python -m trap.utils.rmout label-fragments \
        --strict-hits "${strict}" --negative-qnames "${neg}" --out "${d}/qname_label.tsv"
    samtools collate -@"${THREADS}" -u -O "${bam}" \
        | samtools fastq -f 1 -F 268 -1 "${d}/all_R1.fq" -2 "${d}/all_R2.fq" -0 /dev/null -s /dev/null
    python -m trap.utils.rmout emit-labeled \
        --r1-in "${d}/all_R1.fq" --r2-in "${d}/all_R2.fq" \
        --labels "${d}/qname_label.tsv" --r1-out "${r1o}" --r2-out "${r2o}"
}

# --- stream 1: POSITIVES from L1_CORPUS ------------------------------------
POS_BAM="$(_simulate_align "${POS_INPUT}" "${POS_COV}" pos)"
_emit "${POS_BAM}" pos "${WORK}/fastq/pos_R1.fq" "${WORK}/fastq/pos_R2.fq" pos

# --- stream 2: NEGATIVE from the transcriptome -----------------------------
NEG_BAM="$(_simulate_align "${NEG_INPUT}" "${NEG_COV}" neg)"
_emit "${NEG_BAM}" neg "${WORK}/fastq/neg_R1.fq" "${WORK}/fastq/neg_R2.fq" neg

# --- concat (positives first, then NEGATIVE), R1/R2 in sync ----------------
cat "${WORK}/fastq/pos_R1.fq" "${WORK}/fastq/neg_R1.fq" > "${L1_R1}"
cat "${WORK}/fastq/pos_R2.fq" "${WORK}/fastq/neg_R2.fq" > "${L1_R2}"
echo "[seqlabel] R1 -> ${L1_R1}"; echo "[seqlabel] R2 -> ${L1_R2}"
echo "[seqlabel] class histogram:"
grep -hoE '\|[A-Za-z0-9]+$' "${L1_R1}" | sort | uniq -c | sort -rn

# --- minimap2 CROSS-CHECK (sequence label vs the .out label) ---------------
[[ -f "${CROSSCHECK_LIB}" ]] || CROSSCHECK_LIB="data/external/L19088.1.fa"
if command -v minimap2 >/dev/null && [[ -f "${CROSSCHECK_LIB}" ]]; then
    echo "[seqlabel] === minimap2 cross-check vs ${CROSSCHECK_LIB} ==="
    python -m trap.analysis.relabel_by_sequence run \
        --library "${CROSSCHECK_LIB}" --r1 "${L1_R1}" --r2 "${L1_R2}" \
        --max-reads "${CROSSCHECK_N:-50000}" --out "results/seqlabel_crosscheck.tsv" \
        || echo "[seqlabel] cross-check FAILED (FASTQ still written)"
else
    echo "[seqlabel] cross-check SKIPPED (need minimap2 + an L1 reference; load minimap2/2.29)"
fi
echo "[seqlabel] done."
