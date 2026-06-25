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
# Remap the .out RefSeq chroms (NC_*) to the STAR genome's chr* via the NCBI report
# (auto-used if present). Without it, a chrom-name mismatch is caught by the check below.
CHROM_MAP="${CHROM_MAP:-${DATA_EXTERNAL:-data/external}/GCF_000001405.40_GRCh38.p14_assembly_report.txt}"
CHROM_MAP_ARG=(); [[ -f "${CHROM_MAP}" ]] && CHROM_MAP_ARG=(--chrom-map "${CHROM_MAP}")
WORK="${WORK:-data/external/dataset_build_seqlabel}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
STRICT_FRAC="${STRICT_FRAC:-0.5}"     # min fraction of the READ inside an L1 instance
MAX_DIV="${MAX_DIV:-100}"             # %div cap (e.g. 10 → young-only L1 positives)
POS_COV="${POS_COV:-8}"              # ART coverage for the (small) L1 corpus
NEG_COV="${NEG_COV:-2}"              # ART coverage for the (large) transcriptome
ART_LEN="${ART_LEN:-150}"; ART_FRAG_MEAN="${ART_FRAG_MEAN:-500}"
ART_FRAG_SD="${ART_FRAG_SD:-10}"; ART_SS="${ART_SS:-MSv3}"

# NB: read OUTPUT paths from SEQ_R1/SEQ_R2 (NOT L1_R1/L1_R2 — _common.sh exports those
# to the option-A FASTQ, so a ${L1_R1:-default} would clobber the coordinate-label
# benchmark). Force *seqlabel* names and refuse anything else.
SEQ_R1="${SEQ_R1:-${DATA_EXTERNAL:-data/external}/l1hs_l1pa2_negative.seqlabel.5x_R1.fq}"
SEQ_R2="${SEQ_R2:-${DATA_EXTERNAL:-data/external}/l1hs_l1pa2_negative.seqlabel.5x_R2.fq}"
case "${SEQ_R1}${SEQ_R2}" in
    *seqlabel*) : ;;
    *) echo "[seqlabel] ERROR: refusing non-*seqlabel* output (${SEQ_R1}) to protect option-A FASTQ" >&2; exit 1 ;;
esac
CROSSCHECK_LIB="${CROSSCHECK_LIB:-data/external/l1_subfamily_consensus.fa}"

mkdir -p "${WORK}"/{bed,pos,neg,fastq}

# --- shared L1 BED (built once) --------------------------------------------
echo "[seqlabel] RepeatMasker .out → L1 BED (max_div=${MAX_DIV})"
python -m trap.utils.rmout to-bed --rmout "${LINE1_OUT}" --out "${WORK}/bed/l1.bed" --max-div "${MAX_DIV}" "${CHROM_MAP_ARG[@]}"
sort -k1,1 -k2,2n "${WORK}/bed/l1.bed" > "${WORK}/bed/l1.sorted.bed"
L1BED="${WORK}/bed/l1.sorted.bed"
: > "${WORK}/bed/empty.txt"

# Fail FAST (before the ~hours of ART+STAR) if the L1 BED chrom names don't match
# the STAR genome — the RepeatMasker .out is often RefSeq (NC_000006.12) while the
# index is UCSC (chr6); a mismatch → zero overlaps → garbage all-NEGATIVE labels.
if [[ -f "${REF_DIR}/chrName.txt" ]]; then
    _common_chr=$(comm -12 <(cut -f1 "${L1BED}" | sort -u) <(sort -u "${REF_DIR}/chrName.txt") | wc -l)
    if [[ "${_common_chr}" -eq 0 ]]; then
        echo "[seqlabel] ERROR: L1 BED chrom names do not match the STAR genome (${REF_DIR})." >&2
        echo "  BED chroms:  $(cut -f1 "${L1BED}" | sort -u | head -3 | tr '\n' ' ')" >&2
        echo "  STAR chroms: $(head -3 "${REF_DIR}/chrName.txt" | tr '\n' ' ')" >&2
        echo "  → remap: rmout to-bed --chrom-map <GCF_..._assembly_report.txt>, or use the chr*-named GFF." >&2
        exit 1
    fi
    echo "[seqlabel] chrom check OK (${_common_chr} shared names with the STAR genome)"
fi

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
            | awk 'BEGIN{OFS="\t"} {q=$4; sub(/\/[12]$/,"",q); print q, $(NF-2)}' > "${strict}"
    else
        neg="${d}/negative_qnames.txt"
        bedtools intersect -abam "${bam}" -b "${L1BED}" -v -bed \
            | awk '{q=$4; sub(/\/[12]$/,"",q); print q}' | sort -u > "${neg}"
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
cat "${WORK}/fastq/pos_R1.fq" "${WORK}/fastq/neg_R1.fq" > "${SEQ_R1}"
cat "${WORK}/fastq/pos_R2.fq" "${WORK}/fastq/neg_R2.fq" > "${SEQ_R2}"
echo "[seqlabel] R1 -> ${SEQ_R1}"; echo "[seqlabel] R2 -> ${SEQ_R2}"
echo "[seqlabel] class histogram:"
grep -hoE '\|[A-Za-z0-9]+$' "${SEQ_R1}" | sort | uniq -c | sort -rn || echo "  (no labelled reads — check the chrom/QNAME steps above)"

# --- minimap2 CROSS-CHECK (sequence label vs the .out label) ---------------
[[ -f "${CROSSCHECK_LIB}" ]] || CROSSCHECK_LIB="data/external/L19088.1.fa"
if command -v minimap2 >/dev/null && [[ -f "${CROSSCHECK_LIB}" ]]; then
    echo "[seqlabel] === minimap2 cross-check vs ${CROSSCHECK_LIB} ==="
    python -m trap.analysis.relabel_by_sequence run \
        --library "${CROSSCHECK_LIB}" --r1 "${SEQ_R1}" --r2 "${SEQ_R2}" \
        --max-reads "${CROSSCHECK_N:-50000}" --out "results/seqlabel_crosscheck.tsv" \
        || echo "[seqlabel] cross-check FAILED (FASTQ still written)"
else
    echo "[seqlabel] cross-check SKIPPED (need minimap2 + an L1 reference; load minimap2/2.29)"
fi
echo "[seqlabel] done."
