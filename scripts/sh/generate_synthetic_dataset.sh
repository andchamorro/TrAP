#!/bin/bash
# Generate the synthetic L1-insertion abundance benchmark (chr1) from data/external.
# Deterministic re-creation of the notebook-4.07 dataset, adapted to data/external:
#   reference  = chr1 GENCODE transcripts        (insert L1 into these)
#   L1 inserts = full-length (~6 kb) L1 elements  (getfasta at the LINE1.promoter.bed coords)
#   grid       = insertion level 2^5..2^13  ×  del_prob {0.000,0.025,0.050,0.075,0.100}
#   per cell   = simulate insertions (seeded) → ART paired reads
# Also builds a salmon index from the L1 elements (for the downstream quant).
#
# Heavy (ART × 45 cells) → run on a COMPUTE NODE with bio tools:
#   art_illumina, seqkit, bedtools, samtools, salmon, python(BioPython).
#   source scripts/slurm/_common.sh && load_bio_modules && activate_trap   # salmon is in the env
#   module load SeqKit/2.9.0  # (or have seqkit on PATH)
#   bash scripts/sh/generate_synthetic_dataset.sh
#
# NB on reuse: the legacy generator set no seed, so this is a NEW (seeded, reproducible)
# draw — compare methods at the R²-vs-Simulated level (stable across draws), not read-for-read.
set -euo pipefail

_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/_common.sh"
[[ -z "${DATA_EXTERNAL:-}" && -f "${_COMMON}" ]] && source "${_COMMON}"
DATA_EXTERNAL="${DATA_EXTERNAL:-data/external}"

# --- inputs (data/external) ------------------------------------------------
GENOME="${GENOME:-${DATA_EXTERNAL}/GRCh38.p14.genome.fa.gz}"
PROMOTER_BED="${PROMOTER_BED:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_rm.LINE1.promoter.bed}"
GENCODE_FASTA="${GENCODE_FASTA:-${DATA_EXTERNAL}/gencode.v48.transcripts.fa.gz}"
GENCODE_GTF="${GENCODE_GTF:-${DATA_EXTERNAL}/gencode.v48.annotation.gtf.gz}"
CHR1_TRANSCRIPTS="${CHR1_TRANSCRIPTS:-}"   # optional: skip GTF subset if provided
CHR="${CHR:-chr1}"
# Full-length L1 filter (bp): keep ~6 kb intact elements.
L1_MIN_LEN="${L1_MIN_LEN:-5500}"; L1_MAX_LEN="${L1_MAX_LEN:-6500}"

# --- ART + grid + output ---------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel}"
FCOV="${FCOV:-5}"; ART_LEN="${ART_LEN:-150}"; ART_MFLEN="${ART_MFLEN:-500}"
ART_SDEV="${ART_SDEV:-10}"; ART_SS="${ART_SS:-MSv3}"
SEED="${SEED:-3469}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
PYDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${OUTPUT_DIR}/_inputs"
mkdir -p "${OUTPUT_DIR}/art" "${WORK}"

for f in "${GENOME}" "${PROMOTER_BED}"; do
    [[ -f "${f}" ]] || { echo "[gen] ERROR: missing input ${f}" >&2; exit 1; }
done

# --- 0. genome FASTA (uncompressed + .fai for getfasta) --------------------
GENOME_FA="${GENOME%.gz}"
if [[ "${GENOME}" == *.gz && ! -s "${GENOME_FA}" ]]; then
    echo "[gen] decompressing genome for getfasta"; gunzip -kc "${GENOME}" > "${GENOME_FA}"
fi
[[ -f "${GENOME_FA}.fai" ]] || samtools faidx "${GENOME_FA}"

# --- 1. full-length L1 elements (insert + salmon ref) ----------------------
echo "[gen] (1) full-length L1 elements from ${PROMOTER_BED} (${L1_MIN_LEN}-${L1_MAX_LEN} bp)"
awk -v lo="${L1_MIN_LEN}" -v hi="${L1_MAX_LEN}" 'BEGIN{OFS="\t"}
     ($3-$2)>=lo && ($3-$2)<=hi {print}' "${PROMOTER_BED}" > "${WORK}/l1_fulllength.bed"
L1_FASTA="${WORK}/l1_fulllength.fa"
bedtools getfasta -nameOnly -s -fi "${GENOME_FA}" -bed "${WORK}/l1_fulllength.bed" \
    | sed '/^>/ s/(.)$//' > "${L1_FASTA}"
echo "[gen]     $(grep -c '^>' "${L1_FASTA}") full-length L1 elements"

# --- 2. salmon index from the L1 elements ----------------------------------
L1_INDEX="${OUTPUT_DIR}/l1_synthetic.Index"
echo "[gen] (2) salmon index → ${L1_INDEX}"
salmon index -t "${L1_FASTA}" -i "${L1_INDEX}" -k 31 2>/dev/null \
    || echo "[gen]   WARNING: salmon index failed (salmon on PATH?) — build it before quant"

# --- 3. chr1 transcripts (insertion reference) -----------------------------
if [[ -z "${CHR1_TRANSCRIPTS}" ]]; then
    CHR1_TRANSCRIPTS="${WORK}/${CHR}_transcripts.fa"
    if [[ ! -s "${CHR1_TRANSCRIPTS}" ]]; then
        [[ -f "${GENCODE_GTF}" ]] || { echo "[gen] ERROR: need GENCODE_GTF to subset ${CHR} transcripts, or pass CHR1_TRANSCRIPTS=" >&2; exit 1; }
        echo "[gen] (3) subsetting ${CHR} transcripts via ${GENCODE_GTF}"
        zcat -f "${GENCODE_GTF}" | awk -v c="${CHR}" '$1==c && $3=="transcript"' \
            | grep -oE 'transcript_id "[^"]+"' | sed 's/transcript_id "//; s/"//' | sort -u \
            > "${WORK}/${CHR}_transcript_ids.txt"
        # gencode FASTA ids are 'ENST...|...'; match on the ENST prefix.
        seqkit grep -nr -f <(sed 's/$/|/' "${WORK}/${CHR}_transcript_ids.txt") "${GENCODE_FASTA}" \
            -o "${CHR1_TRANSCRIPTS}" 2>/dev/null || \
        seqkit grep -nr -p "$(paste -sd'|' "${WORK}/${CHR}_transcript_ids.txt")" "${GENCODE_FASTA}" \
            -o "${CHR1_TRANSCRIPTS}"
    fi
fi
echo "[gen]     reference: $(grep -c '^>' "${CHR1_TRANSCRIPTS}") ${CHR} transcripts"

# --- 4. grid: insert L1 → ART ----------------------------------------------
# SKIP_EXISTING=1 (default) resumes: a cell whose gzipped reads already exist is skipped.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    suffix="insert_level_${power}_delprob_${dp}"
    base="GRCh38.p14.${CHR}.${suffix}"
    mod_fa="${OUTPUT_DIR}/${base}.fa"
    ins_bed="${OUTPUT_DIR}/${base}.bed"
    art_prefix="${OUTPUT_DIR}/art/${base}.pair.${FCOV}x"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${art_prefix}1.fq.gz" && -s "${art_prefix}2.fq.gz" ]]; then
        echo "[gen] (4) ${suffix}: SKIP (reads exist)"; n_skip=$((n_skip + 1)); continue
    fi
    echo "[gen] (4) ${suffix}: insert 2^${power} L1 (del_prob=${dp})"
    python "${PYDIR}/generate_synthetic_dataset.py" \
        --transcripts "${CHR1_TRANSCRIPTS}" --l1-elements "${L1_FASTA}" \
        --power "${power}" --del-prob "${dp}" --seed "${SEED}" \
        --out-fasta "${mod_fa}" --out-bed "${ins_bed}"
    art_illumina -sam -na -i "${mod_fa}" -p -l "${ART_LEN}" -f "${FCOV}" \
        -m "${ART_MFLEN}" -s "${ART_SDEV}" -ss "${ART_SS}" -o "${art_prefix}" \
        > "${art_prefix}.art.log" 2>&1
    gzip -f "${art_prefix}1.fq" "${art_prefix}2.fq"
    # The modified-transcript FASTA is large and only needed for ART — drop it unless kept.
    [[ "${KEEP_FASTA:-0}" == "1" ]] || rm -f "${mod_fa}"
    n_done=$((n_done + 1))
  done
done
echo "[gen] grid: ${n_done} generated, ${n_skip} skipped"

echo "[gen] done → ${OUTPUT_DIR}  (art/*.fq.gz, *.bed, l1_synthetic.Index)"
echo "[gen] next: validate one sample — POWER=8 DELPROB=0.025 L1_INDEX=${L1_INDEX} sbatch scripts/slurm/synthetic_validation.slurm"
