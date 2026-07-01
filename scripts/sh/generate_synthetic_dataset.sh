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
# L1 element source (whole-genome annotations; extracted, then inserted into chr1 transcripts):
#   l1base — 146 curated intact full-length L1 (distinguishable UIDs) from L1Base
#   rm     — RepeatMasker promoter-positive full-length genomic L1 (near-identical, many)
L1_SOURCE="${L1_SOURCE:-l1base}"
L1BASE_BED="${L1BASE_BED:-data/ref/l1base/hsflil1_8438.bed}"
PROMOTER_BED="${PROMOTER_BED:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_rm.LINE1.promoter.bed}"
GENCODE_FASTA="${GENCODE_FASTA:-${DATA_EXTERNAL}/gencode.v48.transcripts.fa.gz}"
GENCODE_GTF="${GENCODE_GTF:-${DATA_EXTERNAL}/gencode.v48.annotation.gtf.gz}"
CHR1_TRANSCRIPTS="${CHR1_TRANSCRIPTS:-}"   # optional: skip GTF subset if provided
CHR="${CHR:-chr1}"
# RM-source full-length filter (bp); L1Base elements are pre-curated (no length filter).
L1_MIN_LEN="${L1_MIN_LEN:-5500}"; L1_MAX_LEN="${L1_MAX_LEN:-6500}"

# --- ART + grid + output ---------------------------------------------------
# Dataset dir is suffixed by the L1 source so the l1base and rm benchmarks coexist.
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
FCOV="${FCOV:-5}"; ART_LEN="${ART_LEN:-150}"; ART_MFLEN="${ART_MFLEN:-500}"
ART_SDEV="${ART_SDEV:-10}"; ART_SS="${ART_SS:-MSv3}"
SEED="${SEED:-3469}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
PYDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../python" && pwd)"
WORK="${OUTPUT_DIR}/_inputs"
mkdir -p "${OUTPUT_DIR}/art" "${WORK}"

# SIM_MODEL (transcript = standalone L1 pool, model 2; insert = L1 into host chr1
# transcripts, model 1 — the "background" experiment) is set above with the dir tag.
L1_BED_SRC="${PROMOTER_BED}"; [[ "${L1_SOURCE}" == "l1base" ]] && L1_BED_SRC="${L1BASE_BED}"
for f in "${GENOME}" "${L1_BED_SRC}"; do
    [[ -f "${f}" ]] || { echo "[gen] ERROR: missing input ${f}" >&2; exit 1; }
done

# --- 0. genome FASTA (uncompressed + .fai for getfasta) --------------------
GENOME_FA="${GENOME%.gz}"
if [[ "${GENOME}" == *.gz && ! -s "${GENOME_FA}" ]]; then
    echo "[gen] decompressing genome for getfasta"; gunzip -kc "${GENOME}" > "${GENOME_FA}"
fi
[[ -f "${GENOME_FA}.fai" ]] || samtools faidx "${GENOME_FA}"

# --- 1. full-length L1 elements (insert + salmon ref) ----------------------
# Steps 0–3 are idempotent (skip if the output exists) so parallel array workers
# reuse the shared inputs a prep job built; FORCE_PREP=1 rebuilds them.
L1_FASTA="${WORK}/l1_fulllength.fa"
if [[ ! -s "${L1_FASTA}" || "${FORCE_PREP:-0}" == "1" ]]; then
    if [[ "${L1_SOURCE}" == "l1base" ]]; then
        echo "[gen] (1) L1Base intact full-length L1 (genome-wide) from ${L1BASE_BED}"
        # 146 curated FLI-L1; col4 = UID (already unique). Normalize UID-NNN -> UIDNNN.
        cut -f1-6 "${L1BASE_BED}" > "${WORK}/l1_fulllength.bed"
        bedtools getfasta -nameOnly -s -fi "${GENOME_FA}" -bed "${WORK}/l1_fulllength.bed" \
            | sed -e '/^>/ s/(.)$//' -e 's/^>UID-/>UID/' > "${L1_FASTA}"
    else
        echo "[gen] (1) full-length promoter+ L1 from ${PROMOTER_BED} (${L1_MIN_LEN}-${L1_MAX_LEN} bp)"
        # RM source: name col is <subfamily>.{0,1} (.1 = L19088.1 promoter hit); make the
        # name UNIQUE per element (subfamily.flag::chrom:start-end), else -nameOnly collapses
        # thousands of distinct genomic L1 to ~73 names and breaks the per-element join.
        PROMOTER_ONLY="${PROMOTER_ONLY:-1}"
        awk -v lo="${L1_MIN_LEN}" -v hi="${L1_MAX_LEN}" -v prom="${PROMOTER_ONLY}" 'BEGIN{OFS="\t"}
             ($3-$2)>=lo && ($3-$2)<=hi && (prom!=1 || $4 ~ /\.1$/) {$4=$4"::"$1":"$2"-"$3; print}' \
             "${PROMOTER_BED}" > "${WORK}/l1_fulllength.bed"
        bedtools getfasta -nameOnly -s -fi "${GENOME_FA}" -bed "${WORK}/l1_fulllength.bed" \
            | sed '/^>/ s/(.)$//' > "${L1_FASTA}"
    fi
fi
echo "[gen]     $(grep -c '^>' "${L1_FASTA}") L1 elements, $(grep '^>' "${L1_FASTA}" | sort -u | wc -l) unique names"

# --- 2. salmon index from the L1 elements ----------------------------------
L1_INDEX="${OUTPUT_DIR}/l1_synthetic.Index"
if [[ ! -f "${L1_INDEX}/info.json" || "${FORCE_PREP:-0}" == "1" ]]; then
    echo "[gen] (2) salmon index → ${L1_INDEX}"
    salmon index -t "${L1_FASTA}" -i "${L1_INDEX}" -k 31 2>/dev/null \
        || echo "[gen]   WARNING: salmon index failed (salmon on PATH?) — build it before quant"
else
    echo "[gen] (2) salmon index exists → ${L1_INDEX}"
fi

# --- 3. host chr1 transcripts (insertion reference; model 1 / insert only) --
if [[ "${SIM_MODEL}" == "insert" && -z "${CHR1_TRANSCRIPTS}" ]]; then
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
    echo "[gen]     host reference: $(grep -c '^>' "${CHR1_TRANSCRIPTS}") ${CHR} transcripts"
fi

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
    echo "[gen] PREP_ONLY — shared inputs ready (L1 elements + salmon index${CHR1_TRANSCRIPTS:+ + host transcripts}); skipping grid"
    exit 0
fi

# --- 4. grid: insert L1 → ART ----------------------------------------------
# SKIP_EXISTING=1 (default) resumes: a cell whose reads already exist is skipped.
# GZIP=1 (default) gzips the reads; GZIP=0 leaves them uncompressed (.fq) for faster
# downstream validation (the validation reads them twice).
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GZIP="${GZIP:-1}"
[[ "${GZIP}" == "1" ]] && RDEXT=".fq.gz" || RDEXT=".fq"
n_done=0; n_skip=0
for power in ${POWERS}; do
  for dp in ${DELPROBS}; do
    suffix="insert_level_${power}_delprob_${dp}"
    base="GRCh38.p14.${CHR}.${suffix}"
    art_fa="${OUTPUT_DIR}/${base}.fa"          # what ART reads from (deleted unless KEEP_FASTA)
    art_prefix="${OUTPUT_DIR}/art/${base}.pair.${FCOV}x"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${art_prefix}1${RDEXT}" && -s "${art_prefix}2${RDEXT}" ]]; then
        echo "[gen] (4) ${suffix}: SKIP (reads exist)"; n_skip=$((n_skip + 1)); continue
    fi
    if [[ "${SIM_MODEL}" == "transcript" ]]; then
        echo "[gen] (4) ${suffix}: 2^${power} L1 transcript copies (del_prob=${dp})"
        python "${PYDIR}/generate_synthetic_dataset.py" --model transcript \
            --l1-elements "${L1_FASTA}" --power "${power}" --del-prob "${dp}" --seed "${SEED}" \
            --out-fasta "${art_fa}" --out-counts "${OUTPUT_DIR}/${base}.counts.tsv"
    else
        echo "[gen] (4) ${suffix}: insert 2^${power} L1 into ${CHR} transcripts (del_prob=${dp})"
        python "${PYDIR}/generate_synthetic_dataset.py" --model insert \
            --l1-elements "${L1_FASTA}" --transcripts "${CHR1_TRANSCRIPTS}" \
            --power "${power}" --del-prob "${dp}" --seed "${SEED}" \
            --out-fasta "${art_fa}" --out-bed "${OUTPUT_DIR}/${base}.bed"
    fi
    art_illumina -sam -na -i "${art_fa}" -p -l "${ART_LEN}" -f "${FCOV}" \
        -m "${ART_MFLEN}" -s "${ART_SDEV}" -ss "${ART_SS}" -o "${art_prefix}" \
        > "${art_prefix}.art.log" 2>&1
    [[ "${GZIP}" == "1" ]] && gzip -f "${art_prefix}1.fq" "${art_prefix}2.fq"
    # The ART source FASTA is large and only needed for ART — drop it unless kept.
    [[ "${KEEP_FASTA:-0}" == "1" ]] || rm -f "${art_fa}"
    n_done=$((n_done + 1))
  done
done
echo "[gen] grid: ${n_done} generated, ${n_skip} skipped"

echo "[gen] done → ${OUTPUT_DIR}  (art/*.fq.gz, ground-truth ${SIM_MODEL/transcript/*.counts.tsv}${SIM_MODEL/insert/*.bed}, l1_synthetic.Index)"
echo "[gen] next: validate — POWER=8 DELPROB=0.025 REFDIR=${OUTPUT_DIR} L1_INDEX=${L1_INDEX} sbatch scripts/slurm/synthetic_validation.slurm"
