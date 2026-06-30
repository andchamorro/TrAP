#!/bin/bash
#SBATCH --job-name=trap_synthetic_val
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# End-to-end validation — ONE synthetic chr1 sample through the seqlabel classifier
# FILTER → salmon (the "AlbertSalmon" path with the option-B model). Reuses TrAP's
# inference: predict.processing_dataset → predict.quantify → postprocessing.filter_ids
# → seqkit extract → salmon quant. Plan: .trap/plans/synthetic-e2e-validation.md.
#
# Single sample first (verify quant.sf parses), then fan out over POWER 5..13 ×
# DELPROB {0.000,0.025,0.050,0.075,0.100}.  Does NOT touch L1EM / plain-salmon results.
#
#   POWER=8 DELPROB=0.025 sbatch scripts/slurm/synthetic_validation.slurm
#   # fan-out: for p in $(seq 5 13); do for d in 0.000 0.025 0.050 0.075 0.100; do
#   #   POWER=$p DELPROB=$d sbatch scripts/slurm/synthetic_validation.slurm ; done ; done

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_gpu_modules
activate_trap
# salmon is provided by the trap conda env (activate_trap); seqkit from a module.
module load "${SEQKIT_MODULE:-SeqKit/2.9.0}" 2>/dev/null || echo "[synval] seqkit module not loaded — must be on PATH"
command -v salmon >/dev/null || echo "[synval] WARNING: salmon not on PATH — is the trap env active?"

# --- sample + paths --------------------------------------------------------
CHR="${CHR:-chr1}"; FCOV="${FCOV:-5}"
POWER="${POWER:-8}"; DELPROB="${DELPROB:-0.025}"
TAU="${TAU:-0.5}"                                  # keep reads with P(NEGATIVE) < TAU
SEQ_MODEL="${SEQ_MODEL:-albert.l1hs_l1pa2.v48.k17.salmon.seqlabel}"
TOK="${TOK:-tokenizer.gencode.v48.k17.salmon}"
K="${K:-17}"
# Dataset dir is suffixed by the L1 source (l1base / rm) so benchmarks coexist.
L1_SOURCE="${L1_SOURCE:-l1base}"
SUFFIX="insert_level_${POWER}_delprob_${DELPROB}"
REFDIR="${REFDIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}}"
# salmon index built by generate_synthetic_dataset.sh from the SAME L1 elements that were
# expressed (so quant targets == ground-truth elements).
L1_INDEX="${L1_INDEX:-${REFDIR}/l1_synthetic.Index}"
BASE="GRCh38.p14.${CHR}.${SUFFIX}.pair.${FCOV}x"
WORK="${WORK:-${SCRATCH:-/tmp}/synthetic_val/${BASE}}"
SALMON_OUT="${REFDIR}/salmon/filtered_seqlabel/${BASE}"
mkdir -p "${WORK}" "${SALMON_OUT}"

# Resolve paired reads, preferring UNCOMPRESSED .fq (the reads are parsed twice — for
# tokenise + extract — so avoiding gunzip is a real speedup). Fall back to .fq.gz; with
# DECOMPRESS=1 (default) a gzipped input is expanded once into WORK so neither pass pays
# repeated gunzip. DECOMPRESS=0 streams the .gz directly.
DECOMPRESS="${DECOMPRESS:-1}"
resolve_reads() {  # $1 = mate (1|2) → prints the read path to use
    local mate="$1"
    local plain="${REFDIR}/art/${BASE}${mate}.fq"
    local gz="${REFDIR}/art/${BASE}${mate}.fq.gz"
    if [[ -s "${plain}" ]]; then printf '%s' "${plain}"
    elif [[ -s "${gz}" ]]; then
        if [[ "${DECOMPRESS}" == "1" ]]; then
            local out="${WORK}/reads_${mate}.fq"
            [[ -s "${out}" ]] || gzip -dc "${gz}" > "${out}"
            printf '%s' "${out}"
        else printf '%s' "${gz}"; fi
    fi
}
R1="$(resolve_reads 1)"; R2="$(resolve_reads 2)"
for f in "${R1}" "${R2}"; do
    [[ -n "${f}" && -s "${f}" ]] || { echo "[synval] ERROR: missing reads for ${BASE} (looked for .fq / .fq.gz)" >&2; exit 1; }
done
echo "[synval] sample=${BASE} tau=${TAU} model=${SEQ_MODEL} reads=${R1##*/}"

# Resumable: SKIP_EXISTING=1 (default) reuses a completed tokenise / quantify so a
# re-run continues from the failed step instead of repeating the GPU pass.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# --- 1. tokenize the paired reads (salmon tokenizer) -----------------------
if [[ "${SKIP_EXISTING}" == "1" && -d "${WORK}/short_reads/tokenized" ]]; then
    echo "[synval] (1) tokenized dataset exists — SKIP"
else
    echo "[synval] (1) processing-dataset → ${WORK}/short_reads/tokenized"
    python -m trap.modeling.predict processing-dataset \
        --input-file "${R1}" --pair-file "${R2}" --output-path "${WORK}" \
        --k "${K}" --pretrained-tokenizer-name "${MODELS_DIR}/${TOK}" \
        || { echo "[synval] processing-dataset FAILED"; exit 1; }
fi

# --- 2. classify (GPU) → scores_*.parquet (id + class probs) ---------------
if [[ "${SKIP_EXISTING}" == "1" && -n "$(ls "${WORK}"/scores_*.parquet 2>/dev/null)" ]]; then
    echo "[synval] (2) scores exist — SKIP"
else
    echo "[synval] (2) quantify with ${SEQ_MODEL}"
    python -m trap.modeling.predict quantify \
        --pretrained-model-name "${MODELS_DIR}/${SEQ_MODEL}/final" \
        --output-path "${WORK}" --is-tokenized \
        || { echo "[synval] quantify FAILED"; exit 1; }
fi

# --- 3. filter: keep reads with P(NEGATIVE) < TAU --------------------------
echo "[synval] (3) filter_ids (P(NEGATIVE) < ${TAU})"
python -m trap.modeling.postprocessing filter-ids \
    --output-path "${WORK}" --threshold "${TAU}" \
    --fastq "${R1}" --output-filtered-ids "${WORK}/keep_ids.txt" \
    || { echo "[synval] filter_ids FAILED"; exit 1; }

# --- 4. extract the kept read pairs (ids are R1 names; map to /1,/2) --------
# scores 'id' = the R1 record id (carries the ART /1 suffix). Strip the mate marker
# to the fragment base, then re-add /1 and /2 to match the R1/R2 fq headers.
sed 's,/[12]$,,' "${WORK}/keep_ids.txt" | sort -u > "${WORK}/keep_base.txt"
sed 's,$,/1,' "${WORK}/keep_base.txt" > "${WORK}/keep_r1.txt"
sed 's,$,/2,' "${WORK}/keep_base.txt" > "${WORK}/keep_r2.txt"
echo "[synval] (4) seqkit grep — $(wc -l < "${WORK}/keep_base.txt") fragments kept"
# filtered reads kept uncompressed — salmon reads them once, no gzip overhead.
seqkit grep -f "${WORK}/keep_r1.txt" "${R1}" -o "${WORK}/filtered_1.fq"
seqkit grep -f "${WORK}/keep_r2.txt" "${R2}" -o "${WORK}/filtered_2.fq"

# --- 5. salmon quant on the filtered reads ---------------------------------
# Default to --sketch (pseudoalignment): selective alignment (the deprecated
# --validateMappings default) panics extending mappings near the 3' end of the short
# ~6kb L1 references. Override the mapping flags with SALMON_FLAGS if desired.
echo "[synval] (5) salmon quant → ${SALMON_OUT}"
salmon quant -q -i "${L1_INDEX}" -l A \
    -1 "${WORK}/filtered_1.fq" -2 "${WORK}/filtered_2.fq" \
    ${SALMON_FLAGS:---sketch} -o "${SALMON_OUT}" --threads "${SLURM_CPUS_PER_TASK}" \
    || { echo "[synval] salmon FAILED"; exit 1; }

echo "[synval] done. quant.sf rows: $(($(wc -l < "${SALMON_OUT}/quant.sf") - 1))  (= L1 elements in the index)"
echo "[synval] -> ${SALMON_OUT}/quant.sf"
