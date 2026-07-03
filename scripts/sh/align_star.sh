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
SIM_MODEL="${SIM_MODEL:-transcript}"; _mtag=""; [[ "${SIM_MODEL}" == "insert" ]] && _mtag=".insert" || true
OUTPUT_DIR="${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.${CHR}.withdel.${L1_SOURCE}${_mtag}}"
GENOME="${GENOME:-data/external/GRCh38.p14.genome.fa.gz}"
GENOME_FA="${GENOME_FA:-${GENOME%.gz}}"   # decompressed genome (matches generate_synthetic_dataset.sh)
GENCODE_GTF="${GENCODE_GTF:-data/external/gencode.v48.annotation.gtf.gz}"
STAR_INDEX="${STAR_INDEX:-data/ref/star/GRCh38.p14.genome.StarIndex}"
POWERS="${POWERS:-5 6 7 8 9 10 11 12 13}"
DELPROBS="${DELPROBS:-0.000 0.025 0.050 0.075 0.100}"
ART_LEN="${ART_LEN:-150}"; THREADS="${THREADS:-8}"; SKIP_EXISTING="${SKIP_EXISTING:-1}"

for t in STAR samtools; do command -v "$t" >/dev/null 2>&1 || { echo "[star] ERROR: $t not on PATH" >&2; exit 1; }; done

# --- genome index (one-time, heavy; built ONLY by the prep job) ------------
# Gate on a .complete sentinel, not SA: STAR writes SA *before* the GTF junction
# insertion, so an OOM'd/partial build leaves SA behind and must not be treated as done.
# The build runs ONLY in prep mode (PREP_ONLY=1 or FORCE_PREP=1). Align tasks never build
# it — otherwise every array task would rebuild the whole-genome index into the same dir
# in parallel; they fail fast instead if the prep has not run.
index_ready=0; [[ -f "${STAR_INDEX}/.complete" && "${FORCE_PREP:-0}" != "1" ]] && index_ready=1
if [[ "${index_ready}" == "0" ]]; then
    if [[ "${PREP_ONLY:-0}" != "1" && "${FORCE_PREP:-0}" != "1" ]]; then
        echo "[star] ERROR: STAR index not built (${STAR_INDEX}/.complete missing)." >&2
        echo "[star]   build it once first:  PREP_ONLY=1 bash scripts/sh/align_star.sh" >&2
        echo "[star]   (submit_run_baseline.sh runs this as the prep job before the align array)." >&2
        exit 1
    fi
    if [[ ! -s "${GENOME_FA}" ]]; then
        [[ -s "${GENOME}" ]] || { echo "[star] ERROR: missing genome ${GENOME} (set GENOME=)" >&2; exit 1; }
        [[ "${GENOME}" == *.gz ]] && { echo "[star] decompressing genome → ${GENOME_FA}"; gunzip -kc "${GENOME}" > "${GENOME_FA}"; } \
            || GENOME_FA="${GENOME}"
    fi
    echo "[star] (prep) building STAR index → ${STAR_INDEX}"
    mkdir -p "${STAR_INDEX}"
    gtf_arg=()
    if [[ -f "${GENCODE_GTF}" ]]; then
        gtf="${GENCODE_GTF}"
        [[ "${gtf}" == *.gz ]] && { gtf="${STAR_INDEX}/annotation.gtf"; [[ -s "${gtf}" ]] || zcat -f "${GENCODE_GTF}" > "${gtf}"; }
        gtf_arg=(--sjdbGTFfile "${gtf}" --sjdbOverhang "$((ART_LEN - 1))")
    fi
    # genomeSAsparseD 2 halves the suffix-array RAM and on-disk index size (minor mapping
    # speed cost — fine for a benchmark); limitGenomeGenerateRAM lets STAR use the (larger)
    # prep allocation instead of chunking the SA sort to disk. Junction insertion for the
    # full GENCODE annotation is the memory peak, so the prep job is given extra RAM.
    STAR --runMode genomeGenerate --genomeDir "${STAR_INDEX}" \
         --genomeFastaFiles "${GENOME_FA}" "${gtf_arg[@]}" --runThreadN "${THREADS}" \
         --genomeSAsparseD "${STAR_SA_SPARSE:-2}" \
         --limitGenomeGenerateRAM "${STAR_GEN_RAM:-80000000000}"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STAR_INDEX}/.complete"   # non-empty sentinel (checked with -f)
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
    # STAR's default temp dir is ./_STARtmp in the CWD, so concurrent array tasks (all
    # launched from the repo root) collide on the same path. Give each cell a unique tmp
    # dir on node-local scratch; STAR requires it to NOT pre-exist, so clear any stale one.
    star_tmp="${TMPDIR:-/tmp}/starTmp_${base}_${SLURM_ARRAY_TASK_ID:-$$}"; rm -rf "${star_tmp}"
    STAR --genomeDir "${STAR_INDEX}" --readFilesIn "${r1}" "${r2}" "${readcmd[@]}" \
         --outSAMtype BAM SortedByCoordinate --outFileNamePrefix "${out}/" \
         --outTmpDir "${star_tmp}" \
         --outSAMprimaryFlag AllBestScore --outFilterMultimapNmax 1000 \
         --runThreadN "${THREADS}" > "${out}/STAR.out" 2> "${out}/STAR.err"
    rm -rf "${star_tmp}"
    samtools index "${bam}"
    n_done=$((n_done + 1))
  done
done
echo "[star] done: ${n_done} aligned, ${n_skip} skipped → ${OUTPUT_DIR}/star"
