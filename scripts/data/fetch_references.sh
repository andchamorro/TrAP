#!/bin/bash
# Fetch reference data for the TrAP Phase-3 re-do (plan §6.1 / §9.1) and build
# the STAR + BWA indexes. No OneDrive coupling — everything comes from the
# canonical GENCODE / EBI sources so the pipeline is reproducible from scratch.
#
# Requires (load via scripts/slurm/_common.sh::load_bio_modules): STAR, bwa,
# samtools, plus wget. Override any path/URL via the environment.
set -euo pipefail

DATA_EXTERNAL="${DATA_EXTERNAL:-data/external}"
REF_DIR="${REF_DIR:-${DATA_EXTERNAL}/star_index}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

GENCODE_RELEASE="${GENCODE_RELEASE:-48}"
BASE="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}"
TRANSCRIPTS_URL="${TRANSCRIPTS_URL:-${BASE}/gencode.v${GENCODE_RELEASE}.transcripts.fa.gz}"
GENOME_URL="${GENOME_URL:-${BASE}/GRCh38.p14.genome.fa.gz}"
GTF_URL="${GTF_URL:-${BASE}/gencode.v${GENCODE_RELEASE}.primary_assembly.annotation.gtf.gz}"

# RepeatMasker GFF — NCBI accession for GRCh38.p14; matches the STAR genome
NCBI_ACCESSION="${NCBI_ACCESSION:-GCF_000001405.40}"
RM_GFF="${REPEATMASKER_GFF:-${DATA_EXTERNAL}/${NCBI_ACCESSION}_GRCh38.p14_rm.gff}"

mkdir -p "${DATA_EXTERNAL}" "${REF_DIR}"

fetch() {  # url dest
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "[fetch] exists, skipping: $dest"
    else
        echo "[fetch] $url -> $dest"
        wget -c -O "$dest" "$url"
    fi
}

# --- 1. Downloads ----------------------------------------------------------
TRANSCRIPTS="${DATA_EXTERNAL}/gencode.v${GENCODE_RELEASE}.transcripts.fa.gz"
GENOME_GZ="${DATA_EXTERNAL}/GRCh38.p14.genome.fa.gz"
GTF_GZ="${DATA_EXTERNAL}/gencode.v${GENCODE_RELEASE}.primary_assembly.annotation.gtf.gz"

fetch "${TRANSCRIPTS_URL}" "${TRANSCRIPTS}"
fetch "${GENOME_URL}" "${GENOME_GZ}"
fetch "${GTF_URL}" "${GTF_GZ}"

# --- 2. Decompress genome + GTF (STAR/bwa want plain text) -----------------
GENOME_FA="${GENOME_GZ%.gz}"
GTF="${GTF_GZ%.gz}"
[[ -s "${GENOME_FA}" ]] || { echo "[gunzip] ${GENOME_GZ}"; gunzip -k "${GENOME_GZ}"; }
[[ -s "${GTF}" ]] || { echo "[gunzip] ${GTF_GZ}"; gunzip -k "${GTF_GZ}"; }

# --- 3. STAR index ---------------------------------------------------------
if [[ -s "${REF_DIR}/SAindex" ]]; then
    echo "[STAR] index exists, skipping: ${REF_DIR}"
else
    echo "[STAR] building genome index -> ${REF_DIR}"
    STAR --runMode genomeGenerate \
        --runThreadN "${THREADS}" \
        --genomeDir "${REF_DIR}" \
        --genomeFastaFiles "${GENOME_FA}" \
        --sjdbGTFfile "${GTF}" \
        --sjdbOverhang 149
fi

# --- 4. BWA index ----------------------------------------------------------
if [[ -s "${GENOME_FA}.bwt" ]]; then
    echo "[bwa] index exists, skipping: ${GENOME_FA}"
else
    echo "[bwa] building index for ${GENOME_FA}"
    bwa index "${GENOME_FA}"
fi

# --- 5. RepeatMasker GFF (LINE-1 labelling for build_dataset.sh) -----------
if [[ -s "${RM_GFF}" ]]; then
    echo "[fetch] exists, skipping: ${RM_GFF}"
else
    if [[ -n "${RM_OUT:-}" ]]; then
        echo "[rm_to_gff3] converting local RepeatMasker .out: ${RM_OUT}"
        python "$(dirname "$0")/../rm_to_gff3.py" \
            --input "${RM_OUT}" \
            --report-file "${RM_REF:-}" \
            --output "${RM_GFF}"
    else
        echo "[rm_to_gff3] downloading NCBI RepeatMasker for ${NCBI_ACCESSION}"
        python "$(dirname "$0")/../rm_to_gff3.py" \
            --accession "${NCBI_ACCESSION}" \
            --output "${RM_GFF}" \
            --keep-rm
    fi
fi

# --- 6. L1 RepeatMasker FASTA for tokenizer training ----------------------
# Extract LINE/L1 sequences from the genome at RepeatMasker loci and write a
# FASTA with >ID|Name|Family headers.  Consumed by 10_tokenizer.slurm via
# --extra-corpus / L1_CORPUS to pin LINE-1 k-mers in the vocabulary.
# Includes the GENCODE release in the filename for traceability.
L1_CORPUS="${L1_CORPUS:-${DATA_EXTERNAL}/${NCBI_ACCESSION}_GRCh38.p14_rm.LINE1.gencode.v${GENCODE_RELEASE}.fa}"
if [[ -s "${L1_CORPUS}" ]]; then
    echo "[fetch] exists, skipping: ${L1_CORPUS}"
else
    echo "[build_l1_corpus] extracting LINE1 sequences -> ${L1_CORPUS}"
    python "$(dirname "$0")/../build_l1_corpus.py" \
        --gff "${RM_GFF}" \
        --genome "${GENOME_FA}" \
        --output "${L1_CORPUS}"
fi

echo "[fetch_references] done:"
echo "  transcripts   : ${TRANSCRIPTS}"
echo "  genome        : ${GENOME_FA}"
echo "  gtf           : ${GTF}"
echo "  STAR index    : ${REF_DIR}"
echo "  RepeatMasker  : ${RM_GFF}"
echo "  L1 corpus     : ${L1_CORPUS}"
