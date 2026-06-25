#!/bin/bash
# Build the GENOMIC L1 reference for the minimap2 cross-check (Track B0, no Dfam).
# Extracts the actual L1 ELEMENT sequences from the genome at the RepeatMasker .out
# coordinates, named by subfamily — the most faithful cross-check reference for this
# dataset (the reads derive from these exact genomic copies, so true L1 aligns ~100%,
# across ALL subfamilies and divergence levels).
#
# NB: this is NOT L1_CORPUS (GCF_..._rm.LINE1.gencode.v48.fa). L1_CORPUS is the
# L1-OVERLAPPING TRANSCRIPTS (full mRNA, mostly non-L1 gene sequence — the ART source);
# this is the L1 ELEMENTS themselves (pure repeat sequence). Aligning reads to L1_CORPUS
# would only confirm transcript provenance, not L1 content — hence the genomic instances.
#
# Run on a compute/login node (light): needs bedtools + seqkit + the genome FASTA the
# STAR index was built from (chr*-named, matching the remapped L1 BED).
#   module load GCCcore/13.3.0 BEDTools/2.31.1 SeqKit/2.9.0
#   GENOME=data/external/<genome>.fa bash scripts/build_l1_genomic_ref.sh
set -euo pipefail

_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/_common.sh"
[[ -z "${DATA_EXTERNAL:-}" && -f "${_COMMON}" ]] && source "${_COMMON}"
DATA_EXTERNAL="${DATA_EXTERNAL:-data/external}"

GENOME="${GENOME:?set GENOME=<FASTA the STAR index was built from, chr*-named>}"
[[ -f "${GENOME}" ]] || { echo "[l1ref] ERROR: GENOME not found: ${GENOME}" >&2; exit 1; }
LINE1_OUT="${LINE1_OUT:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_rm.LINE1.out.gz}"
CHROM_MAP="${CHROM_MAP:-${DATA_EXTERNAL}/GCF_000001405.40_GRCh38.p14_assembly_report.txt}"
MAX_DIV="${MAX_DIV:-100}"                # e.g. 10 → young-only L1 reference
OUT="${OUT:-${DATA_EXTERNAL}/l1_genomic_instances.fa}"
WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT

CHROM_MAP_ARG=(); [[ -f "${CHROM_MAP}" ]] && CHROM_MAP_ARG=(--chrom-map "${CHROM_MAP}")

echo "[l1ref] .out → L1 BED (chr*-remapped, max_div=${MAX_DIV})"
python -m trap.utils.rmout to-bed --rmout "${LINE1_OUT}" --out "${WORK}/l1.bed" \
    --max-div "${MAX_DIV}" "${CHROM_MAP_ARG[@]}"

echo "[l1ref] bedtools getfasta (named by subfamily, stranded) → ${OUT}"
# -nameOnly: FASTA header = the BED name col (subfamily), so relabel_by_sequence reads
# target_name=subfamily. -s: extract in the element's annotated orientation.
bedtools getfasta -nameOnly -s -fi "${GENOME}" -bed "${WORK}/l1.bed" \
    | sed '/^>/ s/(.)$//' > "${OUT}"   # drop the "(+)"/"(-)" bedtools appends to -nameOnly headers

command -v seqkit >/dev/null && seqkit stats "${OUT}"
echo "[l1ref] subfamilies (top):"
grep '^>' "${OUT}" | sed 's/^>//' | sort | uniq -c | sort -rn | head -20
echo "[l1ref] done -> ${OUT}"
echo "[l1ref] use:  CROSSCHECK_LIB=${OUT}  (build_seqlabel.slurm or relabel_by_sequence run)"
