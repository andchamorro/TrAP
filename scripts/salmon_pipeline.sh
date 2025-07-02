#!/bin/bash

# ============================================
# Script: salmon_quantify_l1.sh
# Author: Andres D. Chamorro-Parejo, Texas A&M University
# Description: Build Salmon indexes and quantify L1 elements
# ============================================

# Default values
FORCE_BUILD=false
THREADS=8

# Help message
print_help() {
    echo "Usage: $0 [OPTIONS] -g REF_GENOME -b L1_BEDFILE -1 R1_FASTQ -2 R2_FASTQ -o OUTPUT_DIR"
    echo ""
    echo "Required arguments:"
    echo "  -g, --ref-genome PATH        Reference genome FASTA file"
    echo "  -b, --bed-file PATH          BED file with L1 elements"
    echo "  -1, --r1 PATH                Read 1 FASTQ file"
    echo "  -2, --r2 PATH                Read 2 FASTQ file"
    echo "  -o, --output-dir PATH        Output directory"
    echo ""
    echo "Optional arguments:"
    echo "  -d, --l1-outdir PATH         Directory to store BED/FASTA/Index files (default: same as BED file)"
    echo "  -t, --threads N              Number of threads (default: 8)"
    echo "  -f, --force                  Force rebuild of Salmon indexes"
    echo "  -h, --help                   Show this help message and exit"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--ref-genome) REF_GENOME="$2"; shift ;;
        -b|--bed-file) L1_BEDFILE="$2"; shift ;;
        -1|--r1) R1_FASTQ="$2"; shift ;;
        -2|--r2) R2_FASTQ="$2"; shift ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift ;;
        -d|--l1-outdir) L1_OUTDIR="$2"; shift ;;
        -t|--threads) THREADS="$2"; shift ;;
        -f|--force) FORCE_BUILD=true ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown option: $1"; print_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$REF_GENOME" || -z "$L1_BEDFILE" || -z "$R1_FASTQ" || -z "$R2_FASTQ" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    print_help
    exit 1
fi

# Set L1_OUTDIR to the directory of the BED file if not provided
L1_OUTDIR="${L1_OUTDIR:-$(dirname "$L1_BEDFILE")}"

# Derived paths
ONLY_BED="$L1_OUTDIR/only.bed"
RUNON_BED="$L1_OUTDIR/runon.bed"
ONLY_FASTA="$L1_OUTDIR/only.fa"
RUNON_FASTA="$L1_OUTDIR/runon.fa"
ONLY_INDEX="$L1_OUTDIR/only.Index"
RUNON_INDEX="$L1_OUTDIR/runon.Index"

mkdir -p "$OUTPUT_DIR"

# Function to build Salmon index if needed
build_index_if_needed() {
    local fasta=$1
    local index=$2
    local label=$3
    local bed=$4
    if [ "$FORCE_BUILD" = true ] || [ ! -d "$index" ] || [ "$fasta" -nt "$index" ] || [ "$L1_BEDFILE" -nt "$index" ]; then
        echo "Preparing $label BED and FASTA..."

        awk -F'\t' -v label="$label" '{
            split($4, a, ".")
            if (a[2] == "1") {
                split(a[3], loc, ":")
                split(loc[2], r, "-")
                if ($6 == "+") {
                    if (label == "only") print $1, r[1], r[2], $4"_only", 0, $6;
                    else if (label == "runon") print $1, r[1], r[2]+400, $4"_runon", 0, $6;
                } else {
                    if (label == "only") print $1, r[1], r[2], $4"_only", 0, $6;
                    else if (label == "runon") print $1, r[1]-400, r[2], $4"_runon", 0, $6;
                }
            }
        }' OFS='\t' "$L1_BEDFILE" > "$bed"

        echo "Extracting $label sequences..."
        bedtools getfasta -fi "$REF_GENOME" -bed "$bed" -s -name -fo "$fasta"

        echo "Building Salmon index for $label..."
        salmon index -t "$fasta" -i "$index" --threads $THREADS --kmerLen 31
    else
        echo "Index for $label is up to date."
    fi
}

# Build indexes
build_index_if_needed "$ONLY_FASTA" "$ONLY_INDEX" "only" "$ONLY_BED"
build_index_if_needed "$RUNON_FASTA" "$RUNON_INDEX" "runon" "$RUNON_BED"

# ============================================
# Filter reads using trap predictions
# ============================================

FILTERED_IDS_FILE="${OUTPUT_DIR}/filtered_ids.txt"
FILTERED_R1="${OUTPUT_DIR}/filtered_R1.fastq.gz"
FILTERED_R2="${OUTPUT_DIR}/filtered_R2.fastq.gz"

echo "Filtering read IDs using trap predictions..."
python ../trap/modeling/postprocessing.py filter-ids \
    --fastq "$R1_FASTQ" \
    --output-path "$OUTPUT_DIR" \
    --output-filtered-ids "$FILTERED_IDS_FILE"

echo "Filtering FASTQ files using seqkit..."
seqkit grep -f "$FILTERED_IDS_FILE" "$R1_FASTQ" -o "$FILTERED_R1"
seqkit grep -f "$FILTERED_IDS_FILE" "$R2_FASTQ" -o "$FILTERED_R2"

# Run Salmon quantification
run_salmon() {
    local index=$1
    local label=$2
    local outdir="${OUTPUT_DIR}/${label}"
    mkdir -p "$outdir"
    echo "Running Salmon for $label..."
    salmon quant -q -i "$index" -l ISR -1 "$FILTERED_R1" -2 "$FILTERED_R2" \
        --validateMappings -o "$outdir" --threads $THREADS \
        > "$outdir/salmon.out" 2> "$outdir/salmon.err"
}
run_salmon "$ONLY_INDEX" "only"
run_salmon "$RUNON_INDEX" "runon"

echo "All Salmon runs completed."
echo "Results are stored in: $OUTPUT_DIR"
echo "**********************************************"