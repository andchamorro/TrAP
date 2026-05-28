#!/bin/bash

# Define constants
if [ -z "$1" ]; then
    echo "Usage: $0 <record_name>"
    exit 1
fi
fcov=5
record_name=$1
output_dir="GRCh38.p14.genome.${record_name}.mut"
l1em_path="../../../../L1EM"
genome_path="GRCh38.p14.genome.fa"
bed_file="../../../../L1EM/annotation/hsflil1_8438.bed"

# Create L1EM output directory
mkdir -p "${output_dir}/MLEM"

# Loop through powers from 5 to 13
for power in $(seq 5 13); do
    echo "Running MLEM for insertion level 2^${power}..."

    bam_file="${output_dir}/art/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x.bam"
    fq1="${output_dir}/filtered/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x/filtered_r1.fq"
    fq2="${output_dir}/filtered/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x/filtered_r2.fq"

    output_prefix="${output_dir}/MLEM/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x"
    mkdir -p "$output_prefix"
    mkdir -p "${output_prefix}/filtered"

    cp "$fq1" "${output_prefix}/filtered/filtered_r1.fq"
    cp "$fq2" "${output_prefix}/filtered/filtered_r2.fq"

    # Convert paths to absolute
    abs_l1em_path=$(realpath "$l1em_path")
    abs_bam_file=$(realpath "$bam_file")
    abs_genome_path=$(realpath "$genome_path")
    abs_bed_file=$(realpath "$bed_file")

    # Run L1EM
    bash "${abs_l1em_path}/run_MLL1EM.sh" "$abs_bam_file" "$abs_l1em_path" "$abs_genome_path" "$abs_bed_file" "$output_prefix" \
        > "$output_prefix/L1EM.out" 2> "$output_prefix/L1EM.err"
done

echo "All L1EM runs completed."
