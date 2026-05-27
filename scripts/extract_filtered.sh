#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 -r <record_name> -d \"<del_prob_start> <del_prob_end> <step>\" [-f <fcov>] [-t <threads>]"
    exit 1
}

# Default values
fcov=5
threads=8

# Parse arguments
while getopts ":r:d:f:t:" opt; do
  case $opt in
    r) record_name="$OPTARG" ;;
    d) IFS=' ' read -r del_start del_end del_step <<< "$OPTARG" ;;
    f) fcov="$OPTARG" ;;
    t) threads="$OPTARG" ;;
    *) usage ;;
  esac
done

# Check required arguments
if [ -z "$record_name" ] || [ -z "$del_start" ] || [ -z "$del_end" ] || [ -z "$del_step" ]; then
    usage
fi

output_dir="../data/ref/GRCh38.p14.genome.${record_name}.withdel"
mkdir -p "${output_dir}/filtered"

# Define the function to process filtered reads
process_filtered_reads() {
    local output_dir="$1"
    local record_name="$2"
    local fcov="$3"
    local power="$4"
    local del_prob="$5"

    suffix="insert_level_${power}_delprob_${del_prob}"
    echo "Extracting filtered reads for ${suffix}..."

    bed_file="${output_dir}/GRCh38.p14.${record_name}.${suffix}.bed"
    bam_file="${output_dir}/art/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x.bam"
    fq1="${output_dir}/art/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x1.fq.gz"
    fq2="${output_dir}/art/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x2.fq.gz"
    output_prefix="${output_dir}/filtered/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x"
    fq1_out="${output_prefix}1.fq.gz"
    fq2_out="${output_prefix}2.fq.gz"

    mkdir -p "$output_prefix"

    echo "Sorting BED file..."
    cut -f1-6 "$bed_file" > "${output_prefix}/filtered.bed"
    sort -k1,1 -k2,2n "${output_prefix}/filtered.bed" > "${output_prefix}/filtered.sorted.bed"
    bed_file="${output_prefix}/filtered.sorted.bed"

    intersected_bam="${output_prefix}/reads.intersected.bam"
    samtools index "$bam_file"
    bedtools intersect -abam "$bam_file" -b "$bed_file" -u > "$intersected_bam"

    intersected_names_base="${output_prefix}/intersected_read_names_base.txt"
    samtools view "$intersected_bam" | cut -f1 | sort | uniq > "$intersected_names_base"

    intersected_names_r1="${output_prefix}/intersected_read_names_r1.txt"
    intersected_names_r2="${output_prefix}/intersected_read_names_r2.txt"
    sed 's/$/\/1/' "$intersected_names_base" > "$intersected_names_r1"
    sed 's/$/\/2/' "$intersected_names_base" > "$intersected_names_r2"

    seqkit grep -f "$intersected_names_r1" "$fq1" -o "$fq1_out"
    seqkit grep -f "$intersected_names_r2" "$fq2" -o "$fq2_out"

    echo "Cleaning temp files..."
    rm -f "$intersected_bam" "$intersected_names_base" "${output_prefix}/filtered.bed" "${output_prefix}/filtered.sorted.bed"
}

export -f process_filtered_reads

# Generate del_prob and power combinations
combinations=()
for del_prob in $(seq -f "%.3f" $del_start $del_step $del_end); do
  for power in $(seq 5 13); do
    combinations+=("$output_dir $record_name $fcov $power $del_prob")
  done
done

# Run in parallel
printf "%s\n" "${combinations[@]}" | parallel --colsep ' ' --jobs "$threads" --bar process_filtered_reads

echo "All ML preprocessing paired reads are extracted."
# End of script