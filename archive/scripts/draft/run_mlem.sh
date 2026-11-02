#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 -r <record_name> -d \"<del_prob_start> <del_prob_end> <step>\""
    exit 1
}

# Parse arguments
while getopts ":r:d:" opt; do
  case $opt in
    r) record_name="$OPTARG" ;;
    d) IFS=' ' read -r del_start del_end del_step <<< "$OPTARG" ;;
    *) usage ;;
  esac
done

# Check required arguments
if [ -z "$record_name" ] || [ -z "$del_start" ] || [ -z "$del_end" ] || [ -z "$del_step" ]; then
    usage
fi

# Constants
fcov=5
output_dir="../../data/ref/GRCh38.p14.genome.${record_name}.withdel"
l1em_path="../../../../L1EM"
genome_path="../../data/ref/GRCh38.p14.genome.fa"
bed_file="../../../../L1EM/annotation/hsflil1_8438.bed"

# Create MLEM output directory
mkdir -p "${output_dir}/MLEM"

# Loop through del_probs and powers
for del_prob in $(seq -f "%.3f" $del_start $del_step $del_end); do
  for power in $(seq 5 13); do
    suffix="insert_level_${power}_delprob_${del_prob}"
    echo "Running MLEM for ${suffix}..."

    bam_file="${output_dir}/art/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x.bam"
    fq1="${output_dir}/filtered/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x1.fq.gz"
    fq2="${output_dir}/filtered/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x2.fq.gz"

    output_prefix="${output_dir}/MLEM/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x"
    mkdir -p "$output_prefix/filtered"

    gzip -dc "$fq1" > "${output_prefix}/filtered/filtered_r1.fq"
    gzip -dc "$fq2" > "${output_prefix}/filtered/filtered_r2.fq"

    # Convert paths to absolute
    abs_l1em_path=$(realpath "$l1em_path")
    abs_bam_file=$(realpath "$bam_file")
    abs_genome_path=$(realpath "$genome_path")
    abs_bed_file=$(realpath "$bed_file")

    # Run MLEM
    bash "${abs_l1em_path}/run_MLL1EM.sh" "$abs_bam_file" "$abs_l1em_path" "$abs_genome_path" "$abs_bed_file" "$output_prefix" \
        > "$output_prefix/L1EM.out" 2> "$output_prefix/L1EM.err"

    # Clean up
    rm -rf "${output_prefix}/filtered" "${output_prefix}/split_fqs" "${output_prefix}/G_of_R" "${output_prefix}/idL1reads"
  done
done
echo "All MLEM runs completed."
# End of script