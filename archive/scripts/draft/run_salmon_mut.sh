#!/bin/bash

# Define constants
if [ -z "$1" ]; then
    echo "Usage: $0 <record_name>"
    exit 1
fi
fcov=5
record_name=$1
output_dir="data/ref/GRCh38.p14.genome.${record_name}.mut"
index_dir="data/ref/l1base/IntactL1ElementsFLI-L1Ens84.38.Index"
ref_genome="data/ref/l1base/IntactL1ElementsFLI-L1Ens84.38.fa"
decoy_file="decoys.txt"
threads=8

# Create Salmon output directory
mkdir -p "${output_dir}/salmon"

# Loop through powers from 5 to 13
for power in $(seq 5 13); do
    echo "Running Salmon for insertion level 2^${power}..."

    fq1="${output_dir}/art/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x1.fq"
    fq2="${output_dir}/art/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x2.fq"
    output_prefix="${output_dir}/salmon/GRCh38.p14.${record_name}.insert_level_${power}.pair.${fcov}x"

    mkdir -p "$output_prefix"

    # Run Salmon
    salmon quant -q -i $index_dir -l A -1 $fq1 -2 $fq2 --validateMappings -o $output_prefix --threads $threads \
        > "$output_prefix/salmon.out" 2> "$output_prefix/salmon.err"
done

echo "All Salmon runs completed."
