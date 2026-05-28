#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 -r <record_name> -p <prefix_dir> -d \"<del_prob_start> <del_prob_end> <step>\" [-f <fcov>] [-t <threads>]"
    exit 1
}

# Default values
fcov=5
threads=8
prefix_dir="art"

# Parse arguments
while getopts ":r:p:d:f:t:" opt; do
  case $opt in
    r) record_name="$OPTARG" ;;
    p) prefix_dir="$OPTARG" ;;
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

# Define constants
output_dir="../data/ref/GRCh38.p14.genome.${record_name}.withdel"
index_dir="../data/ref/l1base/IntactL1ElementsFLI-L1Ens84.38.Index"
ref_genome="../data/ref/l1base/IntactL1ElementsFLI-L1Ens84.38.fa"

# Create Salmon output directory
mkdir -p "${output_dir}/salmon"

# Calculate total number of iterations
del_probs=($(seq -f "%.3f" $del_start $del_step $del_end))
total_iterations=$((${#del_probs[@]} * 9))  # 9 powers from 5 to 13
current=0

# Loop with progress bar
for del_prob in "${del_probs[@]}"; do
  for power in $(seq 5 13); do
    current=$((current + 1))
    percent=$((100 * current / total_iterations))
    bar=$(printf "%-${percent}s" "#" | tr ' ' '#')
    printf "\rProgress: [%-100s] %d%%" "$bar" "$percent"

    echo -ne "\nRunning Salmon for del_prob=${del_prob}, insertion level 2^${power}...\n"

    suffix="insert_level_${power}_delprob_${del_prob}"
    fq1="${output_dir}/${prefix_dir}/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x1.fq.gz"
    fq2="${output_dir}/${prefix_dir}/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x2.fq.gz"
    output_prefix="${output_dir}/salmon/${prefix_dir}/GRCh38.p14.${record_name}.${suffix}.pair.${fcov}x"

    mkdir -p "$output_prefix"

    salmon quant -q -i "$index_dir" -l A -1 "$fq1" -2 "$fq2" --validateMappings -o "$output_prefix" --threads "$threads" \
        > "$output_prefix/salmon.out" 2> "$output_prefix/salmon.err"
  done
done

echo -e "\nAll Salmon runs completed."