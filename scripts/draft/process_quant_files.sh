#!/bin/bash

salmon_dir="../../data/ref/GRCh38.p14.genome.chr1.withdel/salmon/art"

# Loop over power values from 5 to 13
for power in {5..13}; do
  # Loop over deletion probabilities from 0.000 to 0.100 in steps of 0.025
  for prob in $(seq 0.000 0.025 0.100); do
    # Format the probability to 3 decimal places
    prob_fmt=$(printf "%.3f" $prob)
    # Construct the directory name
    dir="${salmon_dir}/GRCh38.p14.chr1.insert_level_${power}_delprob_${prob_fmt}.pair.5x"
    # Construct the path to quant.sf
    quant_file="${dir}/quant.sf"
    # Check if the file exists
    if [[ -f "$quant_file" ]]; then
      echo "Processing $quant_file with power $power"
      python update_tpm.py "$quant_file" "$power"
    else
      echo "File not found: $quant_file"
    fi
  done
done
