#!/bin/bash
# Define variables
export REFERENCE_GENOME_FILE="data/ref/GRCh38.p14.genome.fa"
export ANNOTATION_SAF_FILE="data/annotations/L1EM/L1EM.400.saf"
export INPUTFILE="data/SRR4099955/SRR4099955.fastq"
export OUTPUT_DIRECTORY="outputs/align_quant/bwa/SRR4099955"
export FEATURECOUNTS_PATH="$HOME/miniconda3/envs/rosetta/bin/featureCounts"
export NUM_THREADS=8

# Log the input information
echo "Aligned Quantification Log"
echo "========================="
echo "Reference FASTA file: $REFERENCE_GENOME_FILE"
echo "Annotation file: $ANNOTATION_SAF_FILE"
echo "Reads : $INPUTFILE"
echo "========================="
echo "Starting quantification..."

# Run the analysis script
bash sh/bwamemAndCount.sh -j $NUM_THREADS \
	-a $ANNOTATION_SAF_FILE \
	-r $REFERENCE_GENOME_FILE \
	-i $INPUTFILE \
	-f $FEATURECOUNTS_PATH \
	-o $OUTPUT_DIRECTORY

# Log completion
echo "Quantification completed."
echo "Results are stored in: $OUTPUT_DIRECTORY"
