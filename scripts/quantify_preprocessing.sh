#!/bin/bash

# ============================
# Script: quantify_preprocessing.sh
# Author: Andres D. Chamorro-Parejo, Texas A&M University
# Description: Runs dataset preprocessing using a pretrained tokenizer.
# Run from: script/ directory in the project root
# ============================

# Default values
TASK="processing-dataset"
K=18
PRETRAINED_TOKENIZER="albert.l1hs_l1pa2.repeatmasker"
NUM_THREADS=32

# Help message
print_help() {
    echo "Usage: $0 [OPTIONS] --input-file PATH --pair-file PATH --output-path PATH"
    echo ""
    echo "Required arguments:"
    echo "  -i, --input-file PATH         Path to input FASTQ file (read 1)"
    echo "  -p, --pair-file PATH          Path to paired FASTQ file (read 2)"
    echo "  -o, --output-path PATH        Output directory for processed data"
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help                    Show this help message and exit"
    echo "  -k, --k VALUE                 K-mer size (default: 18)"
    echo "  -t, --task NAME               Task name (default: processing-dataset)"
    echo "  -n, --pretrained-tokenizer NAME  Pretrained tokenizer name (default: albert.l1hs_l1pa2.repeatmasker)"
    echo "  -w, --num-threads N           Number of CPU threads to use (default: 32)"
    echo ""
    echo "Example:"
    echo "  $0 -i ../data/sample_1.fastq -p ../data/sample_2.fastq -o ../output_dir"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) print_help; exit 0 ;;
        -i|--input-file) INPUT_FILE="$2"; shift ;;
        -p|--pair-file) PAIR_FILE="$2"; shift ;;
        -o|--output-path) OUTPUT_PATH="$2"; shift ;;
        -k|--k) K="$2"; shift ;;
        -t|--task) TASK="$2"; shift ;;
        -n|--pretrained-tokenizer) PRETRAINED_TOKENIZER="$2"; shift ;;
        -w|--num-threads) NUM_THREADS="$2"; shift ;;
        --) shift; break ;;
        -*)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
        *) break ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$INPUT_FILE" || -z "$PAIR_FILE" || -z "$OUTPUT_PATH" ]]; then
    echo "Error: --input-file, --pair-file, and --output-path are required."
    print_help
    exit 1
fi

# ============================
# Environment Variables
# ============================

export HF_HOME="$HOME/.cache/huggingface"
export HF_LOCAL_HOME="HF_LOCAL"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export SCRIPT="../translast/modeling/predict.py"

export SCRIPT_ARGS=" \
    --input-file $INPUT_FILE \
    --pair-file $PAIR_FILE \
    --output-path $OUTPUT_PATH \
    --k $K \
    --pretrained-tokenizer-name $PRETRAINED_TOKENIZER \
    --save-processing \
    --num-workers $NUM_THREADS"

# ============================
# Run Preprocessing
# ============================

echo "Running preprocessing with:"
echo "  Input: $INPUT_FILE"
echo "  Pair: $PAIR_FILE"
echo "  Output: $OUTPUT_PATH"
echo "  K: $K"
echo "  Tokenizer: $PRETRAINED_TOKENIZER"
echo "  Threads: $NUM_THREADS"
echo ""

python $SCRIPT $TASK $SCRIPT_ARGS