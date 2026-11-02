#!/bin/bash

# ============================
# Script: quantify_repeatmasker.sh
# Author: Andres D. Chamorro-Parejo, Texas A&M University
# Description: Quantifies RepeatMasker predictions using a pretrained model.
# Run from: script/ directory in the project root
# ============================

# Default values
TASK="quantify"
K=18
BATCH_SIZE=64
NUM_THREADS=8
GPUS_PER_NODE=2
NUM_MACHINES=1

# Help message
print_help() {
    echo "Usage: $0 [OPTIONS] --pretrained-model NAME --output-path PATH"
    echo ""
    echo "Required arguments:"
    echo "  -m, --pretrained-model NAME   Name of the pretrained model"
    echo "  -o, --output-path PATH        Output directory for quantification results"
    echo ""
    echo "Optional arguments:"
    echo "  -k, --k VALUE                 K-mer size (default: 18)"
    echo "  -b, --batch-size N           Batch size (default: 64)"
    echo "  -t, --num-threads N          Number of CPU threads (default: 8)"
    echo "  -g, --gpus-per-node N        Number of GPUs per node (default: 2)"
    echo "  -n, --num-machines N         Number of machines (default: 1)"
    echo "  -h, --help                   Show this help message and exit"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--pretrained-model) PRETRAINED_MODEL="$2"; shift ;;
        -o|--output-path) OUTPUT_PATH="$2"; shift ;;
        -k|--k) K="$2"; shift ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift ;;
        -t|--num-threads) NUM_THREADS="$2"; shift ;;
        -g|--gpus-per-node) GPUS_PER_NODE="$2"; shift ;;
        -n|--num-machines) NUM_MACHINES="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown option: $1"; print_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$PRETRAINED_MODEL" || -z "$OUTPUT_PATH" ]]; then
    echo "Error: --pretrained-model and --output-path are required."
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

export ACCELERATE_DIR="${ACCELERATE_DIR:-run/accelerate}"
export ACCELERATE_ARGS="--multi_gpu \
    --num_machines $NUM_MACHINES \
    --num_processes=$GPUS_PER_NODE \
    --dynamo_backend no \
    --mixed_precision fp16"

export SCRIPT="../trap/modeling/predict.py"

export SCRIPT_ARGS=" \
    --pretrained-model-name $PRETRAINED_MODEL \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE"

# ============================
# Run Quantification
# ============================

echo "Launching quantification with Accelerate..."
accelerate launch $ACCELERATE_ARGS $SCRIPT $TASK $SCRIPT_ARGS
echo "Quantification completed. Results saved to $OUTPUT_PATH"