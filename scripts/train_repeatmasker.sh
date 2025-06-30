#!/bin/bash

# ============================
# Script: train_repeatmasker.sh
# Author: Andres D. Chamorro-Parejo, Texas A&M University
# Description: Launches training for RepeatMasker classification using Accelerate.
# Run from: script/ directory in the project root
# ============================

# Default values
NUM_THREADS=8
GPUS_PER_NODE=2
NUM_MACHINES=1
MODEL_NAME="albert.l1hs_l1pa2.repeatmasker"
TASK="classification"
TOKENIZER_PATH="../models/transcripts_sentencepiece/gencode.v47.transcripts.k18.32k.skipn.nocompress/huggingface/fast"
PROCESSING_NAME="gencode.v47.transcripts.k18.32k.skipn.nocompress/l1hs_l1pa2"
TRAINER_CONFIG_PATH="trainer_config_base_repeatmasker.json"

# Help message
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                    Show this help message and exit"
    echo "  -t, --num-threads N           Number of CPU threads to use (default: 8)"
    echo "  -g, --gpus-per-node N         Number of GPUs per node (default: 2)"
    echo "  -m, --num-machines N          Number of machines to use (default: 1)"
    echo "  -n, --model-name NAME         Model name identifier"
    echo "  -k, --task TASK               Task name (e.g., classification)"
    echo "  -p, --tokenizer-path PATH     Path to pretrained tokenizer"
    echo "  -r, --processing-name NAME    Preprocessing name"
    echo "  -c, --trainer-config PATH     Path to trainer config JSON"
    echo ""
    echo "Description:"
    echo "  This script launches the training process for the RepeatMasker classification task."
    echo "  It assumes that the environment is already set up (e.g., conda environment activated,"
    echo "  dependencies installed, etc.)."
    echo ""
    echo "  Run this script from the 'script/' directory in the project root."
}

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) print_help; exit 0 ;;
        -t|--num-threads) NUM_THREADS="$2"; shift ;;
        -g|--gpus-per-node) GPUS_PER_NODE="$2"; shift ;;
        -m|--num-machines) NUM_MACHINES="$2"; shift ;;
        -n|--model-name) MODEL_NAME="$2"; shift ;;
        -k|--task) TASK="$2"; shift ;;
        -p|--tokenizer-path) TOKENIZER_PATH="$2"; shift ;;
        -r|--processing-name) PROCESSING_NAME="$2"; shift ;;
        -c|--trainer-config) TRAINER_CONFIG_PATH="$2"; shift ;;
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

export SCRIPT="../translast/modeling/train.py"

export SCRIPT_ARGS=" \
    $MODEL_NAME \
    --pretrained-tokenizer-path $TOKENIZER_PATH \
    --trainer-config-path $TRAINER_CONFIG_PATH \
    --preprocessing-name $PROCESSING_NAME \
    --num-workers $NUM_THREADS \
    --do-eval"

# ============================
# Run Training
# ============================

echo "Launching training with Accelerate..."
accelerate launch $ACCELERATE_ARGS $SCRIPT $TASK $SCRIPT_ARGS
