#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train_transcripts_k18     #Set the job name to "pytorch"
#SBATCH --time=24:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                    #Request 2560MB (2.5GB) per task
#SBATCH --output=run/train_transcripts_k18_%j.out  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --error=run/train_transcripts_k18_%j.err#  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request GPUs
#SBATCH --partition=gpu              #Request the GPU partition/queue

module load Anaconda3/2024.02-1
module load GCCcore/13.2.0 CUDA/12.4.1 NCCL/2.20.5-CUDA-12.4.1 

#########################
### ACTIVATE ANACONDA ###
#########################
eval "$(conda shell.bash hook)"
conda activate translast

######################
### Set enviroment ###
######################
export HF_HOME=$SCRACH/.cache/hugginface
export HF_LOCAL_HOME=HF_LOCAL
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GPUS_PER_NODE=2
export ACCELERATE_DIR="${ACCELERATE_DIR:-run/accelerate}"
export ACCELERATE_ARGS="--multi_gpu \
	--num_machines 1 \
	--num_processes=2 \
	--dynamo_backend no \
	--mixed_precision fp16"

export SCRIPT="translast/modeling/train.py"
export MODEL_NAME="albert.gencode.v47.transcripts.k18.32k"
export TOKENIZER_PATH="transcripts_sentencepiece/unigram/kmerpiece/gencode.v47.transcripts.k18.32k.skipn.nocompress/huggingface/fast"
export PROCESSING_NAME="gencode.v47.transcripts.k18.32k.skipn.nocompress"
export TRAINER_CONFIG_PATH="trainer_config_fsdp.json"
export NUM_THREADS=16

export SCRIPT_ARGS=" \
    $MODEL_NAME \
    --pretrained-tokenizer-path $TOKENIZER_PATH \
    --trainer-config-path $TRAINER_CONFIG_PATH \
    --preprocessing-name $PROCESSING_NAME \
    --num-workers $NUM_THREADS "
######################

# run jobstats in the background (&) to monitor resource usage
jobstats -o run &
# Run the command
accelerate launch $ACCELERATE_ARGS $SCRIPT $SCRIPT_ARGS 

# run jobstats to create graphs of resource usage for this job
jobstats -o run
conda deactivate
