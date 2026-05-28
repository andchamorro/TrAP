#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=preprocessing_sequences     #Set the job name to "pytorch"
#SBATCH --time=00:15:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G                    #Request 2560MB (2.5GB) per task
#SBATCH --output=run_preprocessing_sequences_%j.out  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --error=run_preprocessing_sequences_%j.err#  #Send stdout/err to "Example4Out.[jobID]"

module load Anaconda3/2024.02-1

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate translast

# Define the command variables
# FILE_PATH="/scratch/user/achamorro/ref/gencode/gencode.v47.transcripts.fa.gz"
BUILDER="RepeatMasker/GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript/art/GRCh38.p14.rmtranscriptome.pair.l1hs_l1pa2_negative.5x_1.fq"
PAIR="RepeatMasker/GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript/art/GRCh38.p14.rmtranscriptome.pair.l1hs_l1pa2_negative.5x_2.fq"
MODEL_PATH="transcripts_sentencepiece/unigram/kmerpiece/gencode.v47.transcripts.k18.32k.skipn.nocompress/huggingface/fast"
K=18
MAX_POSITION=1024
OUT_DIR="preprocessing_k18.32k.skipn.nocompress/l1hs_l1pa2"

# Run the command
# python preprocessing_sequences.py masking --pretrained_model_path $MODEL_PATH --builder $FILE_PATH --k $K --chunk_size 512 --preprocessing_dataset $OUT_DIR --num_proc $SLURM_CPUS_PER_TASK
python preprocessing_sequences.py classification --pretrained_model_path $MODEL_PATH --builder $BUILDER --pair $PAIR --file_format "fastq" --k $K --max_position_embeddings $MAX_POSITION --test_split 0.20 --val_split 0.20 --preprocessing_dataset $OUT_DIR --num_proc $SLURM_CPUS_PER_TASK

conda deactivate
