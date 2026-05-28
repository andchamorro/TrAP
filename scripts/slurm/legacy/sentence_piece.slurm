#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=sentencepiece     #Set the job name to "pytorch"
#SBATCH --time=04:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --cpus-per-task=38
#SBATCH --mem=32G                    #Request 2560MB (2.5GB) per task
#SBATCH --output=run_sentencepiece_%j.out  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --error=run_sentencepiece_%j.err#  #Send stdout/err to "Example4Out.[jobID]"

module load Anaconda3/2024.02-1
module load CUDA/12.6.0

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate translast

# Define the command variables
FILE_PATH="data/external/ref/gencode/gencode.v47.transcripts.fa.gz"
VOCAB_SIZE=138648
K=18
BATCH_SIZE=1024
OUT_DIR="data/processed/transcripts_sentencepiece/unigram/huggingface"
NAME="gencode.v47.transcripts.k18"

# Run the command
# python train_sentencepiece.py --file_path $FILE_PATH --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE --k $K --fast --out $OUT_DIR --name $NAME
python write_trainscrips_kmers.py --file_path $FILE_PATH --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE --k $K --fast --out $OUT_DIR --name $NAME

conda deactivate
