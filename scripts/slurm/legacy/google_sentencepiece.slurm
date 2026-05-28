#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=google_sentencepiece     #Set the job name to "pytorch"
#SBATCH --time=24:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --cpus-per-task=38
#SBATCH --mem=128G                    #Request 2560MB (2.5GB) per task
#SBATCH --output=run_google_sentencepiece_%j.out  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --error=run_google_sentencepiece_%j.err#  #Send stdout/err to "Example4Out.[jobID]"

module load GCC/12.3.0 SentencePiece/0.2.0

# Define the command variables
INPUT="gencode.v47.transcripts.k18.txt"
MODEL_PREFIX="transcripts_sentencepiece/unigram/google/gencode.v47.transcripts.k18"
MAX_SENTENCEPIECE_LENGTH=18
VOCAB_SIZE=138648
NUM_THREADS=32

# Run the command
spm_train \
  --input=$INPUT \
  --model_prefix=$MODEL_PREFIX \
  --model_type=unigram \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --normalization_rule_name=identity \
  --vocab_size=$VOCAB_SIZE \
  --max_sentencepiece_length=$MAX_SENTENCEPIECE_LENGTH \
  --max_sentence_length=500000 \
  --input_sentence_size=100000 \
  --user_defined_symbols="[CLS],[SEP],[MASK]" \
  --pad_id=3 \
  --train_extremely_large_corpus=true \
  --num_threads=$NUM_THREADS
