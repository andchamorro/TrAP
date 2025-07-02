# A Transformers Analysis Pipeline (TrAP) to Evaluate Genome LINE-1 Sequence Content

**TrAP** is a natural language processing (NLP) model primarily designed for detecting and analyzing of LINE-1 (L1) retroelement sequences in genomic data. Traditional methods for identifying LINE-1 elements rely on pattern matching and sequence alignment, which often struggle to detect novel or divergent LINE-1 sequences and to distinguish between segmental duplications and true LINE-1 copies.

This project introduces an **alignment-free approach** to LINE-1 analysis using NLP and tokenization techniques, enabling more robust and scalable detection of LINE-1 retroelements in large-scale genomic datasets.

---

## Features

- Alignment-free detection of LINE-1 sequences  
- Transformer-based architecture for sequence modeling  
- Support for custom tokenizers and preprocessing pipelines  
- Multi-GPU training with mixed precision  
- Offline mode for reproducibility and secure environments  

---

## Requirements

- Python 3.8+  
- PyTorch  
- HuggingFace Transformers and Datasets  
- Accelerate  
- Anaconda (for environment management)

## Installation

To install the required dependencies for **TrAP**, we recommend using **Anaconda** for environment management.

### Step 1: Clone the repository

```bash
git clone https://github.com/andchamorro/TrAP.git
cd TrAP
```

### Step 2: Create and activate the environment

```bash
conda create -n trap python=3.10
conda activate trap
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a model using `TrAP`, use the provided script:

```bash
bash script/train_repeatmasker.sh \
    --num-threads 8 \
    --gpus-per-node 2 \
    --num-machines 1 \
    --model-name albert.l1hs_l1pa2.k18.32k \
    --task classification \
    --tokenizer-path ../data/tokenizer/fast \
    --processing-name gencode.v47.transcripts.k18.32k.skipn.nocompress/l1hs_l1pa2 \
    --trainer-config trainer_config_base_repeatmasker.json
```

You can run `bash script/train_repeatmasker.sh --help` to see all available options.


### Quantifying a Sample

#### Proprocess the genomic sample

```bash
bash script/quantify_preprocessing.sh \
    --input-file run/data/DRR494402/DRR494402_1.fastq \
    --pair-file run/data/DRR494402/DRR494402_2.fastq \
    --output-path run/quantify_k18_DRR494402 \
    --k 18 \
    --pretrained-tokenizer albert.l1hs_l1pa2.repeatmasker \
    --num-threads 32
```

You can run `bash script/quantify_preprocessing.sh --help` to view required and optional arguments.

#### Quantifying using the RepeatMasker model

To run quantification using a pretrained model with GPU acceleration:

```bash
bash script/quantify_repeatmasker.sh \
    --pretrained-model albert.l1hs_l1pa2.repeatmasker \
    --output-path run/quantify_k18_SRR4099955 \
    --batch-size 64 \
    --num-threads 8 \
    --gpus-per-node 2 \
    --num-machines 1
```

> **Note:** `--pretrained-model` and `--output-path` are required arguments.

You can also run:

```bash
bash script/quantify_repeatmasker.sh --help
```

to see all available options.


### Quantifying L1 Elements with Salmon

To build indexes and quantify L1 elements from paired-end FASTQ files:

```bash
bash script/salmon_pipeline.sh \
    --ref-genome data/genome.fa \
    --bed-file data/l1_elements.bed \
    --r1 data/sample_R1.fastq \
    --r2 data/sample_R2.fastq \
    --output-dir run/quantify_k18_SRR4099955 \
    --threads 16 \
    --force
```

This will generate `only` and `runon` indexes and perform quantification using Salmon.

This script will:

1. Use `trap/modeling/posprocessing.py` to filter read IDs based on `class_scores.pkl` in the `--output-dir`.
2. Use `seqkit` to extract matching reads from both FASTQ files.
3. Run Salmon quantification using the filtered reads.

> **Note:** The `--output-dir` must match the one used in `quantify_repeatmasker.sh` so that `class_scores.pkl` is available for filtering.