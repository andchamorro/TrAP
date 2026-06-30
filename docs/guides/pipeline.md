# Pipeline walkthrough

TrAP runs as six sequential stages that map directly onto the manuscript
workflow: raw reads → tokenization → classification → abundance.

```
fetch references ─► tokenizer ─► dataset ─► MLM pretrain ─► classify ─► quantify
   (GENCODE v48)    (k=17,32k)   (ART→STAR    (ALBERT MLM    (L1HS/L1PA   (per-read
                                  →label)      + SOP)         /NEGATIVE)   abundance)
```

Each CLI is a Typer app: `python -m <module> <command> --help`.

:::{tip}
Add `--verbosity normal` to any command below to see stage start/end
notifications, elapsed time, and key metrics. Use `--verbosity detailed`
for full operational logs. See {doc}`verbosity` for the complete reference.
:::

## Stage 1 — Fetch references

Download GENCODE v48 transcripts, GRCh38.p14 genome + GTF, and build
STAR / BWA indexes:

```bash
bash scripts/data/fetch_references.sh
```

## Stage 2 — Train the tokenizer

Train a SentencePiece Unigram k-mer tokenizer (k=17, 32k vocabulary) and
wrap it as a HuggingFace `PreTrainedTokenizerFast`:

```bash
python -m trap.loaders.tokenizer train \
    --corpus data/external/gencode.v48.transcripts.fa.gz \
    --out models --name tokenizer.gencode.v48.k17.32k \
    --k 17 --vocab-size 32000
```

The k-mer entropy plateau observed above k=16 motivated the choice of k=17;
the 32k vocabulary covers ≈ 10 % of all possible 17-mers while achieving
full transcriptome coverage.

:::{note}
See {doc}`../development/tokenizer_debug` for known pitfalls and the
three-bug fix applied to `train_sentencepiece` for tokenizers ≥ 0.19.
:::

## Stage 3 — Preprocess datasets

Tokenize and split. The classification dataset uses a **transcript-level
split** so no transcript contributes reads to more than one split (prevents
read-level data leakage):

```bash
# Classification (paired reads, transcript-level split)
python -m trap.utils.preprocessing_sequences classification \
    --pretrained-model-path models/tokenizer.gencode.v48.k17.32k \
    --builder data/external/l1_R1.fq --pair data/external/l1_R2.fq \
    --k 17 --split-strategy transcript-level

# MLM masking corpus
python -m trap.utils.preprocessing_sequences masking \
    --pretrained-model-path models/tokenizer.gencode.v48.k17.32k \
    --builder data/external/gencode.v48.transcripts.fa.gz --k 17
```

## Stage 4 — MLM pretraining

Pre-train an ALBERT model with masked-language modeling (MLM) and
sentence-order prediction (SOP):

```bash
python -m trap.modeling.train masking albert.gencode.v48.k17.32k \
    --pretrained-tokenizer-path models/tokenizer.gencode.v48.k17.32k \
    --albert-config-path config/albert_config_k17_v48.json \
    --trainer-config-path config/training/mlm.json --k 17
```

For multi-GPU use: `accelerate launch -m trap.modeling.train ...`

## Stage 5 — Classification fine-tuning

Fine-tune the MLM checkpoint for three-class sequence classification
(L1HS / L1PA / NEGATIVE):

```bash
python -m trap.modeling.train classification albert.l1hs_l1pa2.v48.k17.32k \
    --pretrained-model-path albert.gencode.v48.k17.32k \
    --trainer-config-path config/training/classification_final.json \
    --k 17 --do-eval
```

## Stage 6 — Quantify a sample

Stream paired FASTQ through the classifier to produce per-read class scores,
then filter by the NEGATIVE-class score threshold before downstream
abundance estimation (Salmon / EM):

```bash
python -m trap.modeling.quantify run \
    --pretrained-model-name albert.l1hs_l1pa2.v48.k17.32k \
    --r1 sample_R1.fastq.gz --r2 sample_R2.fastq.gz \
    --output-path reports/quantify/sample --k 17 --batch-size 64

python -m trap.modeling.postprocessing filter-ids \
    --fastq sample_R1.fastq.gz \
    --output-path reports/quantify/sample --threshold 0.5
```

In the manuscript, these aggregate scores correlate with 5′RACE long-read
references (R² = 0.91 for ALBERT + Salmon) and BWA alignment counts
(r = 0.93 for L1HS).
