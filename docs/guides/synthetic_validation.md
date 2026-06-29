# Synthetic end-to-end validation

A ground-truth benchmark for TrAP's production pipeline (**classifier filter →
salmon quantification**). Full-length L1 elements are inserted into chr1 GENCODE
transcripts at known levels; reads are simulated with ART; each method's recovered
per-locus abundance is regressed (R²) against the known truth. It is an
*out-of-training-distribution* test — the model trained on transcriptome reads is
evaluated on genomic-insertion reads — so it probes generalization beyond the
in-distribution read-level metrics.

See the full plan in `.trap/plans/synthetic-e2e-validation.md`.

## 1. Generate the dataset (once, on a compute node)

Heavy step (ART × 45 grid cells). Needs `art_illumina`, `seqkit`, `bedtools`,
`samtools`, `salmon`, and BioPython. Everything is sourced from `data/external`:

| Input (`data/external/`) | Role |
|---|---|
| `GRCh38.p14.genome.fa.gz` | genome — chr1 + L1-element extraction |
| `GCF_000001405.40_GRCh38.p14_rm.LINE1.promoter.bed` | full-length L1 coords (filtered to ≈6 kb, L19088.1-promoter) — the inserts + salmon index |
| `gencode.v48.transcripts.fa.gz` (+ a gencode GTF) | chr1 transcripts — the insertion reference |

```bash
# SLURM array (recommended): a prep job builds the shared inputs once, then a
# per-cell array (afterok) generates the grid in parallel. Resumable.
bash scripts/slurm/submit_generate_synthetic.sh
#   bash scripts/slurm/submit_generate_synthetic.sh --dry-run   # preview the prep→array chain

# or sequentially on one interactive node:
#   source scripts/slurm/_common.sh && load_bio_modules && activate_trap   # salmon is in the trap env
#   module load SeqKit/2.9.0                                               # seqkit from a module
#   bash scripts/sh/generate_synthetic_dataset.sh
```

Outputs to `data/ref/GRCh38.p14.genome.chr1.withdel/`:
- `art/GRCh38.p14.chr1.insert_level_<power>_delprob_<dp>.pair.5x{1,2}.fq.gz` — reads
- `GRCh38.p14.chr1.insert_level_<power>_delprob_<dp>.bed` — insertion ground truth
- `l1_synthetic.Index` — salmon index built from the inserted L1 elements

**Grid:** insertion level `2^5..2^13` × `del_prob {0.000,0.025,0.050,0.075,0.100}` (45
samples). Override via `POWERS=`, `DELPROBS=`, `FCOV=`, `SEED=`, `L1_MIN_LEN/L1_MAX_LEN=`,
`CHR1_TRANSCRIPTS=` (skip the GTF subset), `OUTPUT_DIR=`.

The insertion simulation (`scripts/sh/generate_synthetic_dataset.py`, ported from
notebook 4.07) applies 5′-biased truncation, internal deletions (`del_prob`), point
mutations, and TSDs. It is **deterministic** (fixed `--seed`) — unlike the legacy
generator, which set none.

## 2. Run the filter → salmon validation

Per sample: TrAP `predict.processing-dataset` → `predict.quantify` →
`postprocessing.filter-ids` (`P(NEGATIVE)<τ`) → `seqkit grep` → `salmon quant`
against `l1_synthetic.Index`. The reads are parsed twice (tokenise + extract), so the
validation **prefers uncompressed `.fq`** for speed: it uses `…pair.5x{1,2}.fq` if
present, else decompresses `.fq.gz` once into `WORK` (`DECOMPRESS=1`, default; set
`DECOMPRESS=0` to stream the `.gz`). Generate uncompressed reads up front with
`GZIP=0` to skip decompression entirely.

```bash
POWER=8 DELPROB=0.025 sbatch scripts/slurm/synthetic_validation.slurm   # one sample
# fan out:
for p in $(seq 5 13); do for d in 0.000 0.025 0.050 0.075 0.100; do
  POWER=$p DELPROB=$d sbatch scripts/slurm/synthetic_validation.slurm
done; done
```

## 3. Figures (R/ggplot)

Per the manuscript figure convention, a Python step emits
`results/synthetic_validation/abundance.csv` and an R notebook
(`notebooks/5.02-ach-synthetic-e2e-validation.ipynb`, `ir-trap` kernel,
`data.table` + `ggplot2`) renders the R² comparison to
`reports/figures/synthetic_validation/`.

## Caveat — reuse of legacy baselines

The legacy generator set no random seed, so this re-creation is a **new, seeded
draw**. Compare methods at the **R²-vs-Simulated** level (a stable property of the
generation process), not read-for-read. The seqlabel `AlbertSalmon` column quantifies
against `l1_synthetic.Index` (the inserted elements); legacy L1EM/Salmon results used a
different element set and are therefore comparable only at the method/R² level.
