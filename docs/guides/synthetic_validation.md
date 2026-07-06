# Synthetic end-to-end validation

A ground-truth benchmark for TrAP's production pipeline (**classifier filter →
salmon quantification**). Full-length L1 elements are inserted into chr1 GENCODE
transcripts at known levels; reads are simulated with ART; each method's recovered
per-locus abundance is regressed (R²) against the known truth. It is an
*out-of-training-distribution* test — the model trained on transcriptome reads is
evaluated on genomic-insertion reads — so it probes generalization beyond the
in-distribution read-level metrics.

See the full plan in `.trap/plans/synthetic-e2e-validation.md`.

## 0. Directory naming (by experiment type)

The reference trees are named by **experiment type** (derived from `SIM_MODEL`), grouped
under a `synthetic/` parent. The single resolver `scripts/sh/_synthetic_paths.sh`
(`experiment_dir` / `experiment_results_dir`, mirrored in
`scripts/python/synthetic_abundance.py`) is the one place this mapping lives — every
synthetic script sources it rather than re-deriving the path:

| `SIM_MODEL` | model | experiment | reference dir | results dir |
|---|---|---|---|---|
| `transcript` | 2 | standalone L1 transcript pool (no host background) | `data/ref/synthetic/l1-transcript-pool.<l1source>` | `results/synthetic_validation/l1-transcript-pool/` |
| `insert` | 1 | full-length L1 spliced into host chr1 transcripts (with background) | `data/ref/synthetic/l1-host-insert.<l1source>` | `results/synthetic_validation/l1-host-insert/` |

The `l1source` (`l1base` / `rm`) stays a trailing qualifier so both sources coexist;
`withdel` and `chr1` are fixed for this benchmark and are recorded in the manifest instead
of the directory name.

**Migrating an existing tree.** The resolver falls back to the pre-refactor name
(`data/ref/GRCh38.p14.genome.chr1.withdel.<l1source>[.insert]`) whenever the new directory
is absent, so unmoved trees keep working. To adopt the new layout, do the one-time move
once (gitignored, so `git` is unaffected):

```bash
mkdir -p data/ref/synthetic
mv data/ref/GRCh38.p14.genome.chr1.withdel.l1base        data/ref/synthetic/l1-transcript-pool.l1base
mv data/ref/GRCh38.p14.genome.chr1.withdel.l1base.insert data/ref/synthetic/l1-host-insert.l1base
```

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

Outputs to the experiment's reference dir (§0, e.g. `data/ref/synthetic/l1-host-insert.l1base/`):
- `art/GRCh38.p14.chr1.insert_level_<power>_delprob_<dp>.pair.5x{1,2}.fq.gz` — reads
- `GRCh38.p14.chr1.insert_level_<power>_delprob_<dp>.bed` — insertion ground truth
- `l1_synthetic.Index` — salmon index built from the inserted L1 elements

**Grid:** insertion level `2^5..2^13` × `del_prob {0.000,0.025,0.050,0.075,0.100}` (45
samples). Override via `POWERS=`, `DELPROBS=`, `FCOV=`, `SEED=`, `L1_MIN_LEN/L1_MAX_LEN=`,
`CHR1_TRANSCRIPTS=` (skip the GTF subset), `OUTPUT_DIR=`.

The insertion simulation (`scripts/python/generate_synthetic_dataset.py`, ported from
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
# one sample (single-cell mode):
POWER=8 DELPROB=0.025 sbatch scripts/slurm/synthetic_validation.slurm
# full grid as a per-cell SLURM array (recommended):
bash scripts/slurm/submit_synthetic_validation.sh
#   bash scripts/slurm/submit_synthetic_validation.sh --dry-run   # preview the array
```

## 3. Figures (R/ggplot)

Per the manuscript figure convention, a Python step emits
`results/synthetic_validation/<experiment>/abundance.csv` (§0, e.g.
`l1-host-insert/`) and an R notebook
(`notebooks/5.03-ach-synthetic-e2e-validation.ipynb`, `ir-trap` kernel,
`data.table` + `ggplot2`) renders the R² comparison to
`reports/figures/synthetic_validation/`.

## 4. Reproduction from the archived source run (option A, 2026-07-04)

`synthetic_salmon_l1em_teht_some` was reproduced **from the archived source-run artifacts**
(fetched with `.trap/scripts/fetch_collect_synthetic_host_insert.sh`), so **no new
experiment was needed**. The archived tree lays methods out as `salmon/art/` (plain →
Salmon) and `salmon/filtered/` (classifier-filtered → AlbertSalmon), with TEtranscripts
`*.cntTable` lacking the `GRCh38.p14.` prefix; the collector now resolves these via subdir
candidates + a filename glob (see `SALMON_UNFILTERED_SUBDIRS`/`SALMON_FILTERED_SUBDIRS`).

**Observed per-element R² (all 45 cells pooled, `abundance` vs `simulated` copies):**

| Method | R² | Published target |
|---|---|---|
| AlbertSalmon | **0.99** | ≈ 0.99 ✓ |
| Salmon | **0.94** | ≈ 0.95 ✓ |
| TEtranscripts | 0.94 | — |
| HTseq | 0.88 | — |
| AlbertEM | 0.85 | — |
| L1EM | 0.84 | — |

AlbertSalmon ≫ Salmon per element reproduces the manuscript result. At the per-**cell**
total level both are near-perfect (Salmon 0.9998, AlbertSalmon 0.9985) and tie — total-R²
spans 2⁵–2¹³ so it is a weak discriminator; the notebook reports it but no longer hard-fails
on the order. The archive has no `_inputs/l1_fulllength.bed`, so the legacy `--normalize`
per-element axis (`simulated_reads`) is unavailable; the raw-copy axis gives the R² above and
matches the published grid, so normalization was not required.

## Caveat — reuse of legacy baselines

The legacy generator set no random seed, so this re-creation is a **new, seeded
draw**. Compare methods at the **R²-vs-Simulated** level (a stable property of the
generation process), not read-for-read. The seqlabel `AlbertSalmon` column quantifies
against `l1_synthetic.Index` (the inserted elements); legacy L1EM/Salmon results used a
different element set and are therefore comparable only at the method/R² level.
