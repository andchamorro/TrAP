# What produced each manuscript item

Maps every table and figure of *A Transformers Analysis Pipeline (TrAP) to
Evaluate LINE-1 RNA Sequence Content* to the code in this repository that
produced it, and names the code that is **not** behind any reported result.

Read it as the index to the archive: the repository is a working tree, so it
carries exploratory analyses alongside the reported ones. This file is the
boundary between the two.

Paths are repository-relative. `python -m trap.analysis <command>` refers to the
Typer app in `trap/analysis/cli.py`; `scripts/slurm/*.slurm` are the Grace HPRC
wrappers that ran each stage, and each records a run manifest (software
versions, seeds, parameters, input checksums) beside its output.

## Dataset

| Item | Code |
|---|---|
| Table 1 — dataset composition | `scripts/build_dataset_seqlabel.sh` → `scripts/slurm/22_build_seqlabel.slurm` |
| Read simulation | `scripts/sh/align_star.sh`, ART (external) |
| Label assignment | `trap/utils/rmout.py`, `trap/utils/labels.py` |
| minimap2 label cross-check | `trap/analysis/relabel_by_sequence.py` |
| Tokenization / encoding | `trap/loaders/salmon_tokenizer.py`, `trap/utils/canonical_kmer.py`, `trap/utils/kmer.py` |
| Dataset assembly | `trap/utils/preprocessing_sequences.py` → `scripts/slurm/20_dataset.slurm` |

The reported corpus is `data/processed/gencode.v48.k17.salmon.seqlabel`; its
split sizes are Table 1.

## k-mer length selection

| Item | Code |
|---|---|
| Figure 1 — entropy vs k | `trap/analysis/entropy.py`, `trap/analysis/kmer_counting.py` |
| Figure 2 — marginal information gain | `trap/analysis/selection.py` |
| Figure 3 — bias-correction magnitude | `trap/analysis/entropy.py` (Miller-Madow, Chao-Shen) |
| Figure 4 — rarefaction | `trap/analysis/entropy.py` (subsample fractions) |
| k = 17 selection report | `python -m trap.analysis run`, `config/analysis/entropy_redundancy.yaml` |

Driver: `scripts/slurm/05_entropy_redundancy.slurm`. Plots:
`notebooks/5.01-ach-entropy-redundancy-kmer-selection.ipynb`.

## Classifier

| Item | Code |
|---|---|
| Figure 5 — PBT hyperparameter search | `trap/modeling/tune.py` → `scripts/slurm/35_tune_classification.slurm`, `35_tune_classification_finalize.slurm` |
| Table 2 — validation-split metrics | `trap/modeling/train.py classification` → `scripts/slurm/40_classification.slurm` |
| Table S1 — held-out test split and confusion matrix | `trap/analysis/heldout_eval.py`, `python -m trap.analysis heldout-eval` → `scripts/slurm/41_heldout_eval.slurm`, `41b_heldout_shard.slurm` |
| Tables S11-S13 — class frequencies, weight vector, recall propagation | `trap/analysis/imbalance.py`, `python -m trap.analysis imbalance` |

Architecture: `trap/modeling/albert.py`. Evaluation notebook:
`notebooks/5.02-ach-hpo-classification-evaluation.ipynb`.

The reported negative result on masked-language-model pre-training ("this
pretraining did not improve downstream read-level classification") comes from
`trap/modeling/train.py masking` run through `scripts/slurm/legacy/mlm/`; the
diagnosis of why the hashed vocabulary makes the objective unlearnable is
`scripts/slurm/spm_gate2_mlm_learnability.slurm`. The SentencePiece comparison
reported in the tokenization Methods is `trap/loaders/tokenizer.py` and
`trap/utils/spm2hugging.py`, run through `scripts/slurm/legacy/sentence_piece.slurm`
and `legacy/train_sentencepiece.py`. These live under `legacy/` because they are
not in the final pipeline, not because they are unreported.

## Attribution and interpretation

| Item | Code |
|---|---|
| Figure 6 — integrated gradients vs attention | `trap/analysis/interpret.py` |
| Figure 7 — ORF2 C-terminal domain enrichment | `trap/analysis/enrichment.py`, `trap/analysis/_l1_align.py` |
| Figure 8 — per-read relevance | `trap/analysis/interpret.py` |
| Read-clustered bootstrap interval | `python -m trap.analysis enrichment-clustered` |

Drivers: `scripts/slurm/70_interpret.slurm`, `71_kmer_overlap.slurm`. Notebook:
`notebooks/5.07-ach-model-interpretability.ipynb`.

## Synthetic insertion benchmark

| Item | Code |
|---|---|
| Table 3 — comparator tools and settings | `config/baselines/baseline_settings.yaml` |
| Figure 9 — insertion-equivalent abundance | `scripts/python/synthetic_abundance.py` |
| Figure 10 — per-method accuracy | `notebooks/5.03-ach-synthetic-e2e-validation.ipynb` |
| Table 4, Tables S2-S5 — per-stratum concordance, paired tests, insertion-load sensitivity | `trap/analysis/synthetic_strata.py`, `python -m trap.analysis synthetic-strata` |

Grid generation: `scripts/python/generate_synthetic_dataset.py`,
`scripts/sh/generate_synthetic_dataset.sh` →
`scripts/slurm/60_generate_synthetic.slurm`. Quantification:
`scripts/slurm/61_synthetic_abundance.slurm`, `62_synthetic_validation.slurm`,
`63_run_baseline.slurm`, `64_baseline_compare.slurm`. Comparator wrappers:
`scripts/sh/run_salmon.sh`, `run_l1em.sh`, `run_htseq.sh`,
`run_tetranscripts.sh`, `run_albertem.sh`, with `scripts/sh/_l1em_reference.sh`
building L1EM's annotation and `scripts/sh/_synthetic_paths.sh` resolving the
grid tree.

## 5'RACE locus-level comparison

| Item | Code |
|---|---|
| Figure 11 — short-read vs 5'RACE per locus | `notebooks/5.06-ach-albert_exp_SRR3997504_SRR4099955.ipynb` |
| Table 5, Tables S6-S7 — leave-one-out sensitivity, within-quantifier margins | `trap/analysis/robustness.py`, `python -m trap.analysis robustness` |
| Abundance harmonization (TPM / RPM) | `trap/analysis/quantification.py` |

Stage A collector: `scripts/aggregate_srr_l1em_exp.py` (on Grace), writing
`results/srr_experiment/l1em_salmon_fpm.csv`.

## Chromosome-level comparison

| Item | Code |
|---|---|
| Figure 12 — classifier scores vs BWA counts | `notebooks/5.05-ach-albert_quantify_SRR3997504.ipynb` |
| Figure 13 — correlation matrix | `notebooks/5.04-ach-albert_quantify_SRR4099955.ipynb` |
| NEGATIVE leak rates at tau = 0.5 | `trap/analysis/negative_control.py`, `python -m trap.analysis negative-control` |
| Tables S8-S9 — correlations adjusted for chromosome length | `python -m trap.analysis chrom-length`, `references/GRCh38.chrom.sizes` |
| Table S10 — leave-one-chromosome-out jackknife | `trap/analysis/negative_control.py` |

Stage A collector: `scripts/aggregate_srr_by_chr.py`. Alignment axis:
`scripts/sh/bwaAndCount.sh`.

## Pipeline and assembly

| Item | Code |
|---|---|
| Figure 14 — workflow overview | `scripts/R/workflow_overview.R` |
| Inference and filtering at tau = 0.5 | `trap/modeling/predict.py`, `trap/modeling/quantify.py`, `trap/modeling/postprocessing.py` |
| Stage listing and submission chain | `trap/reproduce.py`, `scripts/slurm/submit_pipeline.sh` |
| Supplementary tables S1-S13 as pandoc markdown | `scripts/make_supplementary_tables.py` |

## Not behind any reported result

Present in the archive for completeness, and used by no table or figure in the
manuscript or its supplement.

| Code | What it is |
|---|---|
| `scripts/slurm/42_seed_replicates.slurm`, `43_seed_heldout.slurm`, `python -m trap.analysis seed-summary` | Retraining across seven scattered seeds, to test how stable the classifier metrics are under re-initialization. Run, but no number from it is reported. |
| `scripts/slurm/44_logreg_kmer_baseline.slurm`, `scripts/python/explore_logreg_kmer_baseline.py` | Logistic regression on hashed k-mer counts, as a simpler-model reference point. The manuscript names a systematic comparison of classification stages as future work rather than reporting one. |
| `trap/analysis/subfamily_confusion.py`, `scripts/slurm/65_subfamily_confusion.slurm` | Per-RepeatMasker-subfamily breakdown of the predictions, used to diagnose the L1PA/NEGATIVE error axis. |
| `trap/analysis/boundary_calibration.py` | Tuning the probability-to-label rule without retraining. The reported pipeline uses the fixed threshold tau = 0.5. Imported by `quantification.py`, so it is not separable from the reported path. |
| `trap/analysis/ensemble.py` | Soft-vote ensemble over the canonical-k-mer and SentencePiece models. Not used; also imported by `quantification.py`. |
| `trap/analysis/diagnostics.py`, `scripts/slurm/21_dataset_diagnosis.slurm` | Build-time validation of the stage-20 outputs (class balance, transcript leakage, warmup steps). |
| `trap/analysis/ablation.py` | Separability of the three classes across k, measured with a logistic regression on hashed k-mer frequencies as a lightweight proxy for the full model. Run as part of `python -m trap.analysis run`, but no figure reports it, and the manuscript states that the matched-capacity ALBERT ablation remains untested. |
| `scripts/sh/bowtie1AndCount.sh`, `bowtie2AndCount.sh`, `hisat2AndCount.sh`, `bwamemAndCount.sh`, `starAndCount.sh` | Alignment-and-count wrappers from earlier exploration. The reported comparisons use STAR (`align_star.sh`) and BWA (`bwaAndCount.sh`). |
| `scripts/slurm/legacy/preprocessing_sequences.slurm`, `train_repeatmasker.slurm`, `train_transcripts.slurm`, `write_trainscrips_kmers.py`, `google_sentencepiece.slurm` | Superseded by the current dataset and tokenizer stages. |
