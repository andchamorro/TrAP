# Reproducing the manuscript results

This page describes the clean **k=17 / GENCODE v48 / 32k-vocab** re-do
that targets the five metric inconsistencies identified in the original k=18
manuscript results.

## Identified issues and fixes

| # | Issue | Root cause | Fix |
|---|-------|------------|-----|
| 1 | Training perplexity (34.36) dramatically worse than val (1.81) | `train_result.metrics["train_loss"]` is the running mean from step 0, not the final checkpoint | Post-training eval on fixed-masked training sample (`final_train_*` metrics) |
| 2 | Suspiciously low validation perplexity | Prior data leakage at read level | Transcript-level split (validated) |
| 3 | Missing training accuracy | `MaskingTrainer` had no `compute_metrics` | `compute_masking_metrics` wired in |
| 4 | Validation outperforms test | `load_best_model_at_end=True` selects on val | HPO targets regularisation |
| 5 | Masking inconsistency | Training: random masks; eval: fixed pre-masked | Documented; consistent eval collator |

## Reproduce on Grace

```bash
# 1. Set up environment
bash scripts/setup_conda_env.sh
conda activate trap
cp config/hpc/grace.yaml.example config/hpc/grace.yaml
# edit grace.yaml → fill in your account and email

# 2. (Optional) tune hyperparameters first
bash scripts/slurm/submit_tuning.sh

# 3. Submit full pipeline
bash scripts/slurm/submit_pipeline.sh
```

The `afterok` chain submits stages 00 → 50 sequentially; each stage writes
a `manifest.json` (git commit, seed, k, input SHA-256s, throughput).

## Reproducibility seed

The global seed is **3469**, set via `utils.seeding.set_global_seed` at the
start of every training run. It is stored in every `manifest.json` and
propagated to `TPESampler` for Optuna sweeps.

## Transcript-level split validation

`_extract_transcript_id` uses `read_id.split("|")[0]`, which correctly
extracts:

- LINE1 reads: `84120` from `84120|L1PA2|LINE-74`
- GENCODE reads: `ENST...` from `ENST...|gene_name|...`

ART's read-number suffix is in a later `|`-field and does not interfere.
This prevents read-level leakage while keeping all reads from the same
transcript in the same split.
