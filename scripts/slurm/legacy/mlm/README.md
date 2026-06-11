# Archived: MLM pre-training stages (Track A)

These SLURM stage scripts implement the **masked-language-modeling (MLM)
pre-training** phase of the original four-stage pipeline. They are **shelved**,
not deleted, as of the Track A decision (2026-06-10).

## Why they were retired

A 40-epoch / ~35-hour MLM run on Grace (job `18746482`) completed cleanly but
**never learned**: train loss sat at `ln(vocab) = 11.09` and masked-token
accuracy stayed ≈ 0. Root cause (full diagnosis in
`.trap/summaries/refactor-update_core_improve_repro.md` §8.5 and
`.trap/plans/mlm-pretraining-freeze-action-plan.md` §0):

- The Salmon k=17 tokenizer feature-hashes canonical k-mers into 65 536 buckets
  with `splitmix64`, a non-invertible avalanche hash. Adjacent token **ids** are
  uncorrelated (r ≈ −0.03) even though adjacent reads share 16/17 bases.
- Predicting a masked bucket from neighbouring buckets is therefore not a
  representable function, so the MLE-optimal predictor is the marginal token
  distribution → loss freezes at `H(unigram) ≈ ln(vocab)`. This is **structural**
  (confirmed by a fixed-mask memorisation run that also stalled, grad-norm
  collapsing 1.01 → 4e-4), not a wiring or hyperparameter bug.

**Track A** drops MLM entirely and fine-tunes the classifier from random init.
The Salmon tokenizer's separation power is exercised directly by the
classification head, which does not require a pre-trained encoder.

## Shelved files

| File | Role |
|---|---|
| `24_mlm_smoke.slurm` | MLM pre-flight smoke gate (also has a known baseline bug: checks `ln(vocab)` instead of `H(marginal)`) |
| `25_tune_mlm.slurm` | Optuna MLM hyperparameter sweep (array) |
| `25_tune_mlm_finalize.slurm` | Writes `config/training/mlm.tuned.json` |
| `30_mlm_pretrain.slurm` | Full multi-GPU MLM pre-training |

Related configs are **left in place** (not moved) because they are still wired
into dormant MLM plumbing that the rollback strategy keeps re-enableable:

- `config/training/mlm.json`, `mlm.smoke.json`, `mlm.smoke.fp32.json`, `mlm.tuned.json`
- `config/tuning/mlm_optuna.yaml`
- `config/runs/default.yaml` (`mlm_trainer_config:`)
- the `masking` / MLM sub-commands in `trap/modeling/train.py` and `trap/modeling/tune.py`

## Re-enabling (only if MLM becomes load-bearing — "Track B")

Track B would require changing the MLM **objective** (predict the masked k-mer's
nucleotide content) and **representation** (a compositional k-mer embedding
instead of an avalanche-hashed lookup), since the hashed target is unlearnable by
construction. To re-run the existing scripts as-is, move them back into
`scripts/slurm/` and re-add `24_mlm_smoke.slurm` / `30_mlm_pretrain.slurm` to the
`STAGES` array in `scripts/slurm/submit_pipeline.sh`. Note the smoke-gate baseline
bug above must be fixed first or it will green-light another unigram-collapsed run.
