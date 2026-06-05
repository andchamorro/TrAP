# Hyperparameter tuning

TrAP uses **Optuna + HyperbandPruner** as a drop-in replacement for the
original Ray Tune / Population-Based Training recipe. It is:

- Pure-Python, fully offline (`HF_DATASETS_OFFLINE=1`)
- First-class in `transformers` via `Trainer.hyperparameter_search(backend="optuna")`
- Seeded (`TPESampler(seed=3469)`) for single-worker reproducibility
- Parallelisable on SLURM via job arrays sharing one study through a
  `JournalFileBackend`

## Pipeline position

Tuning is **optional** and occupies the `*5_` slots between stages:

```
20_dataset
    │
  [optional] 25_tune_mlm (array)          → 25_tune_mlm_finalize
    │
30_mlm_pretrain
    │
  [optional] 35_tune_classification (array) → 35_tune_classification_finalize
    │
40_classification
```

## Local sweep (quick start)

```bash
# Classification sweep — writes config/training/classification_final.tuned.json
python -m trap.modeling.tune classification albert.l1hs_l1pa2.v48.k17.32k \
    --search-config config/tuning/classification_optuna.yaml \
    --preprocessing-name gencode.v48.k17.32k/l1hs_l1pa2 \
    --pretrained-model-path albert.gencode.v48.k17.32k

# Then use the tuned config for the full training run
python -m trap.modeling.train classification albert.l1hs_l1pa2.v48.k17.32k \
    --trainer-config-path config/training/classification_final.tuned.json ...
```

## SLURM sweep (Grace)

```bash
bash scripts/slurm/submit_tuning.sh                   # MLM + classification
bash scripts/slurm/submit_tuning.sh --classification  # classification only
bash scripts/slurm/submit_tuning.sh --dry-run         # preview chain
```

The submit script handles `afterok` chaining: each finalize job runs only
when all array workers for its sweep complete.

## Search spaces

Search spaces are defined in `config/tuning/*.yaml`:

| File | Objective | Trials | Workers |
|------|-----------|--------|---------|
| `classification_optuna.yaml` | `eval_f1` (maximize) | 16 | 8 × 2 |
| `mlm_optuna.yaml` | `eval_loss` (minimize) | 12 | 6 × 2 |

MLM trials use a **proxy schedule** (`max_trial_epochs` in the YAML,
default 5 epochs) instead of the full 40-epoch run; Hyperband prunes
under-performing trials early.

## Manifest

Each sweep writes `config/training/manifest.json` alongside the tuned
config with full provenance:

```json
{
  "trap_version": "0.0.1",
  "git_commit": "...",
  "seed": 3469,
  "tuning": {
    "metric": "eval_f1",
    "direction": "maximize",
    "sampler": "tpe",
    "pruner": "hyperband",
    "n_trials": 16,
    "best_value": 0.95
  }
}
```
