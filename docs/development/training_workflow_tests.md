# Training & Tuning Workflow Validation Suite

A deep, fast, CI-friendly suite that validates the **training** and
**hyperparameter-tuning** workflows end to end. It lives in
`tests/modeling/` and is built to catch the failure classes that have actually
bitten this project — embedding-table size mismatches, data leakage, unstable
or non-converging training, mis-specified search spaces, and silent metric
drift — while running in seconds.

## Files

| File | Role |
|------|------|
| `tests/modeling/workflow_data.py` | Reusable **data engine** + assertions (not collected by pytest). |
| `tests/modeling/conftest.py` | Fixtures: `size`, `synthetic_classification`, `synthetic_mlm`, `workflow_dataset`. |
| `tests/modeling/test_training_workflow.py` | The test classes (sections 1–7 below). |

## Data strategy: real-first, synthetic-fallback

```
build_dataset(size, seed, prefer_real=True)
  ├─ real data on disk?  → seeded, stratified, column-normalised sample
  └─ otherwise           → synthetic, structurally-equivalent dataset
```

* **Structural checks** (schema, leakage, id bounds) prefer a small **seeded,
  stratified sample of the real** `data/processed/*/classification/tokenized`
  dataset, so they exercise the true column set and dtypes. If no processed data
  is present (fresh clone / CI), an equivalent **synthetic** dataset with the
  identical schema (`MODEL_COLUMNS`) is generated instead.
* **Learning-dynamics checks** (convergence, metric consistency, reproducibility,
  HPO) **always** use synthetic data. The generator makes each label draw tokens
  from a **disjoint band**, so the classes are linearly separable and a tiny
  1-layer ALBERT provably converges in a few epochs. Real data is *not* used here
  because a tiny model is not guaranteed to converge on it — that would make the
  assertions flaky rather than meaningful.

### Determinism & reproducibility

* Every generator takes an explicit `seed` and uses a local NumPy `Generator`
  (never the global RNG). Same seed → bit-identical rows, guarded by
  `fingerprint()`.
* Splits are **transcript-level** (reusing the production
  `trap.utils.preprocessing_sequences._transcript_level_split`), so there is no
  read-level leakage by construction.
* `write_debug_artifact()` dumps a JSON fingerprint (source, size, split sizes,
  per-label counts, sha256) to aid debugging a failing run.

### Configurable size (speed vs. signal)

`SizeConfig` controls every dimension. Presets are selectable without code
changes via an environment variable:

```bash
pytest tests/modeling/test_training_workflow.py            # tiny (CI default)
TRAP_TEST_DATASET_SIZE=small  pytest tests/modeling/...     # more signal, still CPU-fast
TRAP_TEST_DATASET_SIZE=medium pytest tests/modeling/...     # stress / profiling
```

## What each section guards

1. **Data engine** — deterministic seeding, different-seed divergence,
   size scaling, label stratification, real→synthetic fallback, debug artifact.
2. **Schema regression** — synthetic and (if present) real samples match the
   canonical `MODEL_COLUMNS`; a removed/renamed/retyped column fails loudly.
3. **Preprocessing validation** — no transcript leakage across splits, every
   token id `< vocab_size` (the embedding-gather OOB guard), consistent field
   lengths; includes a *negative* test that injects a leak and asserts it is caught.
4. **Model initialization** — `make_masking_model_init` / `make_classification_model_init`
   honour the `vocab_size` override (regression for the SalmonKmerTokenizer OOB),
   embeddings cover every token id, weights are deterministic under a fixed seed.
5. **Classification training loop** — losses finite at every step (instability),
   loss decreases (convergence), eval metrics in `[0, 1]`, accuracy and micro-F1
   agree (consistency), and two same-seed runs reproduce the eval loss.
6. **MLM training loop** — finite eval loss and masked-token accuracy in `[0, 1]`.
7. **Hyperparameter search** — a real 2-trial Optuna search runs and is
   reproducible under a fixed seed; mis-specified samplers/pruners and invalid
   `HPSpec` fields raise clear errors.

## Running

```bash
pytest tests/modeling/test_training_workflow.py -m unit    # structural, ~5 s
pytest tests/modeling/test_training_workflow.py -m slow    # training + HPO, ~3 s
pytest tests/modeling/test_training_workflow.py            # everything, ~7 s
```

All training runs are forced onto CPU (`use_cpu=True`) with dropout disabled for
hermetic, deterministic CI behaviour — no GPU, network, or bf16 dependence.

## Extending the suite

* **New failure mode** → add an assertion helper in `workflow_data.py` and a
  test that exercises it (plus a negative test that proves it fires).
* **Larger / harder data** → add a preset to `_PRESETS`; tests pick it up via
  `TRAP_TEST_DATASET_SIZE`. Keep the CI default (`tiny`) under ~10 s total.
* **New schema** → update `MODEL_COLUMNS` / `assert_model_schema`; the schema
  test will then enforce the new contract and flag drift.

## Assumptions

* The real tokenized classification dataset schema is
  `{label: ClassLabel, input_ids, token_type_ids, attention_mask}` (verified
  against `data/processed/*/classification/tokenized`). If preprocessing changes
  this, update `MODEL_COLUMNS` — section 2 will fail until you do, by design.
* Synthetic separability assumes ≥ 3 token bands fit in `vocab_size - 5`
  specials; `SizeConfig` presets satisfy this.
