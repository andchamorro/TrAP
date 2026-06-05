# Testing

## Running the test suite

```bash
pytest                          # full suite
pytest -m unit                  # fast isolated tests only
pytest -m "not slow"            # skip slow tests
pytest --cov=trap               # with coverage report
```

## Test markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, isolated, no I/O — always run in CI |
| `integration` | Multi-module or filesystem access |
| `slow` | End-to-end tests (tokenizer training, full training loop) |

## Structure

Tests mirror the `trap/` package under `tests/`:

```
tests/
├── conftest.py              # shared fixtures (DNA seqs, temp FASTA/FASTQ)
├── loaders/
│   ├── test_dataset.py
│   ├── test_tokenizer.py
│   └── test_tokenizer_debug.py   # Unigram trainer panic regression tests
├── modeling/
│   ├── conftest.py                  # workflow-suite fixtures (size, datasets)
│   ├── workflow_data.py             # reusable data engine + assertions
│   ├── test_training_workflow.py    # deep training/tuning validation suite
│   ├── test_train.py
│   ├── test_tune.py
│   └── test_postprocessing.py
├── scripts/
│   └── test_divide_gff.py    # RepeatMasker GFF parser (validated vs real data)
└── utils/
    ├── test_kmer.py
    └── ...
```

## Notable test files

### `test_tokenizer_debug.py`

Eight unit tests with 50 bp synthetic reads that document and guard the
three-bug Unigram trainer panic. See {doc}`tokenizer_debug` for full details.

### `test_divide_gff.py`

Validates `scripts/divide_gff.py` against a synthetic GFF3 derived from the
real `GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript.gtf`. Counts match
the ground-truth BED exactly: L1HS = 79, L1PA2 = 392, 132 total subfamilies.

### `test_tune.py`

Tests `_build_hp_space` (Optuna trial mapping), `TuneSearchSchema` YAML
loading, and `_make_study` (in-memory TPE + Hyperband). No full training
trial is run in CI.

### `test_training_workflow.py`

Deep, fast validation of the **training and hyperparameter-tuning workflows**:
deterministic real-or-synthetic data sampling, schema-drift detection, data
leakage, model-init vocab guards, training-loop stability/convergence/metric
consistency, and a reproducible Optuna search. See
{doc}`training_workflow_tests` for the full design, assumptions, and extension
hooks.

## Running the workflow suite on Grace (HPRC)

The whole suite is CPU-only and finishes in seconds, so it runs on a small CPU
allocation. Submit `scripts/slurm/15_tests.slurm`:

```bash
sbatch scripts/slurm/15_tests.slurm                 # full suite
TEST_MARK=unit sbatch scripts/slurm/15_tests.slurm  # fast structural tests only
```

The script loads the CPU modules, activates the `trap` env, and writes JUnit +
coverage XML under `logs/`.
