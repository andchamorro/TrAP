# Verbosity and logging

Every TrAP CLI command accepts a `--verbosity` flag that controls how much the
pipeline writes to the terminal. The default is `off` — only completion notices,
warnings, and errors — which keeps HPC job logs uncluttered. Use `normal` for
stage-level progress and `detailed` for full operational visibility without
debug-level noise.

## Levels

| Level | Loguru floor | What you see |
|---|---|---|
| `off` *(default)* | `SUCCESS` (25) | Completion messages, warnings, errors. |
| `normal` | `STAGE` (23) | + Stage start/end, elapsed time, record counts, key metrics, device info. |
| `detailed` | `INFO` (20) | + All `logger.info()` output: subsample decisions, dataset dimensions, substep progress, model-selection paths. No `DEBUG` (10) noise. |

The custom `STAGE` loguru level (no. 23) sits between `INFO` and `SUCCESS`.
It is only emitted by stage lifecycle calls inside TrAP — not by third-party
libraries — so `normal` output remains compact even with verbose HuggingFace
callbacks enabled.

## Configuring verbosity

### CLI flag

```bash
python -m trap.loaders.tokenizer train \
    --corpus data/external/gencode.v48.transcripts.fa.gz \
    --out models --name tokenizer.gencode.v48.k17.32k \
    --k 17 --vocab-size 32000 \
    --verbosity normal
```

### Environment variable

Set `TRAP_VERBOSITY` in the current shell (or in a SLURM script) to apply
the level to every subsequent command without repeating the flag:

```bash
export TRAP_VERBOSITY=normal

python -m trap.utils.preprocessing_sequences masking ...
python -m trap.modeling.train masking ...
python -m trap.modeling.predict quantify ...
```

The `trap.reproduce submit` and `trap.reproduce tune` commands also forward
`TRAP_VERBOSITY` to all child SLURM stages via the process environment.

### Config file

Add a `verbosity` field to any config file read by a schema:

```json
{
  "verbosity": "normal",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 16
}
```

Supported schemas: `TokenizerConfigSchema`, `TrainerConfigSchema`,
`DatasetBuildSchema`, `QuantifyConfigSchema`.

### Programmatic API

```python
from trap.config.verbosity import set_verbosity, get_verbosity, VerbosityLevel

set_verbosity("normal")          # string
set_verbosity(VerbosityLevel.DETAILED)  # enum

if get_verbosity() is VerbosityLevel.DETAILED:
    ...
```

## Expected output at each level

### `--verbosity off` (default)

Only the final completion message:

```
2026-05-31 12:00:42 | SUCCESS | Tokenizer + manifest written to models/tokenizer.gencode.v48.k17.32k
```

### `--verbosity normal`

Stage lifecycle events with elapsed time and key parameters:

```
2026-05-31 12:00:00 | STAGE   | [tokenizer:train] algorithm=unigram k=17 vocab_size=32,000
2026-05-31 12:00:00 | STAGE   | [tokenizer:train] corpus loaded — 87,324 sequences
2026-05-31 12:00:00 | STAGE   | [tokenizer:train] training started — 87,324 seqs
2026-05-31 12:00:42 | STAGE   | [tokenizer:train] done — elapsed=42.1 s → models/tokenizer.gencode.v48.k17.32k
2026-05-31 12:00:42 | SUCCESS | Tokenizer + manifest written to models/tokenizer.gencode.v48.k17.32k
```

For preprocessing:

```
2026-05-31 12:01:00 | STAGE   | [preprocessing:classification] builder='data/external/l1_R1.fq' k=17 split_strategy='transcript-level'
2026-05-31 12:01:03 | STAGE   | [preprocessing:classification] split sizes — train=9,842, test=1,230
2026-05-31 12:01:45 | STAGE   | [preprocessing:classification] done — elapsed=45.3 s labels=['L1HS', 'L1PA', 'NEGATIVE']
2026-05-31 12:01:45 | SUCCESS | Classification dataset saved to data/processed/.../classification/
```

For training:

```
2026-05-31 12:05:00 | STAGE   | [train:masking] model=albert.gencode.v48.k17.32k preprocessing=gencode.v47.transcripts.k18.skipn.nocompress
2026-05-31 12:05:00 | STAGE   | [train:masking] model parameters: 11M
2026-05-31 12:05:01 | STAGE   | [train:masking] dataset — train=100,000 eval=10,000
2026-05-31 12:05:01 | STAGE   | [train:masking] device=cuda
2026-05-31 12:05:02 | STAGE   | [train:masking] training started
...
2026-05-31 14:23:10 | STAGE   | [train:masking] done — loss=2.1234 elapsed=8288 s → models/albert.gencode.v48.k17.32k/final/
2026-05-31 14:23:10 | SUCCESS | Train model done.
```

For quantification:

```
2026-05-31 14:30:00 | STAGE   | [predict:quantify] model=albert.l1hs_l1pa2.v48.k17.32k out=reports/quantify/sample
2026-05-31 14:30:01 | STAGE   | [predict:quantify] model parameters: 11M
2026-05-31 14:30:05 | STAGE   | [predict:quantify] done — rank=0 elapsed=5 s → reports/quantify/sample/scores_0_0.parquet
2026-05-31 14:30:05 | INFO    | Scores written to reports/quantify/sample/scores_0_0.parquet
```

### `--verbosity detailed`

Everything in `normal`, plus all `INFO`-level detail (subsample guards, data
collator setup, tokenizer choices, etc.). Third-party library output (HuggingFace
Trainer progress bars, tqdm) is unaffected — it is controlled by the libraries
themselves.

## Performance

The verbosity filter is a single integer comparison applied to every log record.
Overhead at `off` (the default) is negligible: the record is discarded immediately
before any string formatting occurs.
