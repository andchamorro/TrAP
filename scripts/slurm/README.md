# TrAP training reproduction pipeline — Grace HPRC

Phase-3 of `.trap/plans/perf-and-reproducibility-plan.md` (§6): a config-driven,
manifest-stamped pipeline for the clean **k=17 / GENCODE v48 / 32k-vocab** re-do,
orchestrated as an `sbatch --dependency=afterok` chain on the TAMU **Grace**
cluster (Cascade Lake; GPU stages default to **2× A100 40 GB**).

```
00_fetch_references       CPU   GENCODE v48 + GRCh38.p14 download, STAR/BWA indexes
        │
10_tokenizer              CPU   SentencePiece Unigram k-mer tokenizer (k=17, vocab=32k)
        │
20_dataset                CPU   ART(-f 5) → STAR(multimap 100) → bedtools → label;
        │                        classification dataset (transcript-level split) +
        │                        GENCODE masking dataset for MLM
        │
24_mlm_smoke              GPU¹  pre-flight GATE: train on a 1k-row subset, FAIL the
        │                        chain unless MLM loss drops below ln(vocab) — catches
        │                        a config that cannot learn before the multi-day run
        │
   [optional]─┬─ 25_tune_mlm (array)       GPU   Optuna+Hyperband MLM sweep (6 workers)
              └─ 25_tune_mlm_finalize       CPU   write config/training/mlm.tuned.json
        │
30_mlm_pretrain           GPU   ALBERT masked-LM pretraining
        │
34_classification_smoke   GPU¹  pre-flight GATE: fine-tune on a 1k-row subset, FAIL
        │                        unless loss drops below ln(num_labels)
        │
   [optional]─┬─ 35_tune_classification (array)  GPU   Optuna+Hyperband cls sweep (8 workers)
              └─ 35_tune_classification_finalize  CPU   write classification_final.tuned.json
        │
40_classification         GPU   fine-tune L1HS / L1PA / NEGATIVE on the MLM checkpoint
        │
50_benchmark              GPU   streaming `quantify` throughput benchmark + manifest
```

¹ The smoke gates run on a **single GPU** (plain `python`, no DDP) so the reported
loss is free of the multi-process aggregation artifact and the threshold check is exact.

Each stage writes a `manifest.json` (git commit, seed, k, SHA256s, throughput)
to its output directory.

## Quick start

```bash
cd <repo root>            # stages resolve paths from the repo root
mkdir -p logs

# submit the whole chain (00 → 50)
bash scripts/slurm/submit_pipeline.sh
#   or: python -m trap.reproduce submit

# preview the sbatch chain without submitting
bash scripts/slurm/submit_pipeline.sh --dry-run

# run a sub-range / resume after a failure
bash scripts/slurm/submit_pipeline.sh --from 30_mlm_pretrain
bash scripts/slurm/submit_pipeline.sh --from 20_dataset --to 40_classification

# list the stages
python -m trap.reproduce stages
```

Submit a single stage directly (from the repo root, so `$SLURM_SUBMIT_DIR` is correct):

```bash
# GPU stages require --gres and --partition since they are not in the .slurm file
sbatch --gres=gpu:a100:2 --partition=gpu scripts/slurm/30_mlm_pretrain.slurm
# CPU stages work as-is
sbatch scripts/slurm/10_tokenizer.slurm
```

## Pre-flight smoke gates (`24_mlm_smoke`, `34_classification_smoke`)

These two stages are **gates**, not training stages: they exist to fail *fast and
loud* when a training config cannot learn, so a misconfiguration never reaches a
multi-day GPU run.

### Why they exist

An MLM pretraining run once executed for **35 hours** and produced a checkpoint
that had learned *nothing* — the loss sat at the uniform-random baseline
(`ln(vocab)`) for all 40 epochs and the masked-token accuracy was frozen across
every epoch. The model weights never moved from their random initialisation. The
root cause was a learning rate appropriate for *fine-tuning* (`5e-5`) being used
for *from-scratch* pretraining (which needs `~1e-4..5e-4`), and the failure was
invisible because:

- the per-step loss is logged but nobody watches 40 epochs of it, and
- the Optuna sweep that "tuned" the LR was **also** stuck at the uniform baseline,
  so it optimised pure noise and returned a meaningless config.

A smoke gate turns "35 h wasted, discovered days later" into "fails in ~10 min,
stops the chain immediately".

### What they check

Each gate trains a few hundred steps on a **1k-row `--debug` subset**, on a
**single GPU** (so the reported loss has no DDP `×num_processes` aggregation
artifact), then asserts the loss dropped a clear margin below the uniform-random
baseline. If it did not, the job exits non-zero and the `afterok` dependency
**stops the whole pipeline** before the expensive stage runs.

| Gate | Runs before | Pass condition | Default margin |
|---|---|---|---|
| `24_mlm_smoke` | `25_tune_mlm`, `30_mlm_pretrain` | `final_train_loss < ln(vocab) − margin` | `1.0` nats |
| `34_classification_smoke` | `35_tune_classification`, `40_classification` | `train_loss < ln(num_labels) − margin` | `0.2` nats |

`24_mlm_smoke` needs only the tokenizer (stage 10) and dataset (stage 20).
`34_classification_smoke` fine-tunes from the MLM checkpoint, so it sits *after*
stage 30. Both are wired into `submit_pipeline.sh` automatically — no extra flags.

### Running a gate on its own

```bash
# MLM gate (validates config/training/mlm.json end-to-end on real data)
RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/24_mlm_smoke.slurm

# Classification gate (needs an MLM checkpoint from stage 30)
RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/34_classification_smoke.slurm
```

A pass prints e.g. `[24_mlm_smoke] PASS: MLM training reduces loss below the
uniform baseline.`; a failure prints the measured loss, the threshold, and the
first things to check (learning rate, bf16 underflow, data/label alignment).

### Configs and tunables

The gates run dedicated short configs so they stay fast and deterministic; the
key hyperparameters (notably the from-scratch LR) match the real configs:

| File | Used by | Notes |
|---|---|---|
| `config/training/mlm.smoke.json` | `24_mlm_smoke` | 3 epochs, `lr=5e-4` (from-scratch), no save/eval |
| `config/training/classification.smoke.json` | `34_classification_smoke` | 3 epochs, fine-tuning LR |

Override per submission via the environment:

```bash
MLM_SMOKE_CONFIG=config/training/my.smoke.json MLM_SMOKE_MARGIN=2.0 \
    sbatch scripts/slurm/24_mlm_smoke.slurm
CLS_SMOKE_MARGIN=0.3 sbatch scripts/slurm/34_classification_smoke.slurm
```

### Fast CPU layer (`pytest`)

The on-GPU gates validate the *real configs on real data (bf16)*. A complementary
CPU test validates the *code path (fp32)* in ~2 s — it trains a tiny
`AlbertForMaskedLM` / `AlbertForSequenceClassification` on a trivially learnable
dataset using the real collator/model classes and asserts the loss falls below
`ln(vocab)`:

```bash
pytest tests/modeling/test_smoke_training.py -v
```

Together they cover both failure hypotheses: LR-too-low (caught on GPU with the
real config) and bf16 update underflow (the CPU test runs fp32, the GPU gate runs
bf16, so a divergence between them isolates the precision path).

## HPC configuration

Module versions, personal settings (account, email), and GPU hardware are all
defined in **`config/hpc/grace.yaml`** — not hardcoded in the `.slurm` files.
The config is read at submission time by `submit_pipeline.sh` /
`submit_tuning.sh` and at job run-time by `_common.sh`.

### First-time setup

```bash
cp config/hpc/grace.yaml.example config/hpc/grace.yaml
# Edit grace.yaml — fill in your account and email
```

`config/hpc/grace.yaml` is **gitignored** so personal settings are never committed.
`config/hpc/default.yaml` (committed) provides the generic fallback used when
`grace.yaml` doesn't exist.

### The config files

| File | Committed | Purpose |
|---|---|---|
| `config/hpc/default.yaml` | ✓ | Generic defaults, no personal info |
| `config/hpc/grace.yaml.example` | ✓ | Grace template with placeholders |
| `config/hpc/grace.yaml` | ✗ gitignored | Your personal Grace settings |

`grace.yaml` uses `extends: default` — it only needs to declare what differs:

```yaml
extends: default

modules:
  anaconda: Anaconda3/2025.12-2
  gcc: GCC/13.2.0
  cuda:
    - GCCcore/13.2.0
    - CUDA/13.1.0
    - NCCL/2.20.5-CUDA-12.4.1
  bio:
    - GCCcore/13.2.0
    - GCC/13.2.0
    - STAR/2.7.11b
    - BWA/0.7.18
    - SAMtools/1.21
    - BEDTools/2.31.1
    - ART/2.5.8        # verify: module spider ART

slurm:
  account: "your_tamu_account"
  mail_user: "you@tamu.edu"
  mail_type: END,FAIL
  partition_gpu: gpu
  gpu_gres: gpu:a100:2   # use gpu:a40:2 for A40 nodes
```

### Module safety checks

After loading each group, `_common.sh` calls `check_required_commands` to verify
that the required tools are in `PATH`. If a module version is wrong or missing you
get a targeted error instead of a silent failure deep in the stage:

```
[_common] ERROR: required commands not in PATH after loading 'bio' modules:
  missing: bwa
  Fix: update the module version in config/hpc/grace.yaml
  Hint: run 'module spider bwa' on Grace to find the available version.
```

Run `module spider <tool>` on Grace to find the correct version string, then
update `config/hpc/grace.yaml`.

### Runtime overrides

Every value is still overridable from the environment without editing files:

```bash
# Override a module version for a single submission
ANACONDA_MODULE=Anaconda3/2024.10 bash scripts/slurm/submit_pipeline.sh

# Override pipeline parameters
GPUS_PER_NODE=2 K=17 VOCAB=32000 bash scripts/slurm/submit_pipeline.sh
TOKENIZER_NAME=my.tok MODEL_CLS=my.model sbatch scripts/slurm/40_classification.slurm

# Override GPU type
bash scripts/slurm/submit_pipeline.sh --gres=gpu:a40:2

# Override account (takes precedence over grace.yaml)
bash scripts/slurm/submit_pipeline.sh --account 123456789

# Target a different HPC site entirely
HPC_CONFIG=/path/to/my_site.yaml bash scripts/slurm/submit_pipeline.sh
```

Key pipeline knobs in `_common.sh`: `K`, `VOCAB`, `MAX_POSITION`, `SEED`,
`CONDA_ENV`, `GENCODE_FASTA`, `TOKENIZER_NAME`, `MLM_PROCESSING_NAME`,
`PROCESSING_NAME`, `MODEL_MLM`, `MODEL_CLS`, `*_TRAINER_CONFIG`, `ALBERT_CONFIG`.

## Optional: hyperparameter tuning (Optuna)

An optional sweep that runs *before* the final train stages and writes the
winning `TrainingArguments` JSON they consume. It uses **Optuna +
`HyperbandPruner`** (not Ray Tune): pure-Python, offline, first-class via
`Trainer.hyperparameter_search(backend="optuna")`, seeded `TPESampler` for
reproducibility. Parallelism on Grace is a **SLURM job array sharing one study**
through a `JournalFileBackend` on `$SCRATCH` — Optuna coordinates trial
assignment; no Ray cluster.

```bash
bash scripts/slurm/submit_tuning.sh                  # MLM + classification sweeps
bash scripts/slurm/submit_tuning.sh --classification # one sweep only
bash scripts/slurm/submit_tuning.sh --dry-run        # print the array -> finalize chain
#   or: python -m trap.reproduce tune --dry-run
```

Each sweep is an array of workers (`35_tune_classification.slurm` = `--array=0-7%4`,
`25_tune_mlm.slurm` = `--array=0-5%3`); each has its own finalize job
(`25_tune_mlm_finalize.slurm`, `35_tune_classification_finalize.slurm`) submitted
`afterok` the array that writes its config. Feed those into the train stages:

```bash
CLS_TRAINER_CONFIG=config/training/classification_final.tuned.json \
    sbatch scripts/slurm/40_classification.slurm
MLM_TRAINER_CONFIG=config/training/mlm.tuned.json \
    sbatch scripts/slurm/30_mlm_pretrain.slurm
```

Search spaces live in `config/tuning/*.yaml` (knobs: `n_trials`, `sampler`,
`pruner`, `seed`, `max_trial_epochs`). Tuning knobs in `_common.sh`:
`TRIALS_PER_WORKER`, `STUDY_CLS`/`STUDY_MLM`, `TUNE_DIR`, `TUNE_JOURNAL_*`.

Notes:
- **MLM trials run a short proxy schedule** (`max_trial_epochs` in the YAML), not
  the full 40 epochs — Hyperband prunes the rest; the winning lr/wd/warmup then
  feed the full `mlm.json`.
- `TPESampler(seed=3469)` is fully reproducible for a single worker; across array
  workers TPE reads completed trials from the shared journal, so it is
  best-effort reproducible.
- A single worker run without `--storage` (in-memory study) writes the tuned
  config itself; the shared-study path relies on `tune_finalize`.

## Caveats (validate before the full re-do)

- **`scripts/divide_gff.py`** — **VALIDATED** (2026-05-29). Tested against a
  28,484-entry synthetic GFF3 derived from the real
  `GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript.gtf`. Subfamily counts match
  the ground-truth BED (L1HS=79, L1PA2=392, 132 total). All three
  `Target "Motif:..."` / `Target=...` / `Target "..."` attribute formats parse
  correctly. Note: the existing k18 pipeline used a GTF (not GFF3); the k17/v48
  re-do downloads the NCBI GFF3 which this script is designed for.

- **`20_dataset` ART/STAR/bedtools chain** needs one real-data validation run
  once the NCBI GFF3 is downloaded by `00_fetch_references`. Set `SKIP_BUILD=1`
  to skip build and reuse existing FASTQ files for the preprocessing stages only.

- **Transcript-level grouping**: validated. `_extract_transcript_id` uses
  `read_id.split("|")[0]`, which correctly extracts the repeat-element ID
  (e.g. `84120` from `84120|L1PA2|LINE-74`) for LINE1 reads and the transcript
  ID (e.g. `ENST...`) for GENCODE reads. ART's read-number suffix appended by
  ART is in a later `|`-field and does not interfere.

Retired `translast`-era scripts are preserved under `legacy/`.
