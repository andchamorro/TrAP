# TrAP training reproduction pipeline — Grace HPRC

Phase-3 of `.trap/plans/perf-and-reproducibility-plan.md` (§6): a config-driven,
manifest-stamped pipeline for the clean **k=17 / GENCODE v48** re-do, orchestrated
as an `sbatch --dependency=afterok` chain on the TAMU **Grace** cluster (Cascade
Lake; GPU stages default to **2× A100 40 GB**).

> **Track A (no MLM pre-training).** The MLM pre-training phase is dropped: the
> Salmon canonical k-mer vocabulary is feature-hashed, which makes the masked-LM
> objective unlearnable (loss freezes at `H(unigram) ≈ ln(vocab)`; full diagnosis
> in `.trap/plans/mlm-pretraining-freeze-action-plan.md`). The classifier is
> fine-tuned **from random init**. The shelved MLM stages (`24_mlm_smoke`,
> `25_tune_mlm`, `30_mlm_pretrain`) live in `scripts/slurm/legacy/mlm/`.

```
00_fetch_references       CPU   GENCODE v48 + GRCh38.p14 download, STAR/BWA indexes
        │
10_tokenizer              CPU   Salmon canonical k-mer tokenizer (k=17; index/hash build)
        │
20_dataset                CPU   ART(-f 5) → STAR(multimap 100) → bedtools → label;
        │                        classification dataset (transcript-level split)
        │
21_dataset_diagnosis      CPU   row-count / token-length / split-leakage diagnostics
        │
34_classification_smoke   GPU¹  pre-flight GATE: fine-tune on a 1k-row subset, FAIL
        │                        unless loss drops below ln(num_labels)
        │
   [optional]─┬─ 35_tune_classification (array)  GPU   Optuna+Hyperband cls sweep (8 workers)
              └─ 35_tune_classification_finalize  CPU   write classification_final.tuned.json
        │
40_classification         GPU   fine-tune L1HS / L1PA / NEGATIVE from random init
        │
50_benchmark              GPU   streaming `quantify` throughput benchmark + manifest
```

¹ The smoke gate runs on a **single GPU** (plain `python`, no DDP) so the reported
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
bash scripts/slurm/submit_pipeline.sh --from 34_classification_smoke
bash scripts/slurm/submit_pipeline.sh --from 20_dataset --to 40_classification

# list the stages
python -m trap.reproduce stages
```

Submit a single stage directly (from the repo root, so `$SLURM_SUBMIT_DIR` is correct):

```bash
# GPU stages require --gres and --partition since they are not in the .slurm file
sbatch --gres=gpu:a100:2 --partition=gpu scripts/slurm/40_classification.slurm
# CPU stages work as-is
sbatch scripts/slurm/10_tokenizer.slurm
```

## Pre-flight smoke gate (`34_classification_smoke`)

This stage is a **gate**, not a training stage: it exists to fail *fast and loud*
when the classification config cannot learn (or the dataset is malformed), so a
misconfiguration never reaches a multi-day GPU run.

### Why it exists

An MLM pretraining run once executed for **35 hours** and produced a checkpoint
that had learned *nothing* — the loss sat at the uniform-random baseline for all
40 epochs and masked-token accuracy was frozen. The root cause was **structural,
not a hyperparameter**: the Salmon tokenizer feature-hashes canonical k-mers
into 65 536 buckets with a non-invertible avalanche hash, so predicting a masked
bucket from its neighbours is not a representable function and the MLE-optimal
predictor is the marginal token distribution (loss = `H(unigram) ≈ ln(vocab)`).
Higher learning rates, fp32, and a fixed-mask memorisation run all stalled
identically. Full diagnosis:
`.trap/plans/mlm-pretraining-freeze-action-plan.md`.

That is why MLM pre-training is dropped (Track A) and the original MLM smoke gate
(`24_mlm_smoke`) is shelved in `legacy/mlm/` — note it has its own baseline bug
(it checks `ln(vocab)` rather than the empirical `H(marginal)`, so it would have
*passed* a unigram-collapsed run). The classification gate below remains: it
turns "35 h wasted, discovered days later" into "fails in ~10 min".

### What it checks

The gate trains a few hundred steps on a **1k-row `--debug` subset**, on a
**single GPU** (so the reported loss has no DDP `×num_processes` aggregation
artifact), then asserts the loss dropped a clear margin below the uniform-random
baseline. If it did not, the job exits non-zero and the `afterok` dependency
**stops the whole pipeline** before the expensive stage runs.

| Gate | Runs before | Pass condition | Default margin |
|---|---|---|---|
| `34_classification_smoke` | `35_tune_classification`, `40_classification` | `train_loss < ln(num_labels) − margin` | `0.2` nats |

`34_classification_smoke` needs the tokenizer (stage 10) and the tokenized
classification dataset (stage 20); under Track A the classifier trains from random
init, so the gate has no MLM-checkpoint dependency. It is wired into
`submit_pipeline.sh` automatically — no extra flags.

### Running the gate on its own

```bash
RUN_CONFIG=config/runs/salmon.yaml sbatch scripts/slurm/34_classification_smoke.slurm
```

A failure prints the measured loss, the threshold, and the first things to check
(learning rate, data/label alignment, dataset row counts).

### Configs and tunables

The gate runs a dedicated short config so it stays fast and deterministic:

| File | Used by | Notes |
|---|---|---|
| `config/training/classification.smoke.json` | `34_classification_smoke` | 3 epochs, fine-tuning LR |

Override per submission via the environment:

```bash
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
# (VOCAB is a legacy SPM/BPE training arg — the Salmon tokenizer ignores it; its
#  effective vocab is 5 + n_hash, overridden into the model from len(tokenizer).)
GPUS_PER_NODE=2 K=17 bash scripts/slurm/submit_pipeline.sh
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

Under Track A only the **classification** sweep runs (MLM tuning is shelved in
`legacy/mlm/`).

```bash
bash scripts/slurm/submit_tuning.sh                  # classification sweep (default)
bash scripts/slurm/submit_tuning.sh --classification # explicit; same as default
bash scripts/slurm/submit_tuning.sh --dry-run        # print the array -> finalize chain
#   or: python -m trap.reproduce tune --dry-run
```

The sweep is an array of workers (`35_tune_classification.slurm` = `--array=0-7%4`)
with its own finalize job (`35_tune_classification_finalize.slurm`) submitted
`afterok` the array that writes its config. Feed that into the train stage:

```bash
# finalize writes a RUN-NAMESPACED tuned config (CLS_TUNED_OUT), e.g.
#   config/training/classification_final.gencode.v48.k17.spm.tuned.json
# so an SPM sweep never overwrites the Salmon one. Feed that exact path back:
CLS_TRAINER_CONFIG=config/training/classification_final.gencode.v48.k17.spm.tuned.json \
    TRAP_SPM_EXPERIMENTAL=1 RUN_CONFIG=config/runs/spm.yaml \
    sbatch scripts/slurm/40_classification.slurm
```

Search spaces live in `config/tuning/*.yaml` (knobs: `n_trials`, `sampler`,
`pruner`, `seed`, `max_trial_epochs`). Tuning knobs in `_common.sh`:
`TRIALS_PER_WORKER`, `STUDY_CLS`, `TUNE_DIR`, `TUNE_JOURNAL_*`.

Notes:
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
