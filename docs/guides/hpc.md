# HPC setup (Grace / TAMU)

TrAP uses SLURM job arrays with `--dependency=afterok` chaining.
See `scripts/slurm/README.md` for the complete reference. This page covers
the one-time site configuration.

## First-time setup

```bash
git clone https://github.com/andchamorro/TrAP.git
cd TrAP

# 1. Create the conda environment (from repo root — see installation guide)
bash scripts/setup_conda_env.sh
conda activate trap

# 2. Create your site config from the template
cp config/hpc/grace.yaml.example config/hpc/grace.yaml
# Edit grace.yaml — fill in your account and email (file is gitignored)
```

`config/hpc/grace.yaml` is **gitignored** so personal settings are never
committed. `config/hpc/default.yaml` (committed) provides generic fallbacks.

## `grace.yaml` structure

```yaml
extends: default        # inherits config/hpc/default.yaml

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
  gpu_gres: gpu:a100:2
```

## Submit the full pipeline

```bash
bash scripts/slurm/submit_pipeline.sh           # 00 → 50
bash scripts/slurm/submit_pipeline.sh --dry-run # preview chain
bash scripts/slurm/submit_pipeline.sh --from 30_mlm_pretrain  # resume
```

## Module safety checks

After loading each module group, `_common.sh` calls `check_required_commands`
which exits with a clear message if any tool is missing:

```
[_common] ERROR: required commands not in PATH after loading 'bio' modules:
  missing: bwa
  Fix: update the module version in config/hpc/grace.yaml
  Hint: run 'module spider bwa' on Grace to find the available version.
```

## Runtime overrides

Any shell variable overrides a YAML value for that submission:

```bash
ANACONDA_MODULE=Anaconda3/2024.10 bash scripts/slurm/submit_pipeline.sh
bash scripts/slurm/submit_pipeline.sh --gres=gpu:a40:2
bash scripts/slurm/submit_pipeline.sh --account 123456789
HPC_CONFIG=/path/to/my_site.yaml bash scripts/slurm/submit_pipeline.sh
```
