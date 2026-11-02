# archive/scripts/

Retired top-level scripts.

| File | Origin | Successor | Why retired |
|---|---|---|---|
| `run_salmon.sh` | `notebooks/scripts/`, moved to `scripts/` in `2f02e59` | `scripts/sh/run_salmon.sh` | The `sh/` version is the refactor used by the benchmark (source-suffixed dataset dirs, resumable, no TPM rewrite). |
| `run_l1em.sh` | as above | `scripts/sh/run_l1em.sh` | Same refactor. |
| `extract_filtered.sh` | `8da3fbf` | none | Extracts the simulated reads that overlap insertion BEDs across deletion-probability sweeps; no stage or document called it. |
| `quantify_preprocessing.sh`, `quantify_repeatmasker.sh`, `train_repeatmasker.sh` | `907163a` (2025, transLAST) | the stage wrappers in `scripts/slurm/` | Accelerate wrappers for the k=18 `albert.l1hs_l1pa2.repeatmasker` model, which the k=17 canonical-hash pipeline replaced. |
