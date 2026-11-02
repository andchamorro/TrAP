# Archive index

Code retired from `main`. Everything listed here is reachable at tag `v1.0.0`
(`0adb9c6`). Files moved under `archive/` keep their original relative path and
carry a `README.md` there; files deleted outright are listed with the last commit
that changed them.

## Deleted

| Path | Last commit | Why |
|---|---|---|
| `trap/features.py` | `632439d` | Cookiecutter template (`# ---- REPLACE THIS WITH YOUR OWN CODE ----`, a dummy loop); nothing imported it. |
| `trap/plots.py` | `632439d` | Cookiecutter template; nothing imported it, and it no longer imported (`FIGURES_DIR` is not defined in `trap.config`). |

## Moved to `archive/`

The reason for each move is in the `README.md` of its `archive/` directory.

| Original path | Archive path | Last commit before the move |
|---|---|---|
| `scripts/draft/run_l1em_mut.sh` | `archive/scripts/draft/run_l1em_mut.sh` | `2f02e59` |
| `scripts/draft/run_mlem.sh` | `archive/scripts/draft/run_mlem.sh` | `2f02e59` |
| `scripts/draft/run_mlem_mut.sh` | `archive/scripts/draft/run_mlem_mut.sh` | `2f02e59` |
| `scripts/draft/run_mlsalmon_mut.sh` | `archive/scripts/draft/run_mlsalmon_mut.sh` | `2f02e59` |
| `scripts/draft/run_salmon_mut.sh` | `archive/scripts/draft/run_salmon_mut.sh` | `2f02e59` |
| `scripts/run_l1em.sh` | `archive/scripts/run_l1em.sh` | `2f02e59` |
| `scripts/run_salmon.sh` | `archive/scripts/run_salmon.sh` | `2f02e59` |
| `scripts/extract_filtered.sh` | `archive/scripts/extract_filtered.sh` | `8da3fbf` |
| `scripts/quantify_preprocessing.sh` | `archive/scripts/quantify_preprocessing.sh` | `632439d` |
| `scripts/quantify_repeatmasker.sh` | `archive/scripts/quantify_repeatmasker.sh` | `632439d` |
| `scripts/train_repeatmasker.sh` | `archive/scripts/train_repeatmasker.sh` | `632439d` |
| `scripts/sh/bowtie1AndCount.sh` | `archive/scripts/sh/bowtie1AndCount.sh` | `2f02e59` |
| `scripts/sh/bowtie2AndCount.sh` | `archive/scripts/sh/bowtie2AndCount.sh` | `2f02e59` |
| `scripts/sh/hisat2AndCount.sh` | `archive/scripts/sh/hisat2AndCount.sh` | `2f02e59` |
| `scripts/sh/starAndCount.sh` | `archive/scripts/sh/starAndCount.sh` | `2f02e59` |
| `config/trainer_config_base_uncased_fsdp.json` | `archive/config/trainer_config_base_uncased_fsdp.json` | `7860f51` |
| `config/runs/spm_k8.yaml` | `archive/config/runs/spm_k8.yaml` | `07c0b8a` |

## Frozen in place

`scripts/slurm/legacy/` is an archive kept at its original path, because some of
its files are cited for reported results (the MLM negative result and the
SentencePiece comparison; see `docs/guides/manuscript_provenance.md`). The
unreported ones are `google_sentencepiece.slurm`, `preprocessing_sequences.slurm`,
`train_repeatmasker.slurm`, `train_transcripts.slurm` and
`write_trainscrips_kmers.py`.
