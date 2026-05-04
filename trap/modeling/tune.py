"""Optuna hyperparameter search for the MLM and classification stages.

Replaces the retired Ray-Tune / PopulationBasedTraining recipe (only the
reconstructed ``config/training/pbt_search_space.yaml`` survived). Optuna is
pure-Python, fully offline, and first-class in ``transformers`` via
``Trainer.hyperparameter_search(backend="optuna")``; ``HyperbandPruner`` kills
unpromising trials early (PBT's main practical benefit) and ``TPESampler(seed=...)``
keeps a single worker reproducible.

Parallelism on Grace: a SLURM job array where every task runs the same
sub-command against one **shared study** (``--storage <journal>`` →
``JournalFileBackend``); Optuna coordinates trial assignment. Run ``finalize``
once afterwards (``afterok``) to merge the winning params into the base
``TrainingArguments`` JSON that stages 30/40 consume.

Examples::

    # one worker of a shared-study sweep
    python -m trap.modeling.tune classification albert.l1hs_l1pa2 \\
        --search-config config/tuning/classification_optuna.yaml \\
        --study-name cls_v48 --storage $SCRATCH/trap_hpo/cls_v48.journal \\
        --n-trials 2 --pretrained-model-path albert.gencode.v48 \\
        --preprocessing-name gencode.v48.k17.32k/l1hs_l1pa2

    # write config/training/classification_final.tuned.json from the study
    python -m trap.modeling.tune finalize \\
        --search-config config/tuning/classification_optuna.yaml \\
        --study-name cls_v48 --storage $SCRATCH/trap_hpo/cls_v48.journal
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from transformers import (
    AlbertConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import typer

from trap.config import manifest as manifest_mod
from trap.config.config import CONFIG_DIR, MODELS_DIR, PROJ_ROOT
from trap.config.schemas import HPSpec, TuneSearchSchema, load_config
from trap.modeling import train as train_mod
from trap.utils.seeding import set_global_seed

app = typer.Typer(help="Optuna hyperparameter search for the MLM / classification stages.")


# ---------------------------------------------------------------------------
# Optuna plumbing
# ---------------------------------------------------------------------------


def _build_hp_space(space: dict[str, HPSpec]) -> Callable:
    """Compile a search-space dict into an Optuna ``hp_space(trial)`` callable.

    The returned callable maps each :class:`HPSpec` to the matching
    ``trial.suggest_*`` call and returns a dict of ``TrainingArguments``
    overrides (e.g. ``learning_rate``, ``weight_decay``).
    """

    def hp_space(trial):
        params: dict = {}
        for name, spec in space.items():
            if spec.sampler == "loguniform":
                params[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
            elif spec.sampler == "uniform":
                params[name] = trial.suggest_float(name, spec.low, spec.high)
            elif spec.sampler == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(spec.low),
                    int(spec.high),
                    step=int(spec.step) if spec.step else 1,
                    log=spec.log,
                )
            elif spec.sampler in ("categorical", "choice"):
                params[name] = trial.suggest_categorical(name, spec.values)
            else:  # pragma: no cover - guarded by the schema Literal
                raise ValueError(f"unknown sampler: {spec.sampler}")
        return params

    return hp_space


def _make_sampler(name: str, seed: int):
    if name == "tpe":
        return TPESampler(seed=seed)
    if name == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler: {name}")


def _make_pruner(name: str):
    if name == "hyperband":
        return HyperbandPruner()
    if name == "median":
        return MedianPruner()
    if name == "none":
        return NopPruner()
    raise ValueError(f"unknown pruner: {name}")


def _make_storage(storage: Optional[Path]):
    """Return a shared ``JournalStorage`` for *storage*, or ``None`` for in-memory."""
    if storage is None:
        return None
    Path(storage).parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(storage)))


def _study_kwargs(study_name, storage, sampler, pruner, seed) -> dict:
    """Build the kwargs that flow through ``hyperparameter_search`` to
    ``optuna.create_study`` (study_name/storage/sampler/pruner/load_if_exists)."""
    kw: dict = {
        "sampler": _make_sampler(sampler, seed),
        "pruner": _make_pruner(pruner),
        "load_if_exists": True,
    }
    if study_name:
        kw["study_name"] = study_name
    st = _make_storage(storage)
    if st is not None:
        kw["storage"] = st
    return kw


def _make_study(study_name, storage, sampler, pruner, seed, direction):
    """Create (or load) the study directly — used by ``finalize`` and tests."""
    return optuna.create_study(
        direction=direction, **_study_kwargs(study_name, storage, sampler, pruner, seed)
    )


def _resolve_repo(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJ_ROOT / p


def _classification_model_vocab(pretrained_model_path, tokenizer) -> int:
    """Effective vocab of the classification model under search.

    Mirrors ``make_classification_model_init``: the pretrained MLM checkpoint's
    embedding size when that ``final/`` dir exists, else the tokenizer vocab
    (the from-config fallback). Used to bound-check the dataset's token ids.
    """
    if pretrained_model_path:
        cfg = MODELS_DIR / pretrained_model_path / "final" / "config.json"
        if cfg.exists():
            return json.loads(cfg.read_text()).get("vocab_size", len(tokenizer))
    return len(tokenizer)


def _trial_training_args(search: TuneSearchSchema, model_name: str) -> TrainingArguments:
    """Base ``TrainingArguments`` for trials: no checkpoints, no best-model reload
    (only the metric matters during search), optional proxy epoch cap."""
    cfg = train_mod._read_config_dict(_resolve_repo(search.base_trainer_config))
    cfg.update(
        {
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "report_to": "none",
            "output_dir": str(MODELS_DIR / model_name / "hpo"),
        }
    )
    if search.max_trial_epochs is not None:
        cfg["num_train_epochs"] = search.max_trial_epochs
    set_global_seed(cfg.get("seed", search.seed))
    return TrainingArguments(**train_mod._resolve_precision(cfg))


def _write_tuned_config(search, best_params: dict, out_path: Path, meta: dict) -> Path:
    """Merge the winning params over the base config and write ``<name>.tuned.json``."""
    base_cfg = train_mod._read_config_dict(_resolve_repo(search.base_trainer_config))
    merged = {**base_cfg, **best_params}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    manifest_mod.write(out_path.parent, **meta)
    logger.success(f"Wrote tuned config -> {out_path}")
    return out_path


def _default_tuned_path(search: TuneSearchSchema) -> Path:
    base = _resolve_repo(search.base_trainer_config)
    return base.with_name(f"{base.stem}.tuned.json")


# ---------------------------------------------------------------------------
# Fast-HPO subsampling
# ---------------------------------------------------------------------------


def _stratified_subsample(ds, target_rows: int, seed: int, label_col: str):
    """Return a fixed-seed subset of ``ds`` with ``target_rows`` examples.

    Stratified by ``label_col`` when that column exists (preserves class
    proportions — critical for the rare L1HS target), else a shuffled select.
    """
    if target_rows >= ds.num_rows:
        return ds
    if label_col in ds.column_names:
        try:
            return ds.train_test_split(
                train_size=target_rows, seed=seed, stratify_by_column=label_col
            )["train"]
        except Exception as exc:  # e.g. column is not a ClassLabel
            logger.warning(f"[subsample] stratified split failed ({exc}); using shuffled select")
    return ds.shuffle(seed=seed).select(range(target_rows))


def _subsample_for_hpo(lm_datasets, keys, *, frac, n, seed, label_col="label"):
    """Shrink the named splits to a fast-HPO size, in place, reproducibly.

    A single fraction (``frac`` directly, or ``n / len(train)``) is applied to
    every split so train and eval stay proportional and class-balanced. The
    seed is fixed and identical across array workers, so every trial sees the
    SAME subset — subset noise cancels out of the trial-to-trial comparison.
    """
    if frac is None and n is None:
        return lm_datasets
    if frac is not None and n is not None:
        raise typer.BadParameter("Use only one of --subsample-frac / --subsample-n.")

    train_key = keys[0]
    train_rows = lm_datasets[train_key].num_rows
    if frac is not None:
        if not 0.0 < frac < 1.0:
            raise typer.BadParameter("--subsample-frac must be in (0, 1).")
        fraction = frac
    else:
        if n <= 0:
            raise typer.BadParameter("--subsample-n must be > 0.")
        fraction = min(1.0, n / train_rows)

    for key in keys:
        ds = lm_datasets[key]
        if n is not None and key == train_key:
            target = n
        else:
            target = max(1, round(fraction * ds.num_rows))
        before = ds.num_rows
        lm_datasets[key] = _stratified_subsample(ds, target, seed, label_col)
        after = lm_datasets[key].num_rows
        if label_col in lm_datasets[key].column_names:
            dist = dict(sorted(Counter(lm_datasets[key][label_col]).items()))
            logger.info(
                f"[subsample] {key}: {before:,} -> {after:,} "
                f"(stratified by {label_col}, seed={seed}; class counts={dist})"
            )
        else:
            logger.info(f"[subsample] {key}: {before:,} -> {after:,} (random, seed={seed})")
    return lm_datasets


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def classification(
    model_name: str = typer.Argument(help="Name for the HPO run / output dir"),
    search_config: Path = typer.Option(
        CONFIG_DIR / "tuning" / "classification_optuna.yaml", help="Tuning search YAML"
    ),
    pretrained_tokenizer_path: Path = typer.Option(None, help="Fallback tokenizer dir"),
    pretrained_model_path: Path = typer.Option(
        None, help="Pretrained MLM model name under models/"
    ),
    albert_config_path: Path = typer.Option(
        CONFIG_DIR / "albert_config_k17_v48.json", help="ALBERT config (fallback)"
    ),
    preprocessing_name: str = typer.Option(..., help="Tokenized classification dataset name"),
    study_name: str = typer.Option(None, help="Shared study name (for the journal storage)"),
    storage: Optional[Path] = typer.Option(
        None, help="JournalFileBackend path; in-memory + writes tuned config if unset"
    ),
    n_trials: int = typer.Option(
        None, help="Trials this worker runs (default: search-config n_trials)"
    ),
    num_workers: int = typer.Option(16, help="Dataloader workers"),
    subsample_frac: Optional[float] = typer.Option(
        None,
        help="Fast HPO: shrink train+eval to this fraction (0<f<1), stratified by "
        "label, fixed seed. Mutually exclusive with --subsample-n.",
    ),
    subsample_n: Optional[int] = typer.Option(
        None,
        help="Fast HPO: shrink the train split to this many rows (eval scaled to "
        "match). Mutually exclusive with --subsample-frac.",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Downsample the dataset for a smoke run"
    ),
):
    """Search classification hyperparameters (maximize ``eval_f1`` by default)."""
    search = load_config(search_config, TuneSearchSchema)
    trainer_args = _trial_training_args(search, model_name)

    lm_datasets, label2id, id2label = train_mod.load_classification_data(preprocessing_name)
    num_labels = len(id2label)
    if debug:
        small = lm_datasets["train"].train_test_split(train_size=200, test_size=40, seed=42)
        lm_datasets = small
    lm_datasets = _subsample_for_hpo(
        lm_datasets, ("train", "test"), frac=subsample_frac, n=subsample_n, seed=search.seed
    )

    final_dir = MODELS_DIR / pretrained_model_path / "final" if pretrained_model_path else None
    tokenizer_src = (
        str(final_dir)
        if final_dir is not None and Path(final_dir).exists()
        else pretrained_tokenizer_path
    )
    # model_max_length needs the model's max_position; read it from the ALBERT config.
    max_position = AlbertConfig.from_json_file(albert_config_path).max_position_embeddings
    tokenizer = train_mod.load_tuned_tokenizer(tokenizer_src, max_position)

    # vocab_size matters only for the from-config fallback (no pretrained MLM
    # checkpoint); without it the model would size to the JSON default (32000)
    # while the salmon tokenizer/dataset uses up to ~65k ids -> CUDA gather OOB.
    model_vocab = _classification_model_vocab(pretrained_model_path, tokenizer)
    train_mod.assert_dataset_ids_in_range(
        lm_datasets, vocab_size=model_vocab, num_labels=num_labels, dataset_name=preprocessing_name
    )

    trainer = Trainer(
        model_init=train_mod.make_classification_model_init(
            pretrained_model_path,
            albert_config_path,
            num_labels,
            id2label,
            label2id,
            vocab_size=len(tokenizer),
        ),
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=train_mod.make_compute_classification_metrics(),
    )
    _run_search(trainer, search, study_name, storage, n_trials, model_name)


@app.command()
def masking(
    model_name: str = typer.Argument(help="Name for the HPO run / output dir"),
    search_config: Path = typer.Option(
        CONFIG_DIR / "tuning" / "mlm_optuna.yaml", help="Tuning search YAML"
    ),
    pretrained_tokenizer_path: Path = typer.Option(..., help="Tokenizer dir"),
    albert_config_path: Path = typer.Option(
        CONFIG_DIR / "albert_config_k17_v48.json", help="ALBERT config"
    ),
    preprocessing_name: str = typer.Option(..., help="Chunked MLM dataset name"),
    study_name: str = typer.Option(None, help="Shared study name (for the journal storage)"),
    storage: Optional[Path] = typer.Option(
        None, help="JournalFileBackend path; in-memory + writes tuned config if unset"
    ),
    n_trials: int = typer.Option(
        None, help="Trials this worker runs (default: search-config n_trials)"
    ),
    num_workers: int = typer.Option(16, help="Dataloader workers"),
    subsample_frac: Optional[float] = typer.Option(
        None,
        help="Fast HPO: shrink train+eval to this fraction (0<f<1), fixed seed. "
        "Mutually exclusive with --subsample-n.",
    ),
    subsample_n: Optional[int] = typer.Option(
        None,
        help="Fast HPO: shrink the train split to this many rows (eval scaled to "
        "match). Mutually exclusive with --subsample-frac.",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Downsample the dataset for a smoke run"
    ),
):
    """Search MLM hyperparameters on a short proxy schedule (minimize ``eval_loss``).

    Full 40-epoch trials are prohibitive; ``max_trial_epochs`` in the search YAML
    caps each trial and Hyperband prunes the rest. The winning params then feed
    the full ``config/training/mlm.json``.
    """
    search = load_config(search_config, TuneSearchSchema)
    trainer_args = _trial_training_args(search, model_name)

    max_position = AlbertConfig.from_json_file(albert_config_path).max_position_embeddings
    tokenizer = train_mod.load_tuned_tokenizer(pretrained_tokenizer_path, max_position)
    lm_datasets, data_collator = train_mod.load_masking_data(
        preprocessing_name, tokenizer, num_workers, debug=debug
    )
    # MLM splits carry no label column -> random fixed-seed subsample.
    lm_datasets = _subsample_for_hpo(
        lm_datasets, ("train", "eval"), frac=subsample_frac, n=subsample_n, seed=search.seed
    )
    train_mod.assert_dataset_ids_in_range(
        lm_datasets, vocab_size=len(tokenizer), dataset_name=preprocessing_name
    )

    # Tuning metric is eval_loss; skip logit accumulation ([n_eval, 1280, vocab_size] OOMs on 40 GB GPU)
    trainer_args.prediction_loss_only = True
    trainer = train_mod.MaskingTrainer(
        model_init=train_mod.make_masking_model_init(
            albert_config_path, vocab_size=len(tokenizer)
        ),
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=train_mod.compute_masking_metrics,
    )
    _run_search(trainer, search, study_name, storage, n_trials, model_name)


def _run_search(trainer, search, study_name, storage, n_trials, model_name) -> None:
    """Drive ``trainer.hyperparameter_search`` and (in-memory only) write the config."""
    hp_space = _build_hp_space(search.search_space)
    metric = search.metric
    n = n_trials or search.n_trials
    logger.info(
        f"Optuna search: metric={metric} direction={search.direction} "
        f"sampler={search.sampler} pruner={search.pruner} n_trials={n} "
        f"storage={'in-memory' if storage is None else storage}"
    )
    best = trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=lambda m: m[metric],
        n_trials=n,
        direction=search.direction,
        backend="optuna",
        **_study_kwargs(study_name, storage, search.sampler, search.pruner, search.seed),
    )
    # Non-main ranks return None under DDP; only rank 0 reports/writes.
    if best is None or not trainer.is_world_process_zero():
        return
    logger.success(
        f"Best trial #{best.run_id}: {metric}={best.objective} params={best.hyperparameters}"
    )
    if storage is None:
        meta = {
            "seed": search.seed,
            "tuning": {
                "study_name": study_name,
                "metric": metric,
                "direction": search.direction,
                "sampler": search.sampler,
                "pruner": search.pruner,
                "n_trials": n,
                "best_value": best.objective,
                "best_params": best.hyperparameters,
            },
        }
        _write_tuned_config(search, best.hyperparameters, _default_tuned_path(search), meta)
    else:
        logger.info("Shared study: run `tune finalize` to write the tuned config.")


@app.command()
def finalize(
    search_config: Path = typer.Option(..., help="The tuning search YAML used for the sweep"),
    study_name: str = typer.Option(..., help="Shared study name"),
    storage: Path = typer.Option(..., help="JournalFileBackend path of the shared study"),
    out: Path = typer.Option(None, help="Output .tuned.json (default: <base>.tuned.json)"),
):
    """Read the shared study and write ``<base>.tuned.json`` from its best trial."""
    search = load_config(search_config, TuneSearchSchema)
    storage_obj = _make_storage(storage)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_obj)
    except KeyError as exc:
        # study_name / storage are derived from the run config (mlm_processing_name).
        # The usual cause is finalizing without the SAME run config the sweep used,
        # so the derived study name points at a journal that has a different study.
        try:
            available = optuna.get_all_study_names(storage=storage_obj)
        except Exception:
            available = []
        raise typer.BadParameter(
            f"Study {study_name!r} not found in storage {storage}. "
            f"Available studies: {available or '(none — wrong journal path?)'}. "
            "The study name is derived from the run config; pass the SAME --run-config "
            "the sweep used, e.g.  RUN_CONFIG=config/runs/salmon.yaml sbatch "
            "scripts/slurm/25_tune_mlm_finalize.slurm"
        ) from exc
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        states = Counter(t.state.name for t in study.trials)
        raise typer.BadParameter(
            f"Study {study_name!r} has no COMPLETE trials to finalize "
            f"({len(study.trials)} total: {dict(states)}). Re-run the sweep; nothing was written."
        )
    best = study.best_trial
    logger.success(
        f"Best trial #{best.number}: value={best.value} params={best.params} "
        f"({len(completed)}/{len(study.trials)} trials complete)"
    )
    out_path = out or _default_tuned_path(search)
    meta = {
        "seed": search.seed,
        "tuning": {
            "study_name": study_name,
            "metric": search.metric,
            "direction": search.direction,
            "sampler": search.sampler,
            "pruner": search.pruner,
            "n_trials": len(study.trials),
            "best_value": best.value,
            "best_params": best.params,
        },
    }
    _write_tuned_config(search, best.params, out_path, meta)


if __name__ == "__main__":
    app()
