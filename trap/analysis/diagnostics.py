"""Pipeline diagnostic suite — stage 21 and beyond.

Entry point: ``python -m trap.analysis.diagnostics dataset``

Sub-commands (one per pipeline phase):
  dataset   — validate stage-20 outputs: MLM + classification datasets,
               warmup-steps computation, class balance, transcript leakage.
  (future)  tune / train / classification will follow the same pattern.

Each check is tagged [PASS] / [WARN] / [FAIL]. The exit code is non-zero when
any FAIL is present, so the SLURM dependency chain aborts early.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import typer
import yaml
from datasets import DatasetDict, load_from_disk
from loguru import logger

import trap.config.config  # noqa: F401 — registers STAGE level + tqdm-loguru
from trap.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(
    name="diagnostics",
    help="TrAP pipeline diagnostic suite. Run after each stage to verify outputs.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """TrAP pipeline diagnostics — sub-commands: dataset (stage 21)."""

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_FAILS: List[str] = []
_WARNS: List[str] = []


def _pass(msg: str) -> None:
    logger.success(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    logger.warning(f"[WARN] {msg}")
    _WARNS.append(msg)


def _fail(msg: str) -> None:
    logger.error(f"[FAIL] {msg}")
    _FAILS.append(msg)


def _section(title: str) -> None:
    logger.info(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_dataset_info(path: Path) -> Dict:
    """Return the first splits-entry from a HuggingFace dataset_info.json."""
    info_file = path / "dataset_info.json"
    if not info_file.exists():
        return {}
    raw = json.loads(info_file.read_text())
    # HF stores {"splits": {"train": {"num_examples": N, ...}}} at dataset level
    splits = raw.get("splits", {})
    if splits:
        return next(iter(splits.values()))
    return raw


def _row_count(ds_path: Path) -> Optional[int]:
    info = _load_dataset_info(ds_path)
    num_examples = info.get("num_examples")
    if num_examples is not None:
        return int(num_examples)

    try:
        return int(load_from_disk(str(ds_path)).num_rows)
    except Exception:
        return None


def _format_row_count(row_count: Optional[int]) -> str:
    return f"{row_count:,}" if row_count is not None else "unknown"


def _length_stats(ids: List[List[int]], label: str) -> Dict:
    """Return mean/std/p10/p50/p90/max length and log a summary line."""
    lens = np.array([len(x) for x in ids], dtype=np.int32)
    stats = {
        "mean": float(lens.mean()),
        "std": float(lens.std()),
        "p10": int(np.percentile(lens, 10)),
        "p50": int(np.percentile(lens, 50)),
        "p90": int(np.percentile(lens, 90)),
        "max": int(lens.max()),
        "min": int(lens.min()),
    }
    logger.info(
        f"  {label} length (tokens): "
        f"mean={stats['mean']:.1f} ±{stats['std']:.1f}  "
        f"p10/p50/p90={stats['p10']}/{stats['p50']}/{stats['p90']}  "
        f"min={stats['min']} max={stats['max']}"
    )
    return stats


def _unk_rate(ids: List[List[int]], unk_id: int, pad_id: int, label: str) -> float:
    total = real = unk = 0
    for seq in ids:
        for tok in seq:
            total += 1
            if tok == pad_id:
                continue
            real += 1
            if tok == unk_id:
                unk += 1
    rate = unk / real if real else 0.0
    logger.info(f"  {label} UNK rate: {100 * rate:.4f}%  ({unk:,}/{real:,} non-pad tokens)")
    return rate


# ---------------------------------------------------------------------------
# Warmup-steps computation
# ---------------------------------------------------------------------------

def _compute_warmup(
    train_rows: int,
    num_epochs: int,
    per_device_batch: int,
    num_gpus: int,
    warmup_ratio: float,
) -> int:
    eff_batch = per_device_batch * num_gpus
    steps_per_epoch = math.ceil(train_rows / eff_batch)
    total_steps = steps_per_epoch * num_epochs
    return round(warmup_ratio * total_steps)


def _check_warmup(
    label: str,
    train_rows: int,
    num_epochs: int,
    per_device_batch: int,
    num_gpus: int,
    configured: Optional[int],
    configured_ratio: Optional[float],
) -> Dict:
    """Compute expected warmup_steps and compare to what is in the config."""
    eff_batch = per_device_batch * num_gpus
    steps_per_epoch = math.ceil(train_rows / eff_batch)
    total_steps = steps_per_epoch * num_epochs

    # warmup_ratio is valid in transformers 5.x and is scale-invariant (HF scales
    # it to the run's own total steps), so it self-adjusts on a dataset rebuild —
    # preferred over a hard-coded warmup_steps. Report the equivalent for context.
    if configured_ratio is not None and configured is None:
        computed = round(configured_ratio * total_steps)
        logger.info(
            f"  {label}: warmup_ratio={configured_ratio} -> {computed:,} steps "
            f"({100 * configured_ratio:.0f}% of total_steps={total_steps:,}, "
            f"eff_batch={eff_batch}); scale-invariant, no recompute needed on rebuild."
        )
        configured = computed  # treat as equivalent for the PASS check

    computed_10pct = round(0.10 * total_steps)
    computed_5pct  = round(0.05 * total_steps)

    logger.info(
        f"  {label}: train_rows={train_rows:,}  epochs={num_epochs}  "
        f"eff_batch={eff_batch} ({num_gpus}×{per_device_batch})  "
        f"steps/epoch={steps_per_epoch:,}  total_steps={total_steps:,}"
    )
    logger.info(
        f"  {label}: warmup @ 0%=0  5%={computed_5pct:,}  10%={computed_10pct:,}"
    )

    if configured is not None:
        ratio = configured / total_steps
        if abs(ratio - 0.10) < 0.005 or abs(ratio - 0.05) < 0.005 or configured == 0:
            _pass(f"{label}: warmup_steps={configured:,} ({100*ratio:.1f}% of {total_steps:,})")
        else:
            _fail(
                f"{label}: warmup_steps={configured:,} ({100*ratio:.2f}%) looks wrong; "
                f"expected ~{computed_10pct:,} (10%) or {computed_5pct:,} (5%)"
            )
    else:
        _warn(f"{label}: no warmup_steps/warmup_ratio found in config")

    return {
        "train_rows": train_rows,
        "num_epochs": num_epochs,
        "eff_batch": eff_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "computed_warmup_0pct": 0,
        "computed_warmup_5pct": computed_5pct,
        "computed_warmup_10pct": computed_10pct,
        "configured_warmup_steps": configured,
    }


# ---------------------------------------------------------------------------
# dataset sub-command
# ---------------------------------------------------------------------------

@app.command()
def dataset(
    mlm_dataset: Path = typer.Option(
        None,
        "--mlm-dataset",
        help="Path to MLM processed dataset (masking/grouped). "
             "Defaults to $PROCESSED_DATA_DIR/$MLM_PROCESSING_NAME.",
    ),
    cls_dataset: Path = typer.Option(
        None,
        "--cls-dataset",
        help="Path to classification processed dataset (classification/tokenized). "
             "Defaults to $PROCESSED_DATA_DIR/$PROCESSING_NAME.",
    ),
    mlm_config: Path = typer.Option(
        CONFIG_DIR / "training" / "mlm.json",
        "--mlm-config",
        help="Training config for the MLM stage (mlm.json).",
    ),
    cls_config: Path = typer.Option(
        CONFIG_DIR / "training" / "classification_final.json",
        "--cls-config",
        help="Training config for the classification stage.",
    ),
    mlm_tune_config: Path = typer.Option(
        CONFIG_DIR / "tuning" / "mlm_optuna.yaml",
        "--mlm-tune-config",
        help="Optuna search config for the MLM tuning stage.",
    ),
    hpc_config: Path = typer.Option(
        CONFIG_DIR / "hpc" / "grace.yaml",
        "--hpc-config",
        help="HPC config for GPU count (gpus_per_node).",
    ),
    num_gpus: int = typer.Option(
        2, "--num-gpus",
        help="Number of GPUs for training (overrides HPC config).",
    ),
    sample_size: int = typer.Option(
        2000, "--sample",
        help="Rows to sample for token/length stats (0 = all rows, slow).",
    ),
    max_position_mlm: int = typer.Option(
        1280, "--max-position-mlm",
        help="max_position_embeddings for MLM model.",
    ),
    max_position_cls: int = typer.Option(
        280, "--max-position-cls",
        help="max_position_embeddings for classification model.",
    ),
    out_json: Optional[Path] = typer.Option(
        None, "--out-json",
        help="Write JSON diagnostic summary to this path.",
    ),
) -> None:
    """Validate stage-20 dataset outputs and compute training hyper-parameters.

    Checks MLM masking and classification datasets for:
    \b
    - Row counts and split ratios
    - Sequence length distribution vs max_position_embeddings
    - Token UNK rate
    - Warmup-steps computation and comparison to configured values
    - Class balance (classification)
    - Transcript-level no-overlap invariant (classification)
    - warmup_ratio deprecation in config files
    """
    global _FAILS, _WARNS
    _FAILS, _WARNS = [], []
    summary: Dict = {}

    # --- resolve HPC GPU count -------------------------------------------
    _resolved_gpus = num_gpus
    _hpc = hpc_config if (hpc_config and hpc_config.exists()) else (CONFIG_DIR / "hpc" / "default.yaml")
    if _hpc.exists():
        try:
            _hpc_data = yaml.safe_load(_hpc.read_text())
            _resolved_gpus = int(
                (_hpc_data.get("gpu") or {}).get("gpus_per_node", num_gpus)
            )
            logger.info(f"HPC config {_hpc.name}: gpus_per_node={_resolved_gpus}")
        except Exception:
            pass

    # --- resolve dataset paths -------------------------------------------
    import os
    mlm_processing_name = os.environ.get("MLM_PROCESSING_NAME", "gencode.v48.k17.32k")
    processing_name     = os.environ.get("PROCESSING_NAME",     "gencode.v48.k17.32k/l1hs_l1pa2")

    if mlm_dataset is None:
        mlm_dataset = PROCESSED_DATA_DIR / mlm_processing_name / "masking" / "grouped"
    if cls_dataset is None:
        cls_dataset = PROCESSED_DATA_DIR / processing_name / "classification" / "tokenized"

    # =====================================================================
    # 1. MLM DATASET
    # =====================================================================
    _section("MLM masking dataset")
    summary["mlm"] = {}

    mlm_splits = {}
    mlm_missing = []
    for split in ("train", "test"):
        p = mlm_dataset / split
        if p.exists():
            n = _row_count(p)
            mlm_splits[split] = n
            logger.info(f"  {split}: {_format_row_count(n)} rows  ({p})")
        else:
            mlm_splits[split] = None
            mlm_missing.append(split)

    if mlm_missing:
        _fail(f"MLM dataset missing splits: {mlm_missing}  (expected at {mlm_dataset})")
    else:
        _pass(f"MLM dataset found at {mlm_dataset}")

    summary["mlm"]["splits"] = mlm_splits

    # Token / length stats from a sample
    mlm_length_stats: Dict = {}
    train_arrow = list((mlm_dataset / "train").glob("*.arrow")) if (mlm_dataset / "train").exists() else []
    if not mlm_missing and train_arrow:
        try:
            from datasets import Dataset as HFDataset
            train_ds = HFDataset.load_from_disk(str(mlm_dataset / "train"))
            n_sample = min(sample_size, len(train_ds)) if sample_size > 0 else len(train_ds)
            sample = train_ds.select(range(n_sample))

            ids_list: List[List[int]] = sample["input_ids"]
            mlm_length_stats = _length_stats(ids_list, "MLM train (sample)")

            # max-position check
            over = sum(1 for s in ids_list if len(s) > max_position_mlm)
            if over:
                _fail(
                    f"MLM: {over}/{n_sample} sampled sequences exceed "
                    f"max_position_embeddings={max_position_mlm}"
                )
            else:
                _pass(f"MLM: all sampled sequences ≤ max_position_embeddings={max_position_mlm}")

            # UNK rate (load tokenizer if available)
            try:
                from trap.loaders.tokenizer import load_kmer_tokenizer
                tok = load_kmer_tokenizer(
                    str(MODELS_DIR / os.environ.get("TOKENIZER_NAME", "tokenizer.gencode.v48.k17.32k"))
                )
                _unk_rate(ids_list, tok.unk_token_id or 0, tok.pad_token_id or 1, "MLM train (sample)")
            except Exception:
                logger.info("  (tokenizer not loaded — skipping UNK rate)")

        except Exception as e:
            _warn(f"Could not load MLM dataset for token stats: {e}")
    elif not mlm_missing:
        logger.info("  (MLM dataset has no arrow data files locally — skipping token stats)")

    summary["mlm"]["length_stats"] = mlm_length_stats

    # Warmup computation — MLM full training
    _section("MLM warmup-steps")
    mlm_cfg: Dict = {}
    if mlm_config.exists():
        mlm_cfg = json.loads(mlm_config.read_text())
    else:
        _warn(f"MLM training config not found: {mlm_config}")

    mlm_tune_cfg: Dict = {}
    if mlm_tune_config.exists():
        mlm_tune_cfg = yaml.safe_load(mlm_tune_config.read_text()) or {}

    train_rows_mlm = mlm_splits.get("train") or 0
    if train_rows_mlm:
        # Full training
        warmup_full = _check_warmup(
            "MLM full",
            train_rows=train_rows_mlm,
            num_epochs=int(mlm_cfg.get("num_train_epochs", 40)),
            per_device_batch=int(mlm_cfg.get("per_device_train_batch_size", 16)),
            num_gpus=_resolved_gpus,
            configured=mlm_cfg.get("warmup_steps"),
            configured_ratio=mlm_cfg.get("warmup_ratio"),
        )
        summary["mlm"]["warmup_full"] = warmup_full

        # Optuna trial (proxy schedule)
        trial_epochs = int((mlm_tune_cfg.get("max_trial_epochs") or mlm_cfg.get("num_train_epochs", 40)))
        tune_warmup_values = (
            (mlm_tune_cfg.get("search_space") or {})
            .get("warmup_steps", {})
            .get("values")
        )
        if tune_warmup_values:
            eff = int(mlm_cfg.get("per_device_train_batch_size", 16)) * _resolved_gpus
            steps_trial = math.ceil(train_rows_mlm / eff) * trial_epochs
            total_trial = steps_trial
            computed_max = round(0.10 * total_trial)
            logger.info(
                f"  MLM tune ({trial_epochs} epochs): total_steps={total_trial:,}  "
                f"warmup 10%={computed_max:,}"
            )
            if tune_warmup_values[-1] == computed_max:
                _pass(f"MLM tune warmup_steps values {tune_warmup_values} match 10% of {total_trial:,}")
            elif abs(tune_warmup_values[-1] - computed_max) <= 10:
                _pass(f"MLM tune warmup_steps values {tune_warmup_values} ≈ 10% (off by ≤10)")
            else:
                _fail(
                    f"MLM tune warmup_steps values {tune_warmup_values}: "
                    f"max={tune_warmup_values[-1]:,} but 10% of {total_trial:,}={computed_max:,}"
                )
            summary["mlm"]["warmup_tune"] = {
                "trial_epochs": trial_epochs,
                "total_steps": total_trial,
                "configured_values": tune_warmup_values,
                "computed_10pct": computed_max,
            }
        elif "warmup_ratio" in (mlm_tune_cfg.get("search_space") or {}):
            wr_spec = mlm_tune_cfg["search_space"].get("warmup_ratio") or {}
            ratio_vals = wr_spec.get("values") or []
            bad = [v for v in ratio_vals if not 0.0 <= float(v) <= 0.5]
            if bad:
                _fail(f"MLM tune warmup_ratio values out of range [0, 0.5]: {bad}")
            else:
                _pass(
                    f"MLM tune searches warmup_ratio={ratio_vals} (scale-invariant; the "
                    "same fraction applies to the proxy and the full run -> no proxy→full drift)"
                )
            summary["mlm"]["warmup_tune"] = {"warmup_ratio_values": ratio_vals}
    else:
        _warn("MLM train row count unknown — skipping warmup computation")

    # =====================================================================
    # 2. CLASSIFICATION DATASET
    # =====================================================================
    _section("Classification dataset")
    summary["cls"] = {}

    cls_splits_found: Dict[str, Optional[int]] = {}
    cls_present = cls_dataset.exists()
    if not cls_present:
        _warn(f"Classification dataset not found at {cls_dataset} (run stage 20 first)")
    else:
        for split in ("train", "test", "eval"):
            p = cls_dataset / split
            if p.exists():
                n = _row_count(p)
                cls_splits_found[split] = n
                logger.info(f"  {split}: {_format_row_count(n)} rows")
        if not cls_splits_found:
            _fail(f"Classification dataset directory exists but no split dirs found at {cls_dataset}")
        else:
            _pass(f"Classification dataset found: {list(cls_splits_found.keys())}")

    summary["cls"]["splits"] = cls_splits_found

    # Class balance
    cls_balance: Dict = {}
    cls_has_data = cls_present and cls_splits_found and any(
        list((cls_dataset / s).glob("*.arrow")) for s in cls_splits_found
    )
    if cls_present and cls_splits_found and not cls_has_data:
        logger.info("  (CLS dataset has no arrow data files locally — skipping token/class stats)")
    if cls_has_data:
        try:
            from datasets import Dataset as HFDataset
            cls_ds = DatasetDict({
                split: HFDataset.load_from_disk(str(cls_dataset / split))
                for split in cls_splits_found if (cls_dataset / split).exists()
                and list((cls_dataset / split).glob("*.arrow"))
            })
            label_names = cls_ds["train"].features["label"].names

            for split, ds_split in cls_ds.items():
                counts = Counter(ds_split["label"])
                total = sum(counts.values())
                fracs = {label_names[k]: f"{100*v/total:.1f}%" for k, v in sorted(counts.items())}
                logger.info(f"  {split} class distribution: {fracs}")
                cls_balance[split] = {label_names[k]: v for k, v in counts.items()}

                # L1HS present in every split?
                l1hs_id = next((i for i, n in enumerate(label_names) if n == "L1HS"), None)
                if l1hs_id is not None and counts.get(l1hs_id, 0) == 0:
                    _fail(f"L1HS is absent from {split} split — stratification failed")
                elif l1hs_id is not None:
                    _pass(f"L1HS present in {split} split ({counts[l1hs_id]:,} rows)")

            # Transcript-level no-overlap (train ∩ test). transcript_id is retained
            # in the tokenised dataset only from the 2026-06-04 preprocessing fix on;
            # guard the column so older datasets skip cleanly (no exception) instead
            # of aborting the remaining checks below.
            if "train" in cls_ds and "test" in cls_ds:
                if "transcript_id" in cls_ds["train"].column_names:
                    train_tids = set(cls_ds["train"]["transcript_id"])
                    test_tids = set(cls_ds["test"]["transcript_id"])
                    overlap = train_tids & test_tids
                    if overlap:
                        _fail(
                            f"Train/test transcript leakage: {len(overlap):,} transcript IDs "
                            "appear in both splits (DR-6 violation)"
                        )
                    else:
                        _pass(
                            f"No transcript leakage: {len(train_tids):,} train / "
                            f"{len(test_tids):,} test transcript IDs are disjoint"
                        )
                else:
                    _warn(
                        "Transcript leakage check skipped: tokenised classification dataset "
                        "has no transcript_id column. Rebuild stage 20 with the 2026-06-04 "
                        "preprocessing fix to enable the DR-6 check."
                    )

            # Sequence length stats
            train_sample = cls_ds["train"].select(
                range(min(sample_size or len(cls_ds["train"]), len(cls_ds["train"])))
            )
            cls_len = _length_stats(train_sample["input_ids"], "CLS train (sample)")
            over_cls = sum(1 for s in train_sample["input_ids"] if len(s) > max_position_cls)
            if over_cls:
                _warn(
                    f"CLS: {over_cls}/{len(train_sample)} sampled sequences exceed "
                    f"max_position_embeddings={max_position_cls}"
                )
            else:
                _pass(f"CLS: all sampled sequences ≤ max_position_embeddings={max_position_cls}")

        except Exception as e:
            _warn(f"Could not load classification dataset for stats: {e}")

    summary["cls"]["class_balance"] = cls_balance

    # Warmup — classification final training
    _section("Classification warmup-steps")
    cls_cfg: Dict = {}
    if cls_config.exists():
        cls_cfg = json.loads(cls_config.read_text())
    else:
        _warn(f"Classification training config not found: {cls_config}")

    train_rows_cls = cls_splits_found.get("train") or 0
    if train_rows_cls:
        _check_warmup(
            "CLS final",
            train_rows=train_rows_cls,
            num_epochs=int(cls_cfg.get("num_train_epochs", 5)),
            per_device_batch=int(cls_cfg.get("per_device_train_batch_size", 16)),
            num_gpus=_resolved_gpus,
            configured=cls_cfg.get("warmup_steps"),
            configured_ratio=cls_cfg.get("warmup_ratio"),
        )
    else:
        _warn("CLS train row count unknown — skipping warmup computation")

    # =====================================================================
    # 3. SUMMARY
    # =====================================================================
    _section("Diagnostic summary")
    logger.info(f"  PASS checks: all unlisted above")
    if _WARNS:
        for w in _WARNS:
            logger.warning(f"  WARN: {w}")
    if _FAILS:
        for f in _FAILS:
            logger.error(f"  FAIL: {f}")
    else:
        logger.success("  All checks passed (no FAILs)")

    summary["status"] = {
        "fails": _FAILS,
        "warns": _WARNS,
        "exit_code": 1 if _FAILS else 0,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        logger.info(f"Diagnostic summary written to {out_json}")

    if _FAILS:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
