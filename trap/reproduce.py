"""Thin orchestration wrapper for the Phase-3 training reproduction pipeline.

The real work lives in per-stage SLURM scripts under ``scripts/slurm/``; this
module (plan §6) only lists the stages and shells out to
``scripts/slurm/submit_pipeline.sh`` to submit them as an
``sbatch --dependency=afterok`` chain on Grace.  No compute happens here.

Examples::

    python -m trap.reproduce stages
    python -m trap.reproduce submit --dry-run
    python -m trap.reproduce submit --from 30_mlm_pretrain
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Tuple

from loguru import logger
import typer

from trap.config.config import PROJ_ROOT
from trap.config.verbosity import set_verbosity

app = typer.Typer(help="Submit/inspect the Phase-3 training reproduction pipeline.")

_SUBMIT = PROJ_ROOT / "scripts" / "slurm" / "submit_pipeline.sh"
_SUBMIT_TUNING = PROJ_ROOT / "scripts" / "slurm" / "submit_tuning.sh"

# Track A: MLM pre-training is dropped (the hashed k-mer vocab makes the MLM
# objective unlearnable; see .trap/plans/mlm-pretraining-freeze-action-plan.md).
# The shelved MLM stages live in scripts/slurm/legacy/mlm/.
STAGES: List[Tuple[str, str]] = [
    (
        "00_fetch_references",
        "Download GENCODE v48 + GRCh38.p14; build STAR/BWA indexes + L1 corpus FASTA",
    ),
    ("10_tokenizer", "Build Salmon canonical k-mer tokenizer (k=17; index/hash build)"),
    ("20_dataset", "ART -> STAR -> bedtools -> transcript-level tokenized dataset"),
    ("21_dataset_diagnosis", "Row-count / token-length / split-leakage diagnostics"),
    (
        "34_classification_smoke",
        "Pre-flight GATE: fine-tune a 1k-row subset, fail fast if it cannot learn (GPU)",
    ),
    ("40_classification", "Classification fine-tuning L1HS/L1PA/NEGATIVE from random init (GPU)"),
    ("50_benchmark", "Streaming quantify benchmark on the fixture FASTQ (GPU)"),
]


@app.command()
def stages() -> None:
    """Print the ordered pipeline stages."""
    for i, (name, desc) in enumerate(STAGES):
        typer.echo(f"{i}. {name:<20} {desc}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity for child pipeline stages: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
) -> None:
    """Submit the full pipeline via ``scripts/slurm/submit_pipeline.sh``.

    Extra arguments pass straight through to the driver (e.g. ``--dry-run``,
    ``--from <stage>``, ``--to <stage>``).
    """
    set_verbosity(verbosity)
    os.environ["TRAP_VERBOSITY"] = verbosity
    if not _SUBMIT.exists():
        logger.error(f"Driver not found: {_SUBMIT}")
        raise typer.Exit(code=1)
    cmd = ["bash", str(_SUBMIT), *ctx.args]
    logger.log("STAGE", f"[reproduce:submit] Running: {' '.join(cmd)}")
    logger.info("Running: " + " ".join(cmd))
    raise typer.Exit(code=subprocess.call(cmd))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def tune(
    ctx: typer.Context,
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity for child pipeline stages: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
) -> None:
    """Submit the optional Optuna tuning sweep via ``scripts/slurm/submit_tuning.sh``.

    Extra arguments pass straight through (e.g. ``--classification``, ``--mlm``,
    ``--dry-run``, ``--from <stage>``).
    """
    set_verbosity(verbosity)
    os.environ["TRAP_VERBOSITY"] = verbosity
    if not _SUBMIT_TUNING.exists():
        logger.error(f"Driver not found: {_SUBMIT_TUNING}")
        raise typer.Exit(code=1)
    cmd = ["bash", str(_SUBMIT_TUNING), *ctx.args]
    logger.log("STAGE", f"[reproduce:tune] Running: {' '.join(cmd)}")
    logger.info("Running: " + " ".join(cmd))
    raise typer.Exit(code=subprocess.call(cmd))


if __name__ == "__main__":
    app()
