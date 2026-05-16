"""Typer CLI for the entropy/redundancy k-mer analysis.

Run as ``python -m trap.analysis``.  Commands:

* ``run``           — full config-driven pipeline (all corpora + ablation + select-k)
* ``kmer-spectrum`` — entropy/redundancy spectrum for one corpus
* ``ablation``      — k -> classification macro-F1 sweep
* ``select-k``      — apply the quantitative selection criterion to existing tables

Every command seeds all RNGs (:func:`trap.utils.seeding.set_global_seed`) and
writes a ``manifest.json`` (git commit, seed, input SHA-256, parameters) next to
its outputs.  CSV outputs are row-sorted so re-running with the same seed yields
byte-identical files; Parquet is written alongside for efficient loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from loguru import logger
import pandas as pd
import typer

import trap.config.config  # noqa: F401  (registers STAGE level + tqdm-loguru)
from trap.analysis.ablation import ablation as run_ablation
from trap.analysis.kmer_counting import (
    _jellyfish_available,
    _seqkit_available,
    kmer_spectrum as run_kmer_spectrum,
    load_corpus,
)
from trap.analysis.selection import compute_selection
from trap.config import manifest
from trap.config.schemas import KmerSpectrumConfigSchema, load_config
from trap.config.verbosity import set_verbosity
from trap.utils.seeding import set_global_seed

app = typer.Typer(help="Entropy/redundancy k-mer length analysis (reviewer re-do).")

_SPECTRUM_SORT = ["corpus", "k", "subsample_frac", "replicate"]


def _parse_fracs(text: str) -> List[float]:
    """Parse a comma-separated list of subsample fractions."""
    return [float(x) for x in text.split(",") if x.strip()]


def _write_table(rows: List[dict], out_dir: Path, stem: str, sort_by: List[str]) -> pd.DataFrame:
    """Write *rows* as deterministic CSV + Parquet under *out_dir*.

    Args:
        rows: Tidy records.
        out_dir: Destination directory (created if needed).
        stem: File stem (``<stem>.csv`` / ``<stem>.parquet``).
        sort_by: Columns to sort on for byte-stable CSV output.

    Returns:
        The sorted DataFrame that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    sort_cols = [c for c in sort_by if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False)
    try:
        frame.to_parquet(out_dir / f"{stem}.parquet", index=False)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        logger.warning(f"parquet write skipped ({exc}); CSV is authoritative")
    logger.success(f"wrote {out_dir / stem}.csv ({len(frame)} rows)")
    return frame


@app.command()
def kmer_spectrum(
    corpus: Path = typer.Option(..., help="FASTA/FASTQ corpus path"),
    name: str = typer.Option("corpus", help="Short corpus label written into rows"),
    file_format: str = typer.Option("fasta", help="BioPython SeqIO format"),
    k_min: int = typer.Option(2),
    k_max: int = typer.Option(20),
    subsample_fracs: str = typer.Option("0.1,0.25,0.5,0.75,1.0"),
    replicates: int = typer.Option(3),
    bootstrap: int = typer.Option(200, help="Bootstrap replicates for the entropy CI"),
    alpha: float = typer.Option(0.05),
    canonical: bool = typer.Option(False, help="Collapse reverse complements"),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/entropy_redundancy"), help="Output directory"),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Compute the entropy/redundancy spectrum (with rarefaction) for one corpus."""
    set_verbosity(verbosity)
    set_global_seed(seed)
    k_values = list(range(k_min, k_max + 1))
    logger.log("STAGE", f"[kmer-spectrum] {name}: k={k_min}..{k_max} from {corpus}")
    sequences = load_corpus(corpus, file_format)
    rows = run_kmer_spectrum(
        sequences,
        k_values,
        subsample_fracs=_parse_fracs(subsample_fracs),
        replicates=replicates,
        seed=seed,
        canonical=canonical,
        bootstrap=bootstrap,
        alpha=alpha,
        corpus_name=name,
    )
    _write_table(rows, out, f"{name}.spectrum", _SPECTRUM_SORT)
    manifest.write(
        out,
        command="kmer-spectrum",
        seed=seed,
        params={
            "name": name,
            "k_min": k_min,
            "k_max": k_max,
            "subsample_fracs": _parse_fracs(subsample_fracs),
            "replicates": replicates,
            "bootstrap": bootstrap,
            "canonical": canonical,
        },
        inputs={"corpus": {"path": str(corpus), "sha256": manifest.sha256_file(corpus)}},
    )


@app.command()
def ablation(
    reads: Path = typer.Option(..., help="FASTQ with TrAP labelled headers"),
    k_min: int = typer.Option(2),
    k_max: int = typer.Option(20),
    max_per_class: int = typer.Option(5000),
    n_features: int = typer.Option(1 << 18, help="Hash buckets for featurisation"),
    n_splits: int = typer.Option(5, help="Stratified CV folds"),
    canonical: bool = typer.Option(False),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/entropy_redundancy")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Run the k -> classification macro-F1 ablation."""
    set_verbosity(verbosity)
    set_global_seed(seed)
    k_values = list(range(k_min, k_max + 1))
    logger.log("STAGE", f"[ablation] k={k_min}..{k_max} from {reads}")
    rows = run_ablation(
        reads,
        k_values,
        max_per_class=max_per_class,
        n_features=n_features,
        canonical=canonical,
        n_splits=n_splits,
        seed=seed,
    )
    _write_table(rows, out, "ablation", ["k"])
    manifest.write(
        out,
        command="ablation",
        seed=seed,
        params={
            "k_min": k_min,
            "k_max": k_max,
            "max_per_class": max_per_class,
            "n_features": n_features,
            "n_splits": n_splits,
            "canonical": canonical,
        },
        inputs={"reads": {"path": str(reads), "sha256": manifest.sha256_file(reads)}},
    )


@app.command(name="select-k")
def select_k(
    spectrum: Path = typer.Option(..., help="Spectrum CSV (or directory of *.spectrum.csv)"),
    ablation_csv: Optional[Path] = typer.Option(None, "--ablation", help="ablation.csv"),
    epsilon: float = typer.Option(0.01, help="Saturation gate threshold"),
    tau: float = typer.Option(0.05, help="Marginal-gain gate threshold (bits)"),
    out: Path = typer.Option(Path("results/entropy_redundancy/selection.json")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Apply the quantitative selection criterion to existing tables."""
    set_verbosity(verbosity)
    spectrum_df = _load_spectrum(spectrum)
    ablation_df = pd.read_csv(ablation_csv) if ablation_csv and ablation_csv.exists() else None
    report = compute_selection(spectrum_df, ablation_df, epsilon=epsilon, tau=tau)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for corpus, rep in report.items():
        if corpus == "overall":
            continue
        logger.success(f"[{corpus}] recommended k={rep['recommended_k']} — {rep['verdict']}")


def _load_spectrum(path: Path) -> pd.DataFrame:
    """Load one spectrum CSV or concatenate all ``*.spectrum.csv`` in a dir."""
    if path.is_dir():
        frames = [pd.read_csv(p) for p in sorted(path.glob("*.spectrum.csv"))]
        if not frames:
            raise typer.BadParameter(f"no *.spectrum.csv files in {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path)


@app.command()
def run(
    config: Path = typer.Option(
        Path("config/analysis/entropy_redundancy.yaml"), help="KmerSpectrumConfigSchema YAML"
    ),
) -> None:
    """Full pipeline: every corpus spectrum + ablation + selection from a config."""
    cfg = load_config(config, KmerSpectrumConfigSchema)
    set_verbosity(cfg.verbosity)
    set_global_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    k_values = list(range(cfg.k_min, cfg.k_max + 1))

    # Resolve n_jobs: explicit config value > SLURM_CPUS_PER_TASK > 1.
    n_jobs: int
    if cfg.n_jobs is not None:
        n_jobs = cfg.n_jobs
    else:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger.log("STAGE", f"[run] n_jobs={n_jobs} (from {'config' if cfg.n_jobs is not None else 'SLURM_CPUS_PER_TASK/default'})")

    inputs: dict = {}
    spectrum_rows: List[dict] = []
    _streaming = _seqkit_available() and _jellyfish_available()
    if _streaming:
        logger.log("STAGE", "[run] seqkit+jellyfish detected — streaming path active (no load_corpus)")
    else:
        logger.log("STAGE", "[run] seqkit or jellyfish missing — using legacy load_corpus path")

    for name, path_str in cfg.corpora.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"corpus '{name}' missing at {path}; skipping")
            continue

        if _streaming:
            sequences: list = []
        else:
            logger.log("STAGE", f"[run] corpus {name}: loading {path}")
            sequences = load_corpus(path, cfg.file_format)

        spectrum_rows.extend(
            run_kmer_spectrum(
                sequences,
                k_values,
                subsample_fracs=cfg.subsample_fracs,
                replicates=cfg.replicates,
                seed=cfg.seed,
                canonical=cfg.canonical,
                bootstrap=cfg.bootstrap,
                alpha=cfg.alpha,
                corpus_name=name,
                chunk_size=cfg.chunk_size,
                n_jobs=n_jobs,
                jf_hash_size=cfg.jf_hash_size,
                corpus_path=path,
            )
        )
        inputs[name] = {"path": str(path), "sha256": manifest.sha256_file(path)}

    if not spectrum_rows:
        raise typer.Exit(code=1)
    spectrum_df = _write_table(spectrum_rows, out_dir, "spectrum", _SPECTRUM_SORT)

    ablation_df = None
    reads = Path(cfg.ablation_reads)
    if reads.exists():
        logger.log("STAGE", f"[run] ablation from {reads}")
        ablation_rows = run_ablation(
            reads,
            k_values,
            max_per_class=cfg.ablation_max_per_class,
            n_features=cfg.ablation_n_features,
            canonical=cfg.canonical,
            n_splits=cfg.ablation_n_splits,
            seed=cfg.seed,
        )
        ablation_df = _write_table(ablation_rows, out_dir, "ablation", ["k"])
        inputs["ablation_reads"] = {"path": str(reads), "sha256": manifest.sha256_file(reads)}
    else:
        logger.warning(f"ablation reads missing at {reads}; skipping ablation")

    report = compute_selection(
        spectrum_df, ablation_df, epsilon=cfg.saturation_epsilon, tau=cfg.marginal_gain_tau
    )
    (out_dir / "selection.json").write_text(json.dumps(report, indent=2))
    manifest.write(
        out_dir,
        command="run",
        seed=cfg.seed,
        params=cfg.model_dump(),
        inputs=inputs,
        outputs=["spectrum.csv", "ablation.csv", "selection.json"],
    )
    for corpus, rep in report.items():
        if corpus == "overall":
            continue
        logger.success(f"[{corpus}] recommended k={rep['recommended_k']} — {rep['verdict']}")


if __name__ == "__main__":
    app()
