"""Typer CLI for the entropy/redundancy k-mer analysis.

Run as ``python -m trap.analysis``.  Commands:

* ``run``           — full config-driven pipeline (all corpora + ablation + select-k)
* ``kmer-spectrum`` — entropy/redundancy spectrum for one corpus
* ``ablation``      — k -> classification macro-F1 sweep
* ``select-k``      — apply the quantitative selection criterion to existing tables
* ``robustness``    — 5'RACE locus-set / leave-one-out / bootstrap / L1Base tables
* ``synthetic-strata`` — stratified statistics for the synthetic insertion benchmark
* ``enrichment``    — domain + ORF0 enrichment of top-attribution k-mers
* ``negative-control`` — is the NEGATIVE class a clean control? (Figure 13)
* ``chrom-length``   — are the Figure 13 correlations carried by chromosome size?
* ``imbalance``     — class weights, imbalance ratios, L1HS deficit propagation
* ``heldout-eval``  — score a held-out split once, eval-dataloader cache disabled
* ``model-params``  — parameter count of the reported classifier, two ways
* ``enrichment-clustered`` — attribution enrichment re-tested with the read as unit
* ``seed-summary``  — aggregate the seed replicates into the stability table

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

from trap.analysis.ablation import ablation as run_ablation
from trap.analysis.enrichment import (
    verify_against_published,
)
from trap.analysis.enrichment import clustered_enrichment as run_clustered_enrichment
from trap.analysis.enrichment import enrichment as run_enrichment
from trap.analysis.imbalance import imbalance as run_imbalance
from trap.analysis.kmer_counting import (
    _jellyfish_available,
    _seqkit_available,
)
from trap.analysis.kmer_counting import (
    load_corpus,
)
from trap.analysis.kmer_counting import kmer_spectrum as run_kmer_spectrum
from trap.analysis.negative_control import (
    load_by_chromosome,
    load_chrom_lengths,
)
from trap.analysis.negative_control import chrom_length_correlations as run_chrom_length
from trap.analysis.negative_control import negative_control as run_negative_control
from trap.analysis.robustness import (
    DEFAULT_N_BOOT,
    L1BASE_TRACKS,
    MIN_ELEMENT_BP,
    PRIMARY_LOCUS_SET,
)
from trap.analysis.robustness import robustness as run_robustness
from trap.analysis.selection import compute_selection
from trap.analysis.synthetic_strata import (
    REFERENCE_METHOD,
)
from trap.analysis.synthetic_strata import synthetic_strata as run_synthetic_strata
from trap.config import manifest
import trap.config.config  # noqa: F401  (registers STAGE level + tqdm-loguru)
from trap.config.config import MODELS_DIR
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
def robustness(
    table: Path = typer.Option(
        Path("results/srr_experiment/l1em_salmon_fpm.csv"),
        help="Tidy 5'RACE comparison table (locus x method)",
    ),
    l1base_dir: Optional[Path] = typer.Option(
        Path("data/ref/l1base"),
        "--l1base-dir",
        help="Directory of L1Base2 BED tracks for the orthogonal-annotation analyses",
    ),
    min_element_bp: int = typer.Option(
        MIN_ELEMENT_BP, help="Keep L1Base elements whose body exceeds this length"
    ),
    locus_set: str = typer.Option(
        PRIMARY_LOCUS_SET, help="Primary locus set for leave-one-out and the bootstrap"
    ),
    n_boot: int = typer.Option(DEFAULT_N_BOOT, help="Bootstrap resamples for the dR2 CIs"),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/srr_experiment"), help="Output directory"),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Robustness of the 5'RACE concordance benchmark (locus sets, leverage, dR2 CIs)."""
    set_verbosity(verbosity)
    set_global_seed(seed)
    logger.log("STAGE", f"[robustness] {table} -> {out} (primary set {locus_set})")
    l1base = l1base_dir if l1base_dir and l1base_dir.is_dir() else None
    if l1base_dir and l1base is None:
        logger.warning(f"L1Base directory missing at {l1base_dir}; overlap analyses skipped")
    tables = run_robustness(
        table,
        l1base_dir=l1base,
        locus_set=locus_set,
        n_boot=n_boot,
        seed=seed,
        min_element_bp=min_element_bp,
    )
    for stem, frame in tables.items():
        # Row order is fixed by construction (definition x method), so no sort
        # key is needed for byte-stable output.
        _write_table(frame.to_dict("records"), out, stem, [])
    inputs = {"table": {"path": str(table), "sha256": manifest.sha256_file(table)}}
    if l1base is not None:
        for track in L1BASE_TRACKS:
            path = l1base / track.filename
            if path.exists():
                inputs[f"l1base_{track.name}"] = {
                    "path": str(path),
                    "sha256": manifest.sha256_file(path),
                }
    manifest.write(
        out,
        filename="manifest.robustness.json",
        command="robustness",
        seed=seed,
        params={
            "locus_set": locus_set,
            "n_boot": n_boot,
            "min_element_bp": min_element_bp,
        },
        inputs=inputs,
        outputs=[f"{stem}.csv" for stem in tables],
    )


@app.command()
def enrichment(
    attribution: Path = typer.Option(
        Path("results/interpret/per_kmer_attribution.csv"),
        help="per_kmer_attribution.csv from `trap.analysis.interpret attribute`",
    ),
    published: Optional[Path] = typer.Option(
        Path("results/interpret/domain_enrichment.csv"),
        help="Published domain_enrichment.csv, checked for exact reproduction",
    ),
    out: Path = typer.Option(Path("results/interpret")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Domain + ORF0 enrichment of top-attribution k-mers (no model re-run)."""
    set_verbosity(verbosity)
    logger.log("STAGE", f"[enrichment] {attribution} -> {out}")
    tables = run_enrichment(attribution)
    if published and published.exists():
        report = verify_against_published(tables["domain_enrichment.recomputed"], published)
        tables["domain_enrichment.verification"] = report
        if not report["matches"].all():
            raise typer.Exit(code=1)
    for stem, frame in tables.items():
        # Row order follows the consensus, so no sort key is needed.
        _write_table(frame.to_dict("records"), out, stem, [])
    manifest.write(
        out,
        filename="manifest.enrichment.json",
        command="enrichment",
        seed=3469,
        params={"top_quantile": 0.90},
        inputs={
            "attribution": {
                "path": str(attribution),
                "sha256": manifest.sha256_file(attribution),
            }
        },
        outputs=[f"{stem}.csv" for stem in tables],
    )


@app.command()
def imbalance(
    composition: Path = typer.Option(
        Path("results/hpo_classification/dataset_composition.csv"),
        help="Per-split class counts",
    ),
    metrics: Path = typer.Option(
        Path("results/hpo_classification/per_class_metrics_table.csv"),
        help="Per-class precision/recall table",
    ),
    locus_table: Path = typer.Option(
        Path("results/srr_experiment/l1em_salmon_fpm.csv"),
        help="Tidy 5'RACE comparison table, for the propagation sensitivity",
    ),
    minority: str = typer.Option("L1HS", help="Rare class"),
    out: Path = typer.Option(Path("results/hpo_classification")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Class imbalance: weights, ratios, and how the L1HS deficit propagates."""
    set_verbosity(verbosity)
    logger.log("STAGE", f"[imbalance] {composition} -> {out}")
    tables = run_imbalance(composition, metrics, locus_table, minority=minority)
    for stem, frame in tables.items():
        _write_table(frame.to_dict("records"), out, stem, [])
    manifest.write(
        out,
        filename="manifest.imbalance.json",
        command="imbalance",
        seed=3469,
        params={"minority": minority},
        inputs={
            name: {"path": str(path), "sha256": manifest.sha256_file(path)}
            for name, path in (
                ("composition", composition),
                ("metrics", metrics),
                ("locus_table", locus_table),
            )
        },
        outputs=[f"{stem}.csv" for stem in tables],
    )


@app.command(name="negative-control")
def negative_control_cmd(
    by_chrom: Path = typer.Option(
        Path("results/srr4099955/aggregate_by_chr.csv"),
        help="Per-chromosome class aggregation",
    ),
    confusion: Optional[Path] = typer.Option(
        Path("results/subfamily_confusion.salmon.csv"),
        help="Subfamily confusion table, for the leakage mechanism",
    ),
    annotation: Optional[Path] = typer.Option(
        Path("L1EM/annotation/L1EM.400.bed"),
        help="LINE-1 annotation BED, for the annotation-density adjustment",
    ),
    logits: Optional[Path] = typer.Option(
        Path("results/eval_logits.salmon.npz"),
        help="Held-out evaluation logits, for the deployment-prior leak rates",
    ),
    out: Path = typer.Option(Path("results/srr4099955")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Test whether the NEGATIVE class is a clean control (R2-9, Figure 13)."""
    set_verbosity(verbosity)
    logger.log("STAGE", f"[negative-control] {by_chrom} -> {out}")
    cm = confusion if confusion and confusion.exists() else None
    if confusion and cm is None:
        logger.warning(f"confusion table missing at {confusion}; leakage table skipped")
    ann = annotation if annotation and annotation.exists() else None
    if annotation and ann is None:
        logger.warning(f"annotation BED missing at {annotation}; density adjustment skipped")
    lg = logits if logits and logits.exists() else None
    if logits and lg is None:
        logger.warning(f"eval logits missing at {logits}; operating-point table skipped")
    tables = run_negative_control(by_chrom, confusion_path=cm, annotation_path=ann, logits_path=lg)
    for stem, frame in tables.items():
        _write_table(frame.to_dict("records"), out, stem, [])
    inputs = {"by_chrom": {"path": str(by_chrom), "sha256": manifest.sha256_file(by_chrom)}}
    if cm is not None:
        inputs["confusion"] = {"path": str(cm), "sha256": manifest.sha256_file(cm)}
    if ann is not None:
        inputs["annotation"] = {"path": str(ann), "sha256": manifest.sha256_file(ann)}
    if lg is not None:
        inputs["logits"] = {"path": str(lg), "sha256": manifest.sha256_file(lg)}
    manifest.write(
        out,
        filename="manifest.negative_control.json",
        command="negative-control",
        seed=3469,
        params={"figure": "SRR4099955_agreggate_mlvs_bwa_by_chr"},
        inputs=inputs,
        outputs=[f"{stem}.csv" for stem in tables],
    )


@app.command(name="chrom-length")
def chrom_length_cmd(
    by_chrom: Path = typer.Option(
        Path("results/srr4099955/aggregate_by_chr.csv"),
        help="Per-chromosome class aggregation",
    ),
    chrom_sizes: Path = typer.Option(
        Path("references/GRCh38.chrom.sizes"),
        help="Two-column chrom.sizes or .fai giving chromosome lengths",
    ),
    out: Path = typer.Option(Path("results/srr4099955")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Adjust the Figure 13 correlations for chromosome length (R2-9).

    Deliberately a separate command from ``negative-control``: it needs only
    the aggregation and the assembly, so it runs anywhere, and it writes one
    table of its own rather than rewriting that command's five.
    """
    set_verbosity(verbosity)
    logger.log("STAGE", f"[chrom-length] {by_chrom} -> {out}")
    classifier, alignment = load_by_chromosome(by_chrom)
    lengths = load_chrom_lengths(chrom_sizes)
    table = run_chrom_length(classifier, alignment, lengths)
    _write_table(table.to_dict("records"), out, "negative_control_chrom_length", [])
    focus = table[
        (table["classifier_class"] == "NEGATIVE") & (table["alignment_class"] == "L1PA2")
    ].iloc[0]
    logger.success(
        f"NEGATIVE vs alignment L1PA2: r={focus['pearson_r']:.3f} -> "
        f"partial r | chromosome length = {focus['partial_r_given_chrom_length']:.3f} "
        f"(p={focus['partial_p_given_chrom_length']:.4f})"
    )
    diagonal = table[table["classifier_class"] == table["alignment_class"]]
    for _, row in diagonal.iterrows():
        logger.info(
            f"  {row['classifier_class']} diagonal: {row['pearson_r']:+.3f} -> "
            f"{row['partial_r_given_chrom_length']:+.3f} (delta {row['delta_r']:+.3f})"
        )
    manifest.write(
        out,
        filename="manifest.chrom_length.json",
        command="chrom-length",
        seed=3469,
        params={
            "figure": "SRR4099955_agreggate_mlvs_bwa_by_chr",
            "assembly": "GRCh38.p14 (GCA_000001405.29)",
        },
        inputs={
            "by_chrom": {"path": str(by_chrom), "sha256": manifest.sha256_file(by_chrom)},
            "chrom_sizes": {
                "path": str(chrom_sizes),
                "sha256": manifest.sha256_file(chrom_sizes),
            },
        },
        outputs=["negative_control_chrom_length.csv"],
    )


@app.command(name="heldout-eval")
def heldout_eval_cmd(
    model: str = typer.Option(
        "albert.l1hs_l1pa2.v48.k17.salmon.seqlabel",
        help="Model directory under MODELS_DIR; its final/ checkpoint is scored",
    ),
    preprocessing_name: str = typer.Option(
        "gencode.v48.k17.salmon.seqlabel/l1hs_l1pa2",
        help="Tokenized classification dataset name",
    ),
    split: str = typer.Option("eval", help="DatasetDict split to score (Table 1 held-out = eval)"),
    batch_size: int = typer.Option(128, help="Per-device eval batch size"),
    num_workers: int = typer.Option(8, help="Dataloader workers"),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/heldout")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Score a held-out split once, with the eval-dataloader cache disabled.

    Trains nothing. See ``trap.analysis.heldout_eval`` for why the split named
    ``eval`` was never actually scored by the training run of record.
    """
    import numpy as np

    from trap.analysis.heldout_eval import _load_split, _metric_frames, _predict_logits

    set_verbosity(verbosity)
    set_global_seed(seed)
    model_path = MODELS_DIR / model / "final"
    logger.log("STAGE", f"[heldout-eval] {model_path} x {preprocessing_name}:{split} -> {out}")

    dataset, id2label = _load_split(preprocessing_name, split)
    labels = np.asarray(dataset["label"])
    logits, precision = _predict_logits(model_path, dataset, batch_size, num_workers)

    per_class, confusion, summary = _metric_frames(logits, labels, id2label)
    _write_table(per_class.to_dict("records"), out, f"heldout_per_class.{split}", [])
    _write_table(confusion.to_dict("records"), out, f"heldout_confusion.{split}", [])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"heldout_logits.{split}.npz", logits=logits, labels=labels)
    (out / f"heldout_summary.{split}.json").write_text(
        json.dumps({**summary, "compute_precision": precision}, indent=2, default=float)
    )

    logger.success(
        f"[heldout-eval] n={len(labels):,} macro-F1={summary['f1']:.4f} "
        f"accuracy={summary['accuracy']:.4f}"
    )
    manifest.write(
        out,
        filename=f"manifest.heldout_eval.{split}.json",
        command="heldout-eval",
        seed=seed,
        params={
            "model": model,
            "split": split,
            "batch_size": batch_size,
            "n_rows": len(labels),
            "precision": precision,
        },
        inputs={
            "model": {
                "path": str(model_path),
                "sha256": manifest.sha256_file(model_path / "model.safetensors"),
            },
            "dataset": {"path": f"{preprocessing_name}/classification/tokenized/{split}"},
        },
        outputs=[
            f"heldout_per_class.{split}.csv",
            f"heldout_confusion.{split}.csv",
            f"heldout_logits.{split}.npz",
            f"heldout_summary.{split}.json",
        ],
    )


@app.command(name="enrichment-clustered")
def enrichment_clustered_cmd(
    attribution: Path = typer.Option(
        Path("results/interpret/per_kmer_attribution.csv"),
        help="Per-k-mer attribution table; must carry read_uid",
    ),
    domain: str = typer.Option("ORF2_C", help="Region to re-test against the rest"),
    n_boot: int = typer.Option(2000, help="Cluster-bootstrap resamples over reads"),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/interpret")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Re-test the attribution enrichment with the read as the unit (R3, C9).

    The published Fisher interval assumes independent k-mers; this replaces it
    with a read-clustered one and reports the design effect, so the inflation is
    quantified rather than merely conceded.
    """
    set_verbosity(verbosity)
    set_global_seed(seed)
    logger.log("STAGE", f"[enrichment-clustered] {attribution} -> {out}")
    tables = run_clustered_enrichment(attribution, domain=domain, n_boot=n_boot, seed=seed)
    for name, frame in tables.items():
        _write_table(frame.to_dict("records"), out, f"{name}.{domain}", [])
    row = tables["clustered_enrichment"].iloc[0]
    manifest.write(
        out,
        filename=f"manifest.enrichment_clustered.{domain}.json",
        command="enrichment-clustered",
        seed=seed,
        params={
            "domain": domain,
            "n_boot": n_boot,
            "n_reads": int(row["n_reads"]),
            "n_kmers": int(row["n_kmers"]),
            "odds_ratio": float(row["odds_ratio"]),
            "or_ci_clustered": [
                float(row["or_ci_lo_clustered"]),
                float(row["or_ci_hi_clustered"]),
            ],
            "design_effect": float(row["design_effect"]),
        },
        inputs={
            "attribution": {
                "path": str(attribution),
                "sha256": manifest.sha256_file(attribution),
            }
        },
        outputs=[f"clustered_enrichment.{domain}.csv", f"clustered_reads.{domain}.csv"],
    )


@app.command(name="seed-summary")
def seed_summary_cmd(
    seeds_dir: Path = typer.Option(
        Path("results/seeds"), help="Directory of per-replicate heldout-eval outputs"
    ),
    reference: Path = typer.Option(
        Path("results/heldout_seqlabel/heldout_summary.eval.json"),
        help="Held-out summary of the model of record (seed 3469)",
    ),
    out: Path = typer.Option(Path("results/seeds")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Aggregate the seed replicates into the stability table (R3, Comment 5).

    Reads each replicate's ``heldout_summary.eval.*.json`` and reports mean, sd
    and range per metric, with the run of record shown alongside so the reader
    can see whether it is typical of the ensemble rather than a lucky draw.
    """
    import json as _json

    set_verbosity(verbosity)
    logger.log("STAGE", f"[seed-summary] {seeds_dir} -> {out}")

    rows = []
    for summary_path in sorted(seeds_dir.glob("*/heldout_summary.eval*.json")):
        payload = _json.loads(summary_path.read_text())
        name = summary_path.parent.name
        rows.append(
            {
                "model": name,
                "seed": int(name.rsplit("seed", 1)[-1]) if "seed" in name else -1,
                **{k: v for k, v in payload.items() if isinstance(v, (int, float))},
            }
        )
    if not rows:
        raise typer.BadParameter(
            f"no replicate summaries under {seeds_dir}/*/heldout_summary.eval*.json; "
            "run stage 43 first"
        )
    frame = pd.DataFrame(rows).sort_values("seed")

    metrics = [
        m
        for m in ("accuracy", "f1", "precision", "recall", "f1_L1HS", "f1_L1PA", "f1_NEGATIVE")
        if m in frame.columns
    ]
    ref = {}
    if reference.exists():
        ref = _json.loads(reference.read_text())

    stats = pd.DataFrame(
        [
            {
                "metric": m,
                "n_replicates": int(frame[m].notna().sum()),
                "mean": float(frame[m].mean()),
                "sd": float(frame[m].std(ddof=1)) if frame[m].notna().sum() > 1 else float("nan"),
                "min": float(frame[m].min()),
                "max": float(frame[m].max()),
                "run_of_record": float(ref[m]) if m in ref else float("nan"),
            }
            for m in metrics
        ]
    )
    _write_table(frame.to_dict("records"), out, "seed_replicates", [])
    _write_table(stats.to_dict("records"), out, "seed_stability", [])
    for _, r in stats.iterrows():
        logger.info(
            f"  {r['metric']:<12} {r['mean']:.4f} +/- {r['sd']:.4f} "
            f"[{r['min']:.4f}, {r['max']:.4f}]  record={r['run_of_record']:.4f}"
        )
    logger.success(f"[seed-summary] {len(frame)} replicates -> {out}/seed_stability.csv")
    manifest.write(
        out,
        filename="manifest.seed_summary.json",
        command="seed-summary",
        params={
            "n_replicates": int(len(frame)),
            "seeds": [int(x) for x in frame["seed"].tolist()],
            "reference": str(reference),
        },
        inputs={"seeds_dir": {"path": str(seeds_dir)}},
        outputs=["seed_replicates.csv", "seed_stability.csv"],
    )


@app.command(name="model-params")
def model_params_cmd(
    model: str = typer.Option(
        "albert.l1hs_l1pa2.v48.k17.salmon.seqlabel",
        help="Model directory under MODELS_DIR; its final/ checkpoint is counted",
    ),
    out: Path = typer.Option(Path("results/model_params")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Count the reported classifier's parameters, with a manifest (R3, Minor).

    The Methods quote a parameter count, so it must come from the checkpoint of
    record rather than from a hand tally of the config. Two independent counts are
    taken and required to agree: ``num_parameters()`` on the instantiated model,
    and the summed element count of every tensor in ``model.safetensors`` read
    straight from the safetensors header. They can legitimately differ when a
    checkpoint ties weights, so the command fails rather than silently reporting
    one of them.
    """
    import json as _json

    from safetensors import safe_open
    from transformers import AlbertForSequenceClassification

    set_verbosity(verbosity)
    model_path = MODELS_DIR / model / "final"
    logger.log("STAGE", f"[model-params] {model_path} -> {out}")

    net = AlbertForSequenceClassification.from_pretrained(str(model_path))
    live = net.num_parameters()

    weights = model_path / "model.safetensors"
    per_tensor = {}
    with safe_open(str(weights), framework="numpy") as fh:
        for key in fh.keys():
            shape = tuple(fh.get_slice(key).get_shape())
            n = 1
            for dim in shape:
                n *= dim
            per_tensor[key] = {"shape": list(shape), "n_parameters": n}
    checkpoint = sum(v["n_parameters"] for v in per_tensor.values())

    if live != checkpoint:
        raise typer.BadParameter(
            f"parameter counts disagree: num_parameters()={live:,} but the "
            f"checkpoint holds {checkpoint:,} elements across {len(per_tensor)} "
            "tensors. One of them is not what the Methods should quote; resolve "
            "before using either (tied weights are the usual cause)."
        )

    embedding = net.albert.embeddings.word_embeddings.weight.numel()
    cfg = net.config
    summary = {
        "n_parameters": live,
        "n_tensors": len(per_tensor),
        "word_embedding_parameters": embedding,
        "word_embedding_fraction": embedding / live,
        "classifier_head_parameters": sum(p.numel() for p in net.classifier.parameters()),
        "vocab_size": cfg.vocab_size,
        "embedding_size": cfg.embedding_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_hidden_groups": cfg.num_hidden_groups,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "num_labels": cfg.num_labels,
        "id2label": {int(k): v for k, v in cfg.id2label.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_params.json").write_text(_json.dumps(summary, indent=2))
    _write_table(
        [{"tensor": k, **v} for k, v in sorted(per_tensor.items())],
        out,
        "model_params_by_tensor",
        [],
    )

    logger.success(
        f"[model-params] {live:,} parameters in {len(per_tensor)} tensors "
        f"(both counts agree); word embeddings are {embedding:,} "
        f"({100 * embedding / live:.1f}%)"
    )
    logger.info(
        f"  ALBERT shares one layer group across {cfg.num_hidden_layers} layers "
        f"(num_hidden_groups={cfg.num_hidden_groups}), which is why the encoder is "
        "far smaller than the layer count suggests."
    )
    manifest.write(
        out,
        filename="manifest.model_params.json",
        command="model-params",
        params={
            "model": model,
            "n_parameters": live,
            "counts_agree": True,
        },
        inputs={
            "model": {"path": str(model_path), "sha256": manifest.sha256_file(weights)},
            "config": {
                "path": str(model_path / "config.json"),
                "sha256": manifest.sha256_file(model_path / "config.json"),
            },
        },
        outputs=["model_params.json", "model_params_by_tensor.csv"],
    )


@app.command(name="synthetic-strata")
def synthetic_strata_cmd(
    synthetic: Path = typer.Option(
        Path("results/synthetic/chr1"),
        help="Directory holding by_insertion_rate.csv and by_del_probability.csv",
    ),
    reference: str = typer.Option(REFERENCE_METHOD, help="Method used as the contrast reference"),
    n_boot: int = typer.Option(5000, help="Bootstrap resamples for the mean-dR2 CI"),
    seed: int = typer.Option(3469),
    out: Path = typer.Option(Path("results/synthetic/chr1")),
    verbosity: str = typer.Option("normal"),
) -> None:
    """Stratified statistics for Figure 10 (dispersion, Wilcoxon, insertion load)."""
    set_verbosity(verbosity)
    set_global_seed(seed)
    logger.log("STAGE", f"[synthetic-strata] {synthetic} -> {out}")
    tables = run_synthetic_strata(synthetic, reference=reference, n_boot=n_boot, seed=seed)
    for stem, frame in tables.items():
        # Row order follows the stored tables, so no sort key is needed.
        _write_table(frame.to_dict("records"), out, stem, [])
    inputs = {
        name: {"path": str(synthetic / name), "sha256": manifest.sha256_file(synthetic / name)}
        for name in ("by_insertion_rate.csv", "by_del_probability.csv")
    }
    manifest.write(
        out,
        filename="manifest.synthetic_strata.json",
        command="synthetic-strata",
        seed=seed,
        params={"reference": reference, "n_boot": n_boot},
        inputs=inputs,
        outputs=[f"{stem}.csv" for stem in tables],
    )


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
    logger.log(
        "STAGE",
        f"[run] n_jobs={n_jobs} (from {'config' if cfg.n_jobs is not None else 'SLURM_CPUS_PER_TASK/default'})",
    )

    inputs: dict = {}
    spectrum_rows: List[dict] = []
    _streaming = _seqkit_available() and _jellyfish_available()
    if _streaming:
        logger.log(
            "STAGE", "[run] seqkit+jellyfish detected — streaming path active (no load_corpus)"
        )
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
