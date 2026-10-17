#!/usr/bin/env python3
"""Stage-A translator for the SRR3997504+SRR4099955 salmon/L1EM experiment figure (5.06).

Join five per-L1-locus quantification tables on the locus id (``region``) and emit the tidy
long table

    results/srr_experiment/l1em_salmon_fpm.csv
        columns: region, chrom, chromStart, chromEnd, target, category, strand, race5, method, value

- ``race5``  = 5'RACE percentage (100 * featureCounts count / total) from the SRR4099955
  5'RACE experiment (``counts.l1em.txt``) — the ground-truth x-axis, shared across methods.
- ``method`` in {AlbertSalmon, Salmon, AlbertEM, L1EM}; ``value`` = that method's per-locus
  estimate: salmon TPM (AlbertSalmon = classifier-filtered, Salmon = plain) or EM rpm
  (only + 3prunon; AlbertEM = ML-L1EM on SRR3997504, L1EM = plain L1EM).

The Stage-B notebook (5.06) regresses each method's ``value`` against ``race5`` per panel.

Runs on Grace where the raw tables live (gitignored). CANNOT run off-HPC — the durable,
reproducible artifact is the emitted CSV.

    python scripts/aggregate_srr_l1em_exp.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import reduce
import json
from pathlib import Path
import subprocess

from loguru import logger
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help=__doc__)

# family.category.locus.strand[::chrom:start-end]
_FAMILY = r"([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)"
_LOCUS = _FAMILY + r"::(chr[^:]+):(\d+)-(\d+)"
_META = ["target", "category", "region", "strand", "chrom", "chromStart", "chromEnd"]


def load_fpm(path: Path, name: str) -> pd.DataFrame:
    """L1EM / MLEM ``L1HS-L1PA_FPM.txt`` -> region + rpm(only+3prunon), renamed to ``name``."""
    df = pd.read_csv(path, sep="\t")
    meta = df["family.category.locus.strand"].str.extract(_LOCUS)
    meta.columns = _META
    out = pd.concat([meta, df[["only", "3prunon"]].astype(float)], axis=1)
    out[name] = out["only"] + out["3prunon"]
    return out[_META + [name]]


def load_salmon(path: Path, name: str) -> pd.DataFrame:
    """salmon quant.sf (L1EM-named targets) -> region + TPM, renamed to ``name``."""
    df = pd.read_csv(path, sep="\t", comment="#")
    meta = df["Name"].str.extract(_LOCUS)
    meta.columns = _META
    out = pd.concat([meta, df[["TPM"]].astype(float).rename(columns={"TPM": name})], axis=1)
    return out[_META + [name]]


def load_race5(path: Path) -> pd.DataFrame:
    """SRR4099955 5'RACE featureCounts ``counts.l1em.txt`` -> region + race5 (percentage)."""
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df.rename(columns={df.columns[-1]: "count"})
    fam = df["Geneid"].str.extract(_FAMILY)
    fam.columns = ["target", "category", "region", "strand"]
    out = pd.concat(
        [
            df[["Chr", "Start", "End", "Length"]].rename(
                columns={
                    "Chr": "chrom",
                    "Start": "chromStart",
                    "End": "chromEnd",
                    "Length": "length",
                }
            ),
            fam,
            df[["count"]].astype(float),
        ],
        axis=1,
    )
    total = out["count"].sum()
    out["race5"] = 100.0 * out["count"] / total if total else 0.0
    return out[
        ["chrom", "chromStart", "chromEnd", "target", "category", "region", "strand", "race5"]
    ]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@app.command()
def main(
    race5_counts: Path = typer.Option("run/outputs/align_quant/bwa/SRR4099955/counts.l1em.txt"),
    l1em_fpm: Path = typer.Option("run/outputs/l1em/bwa/SRR3997504/L1HS-L1PA_FPM.txt"),
    mlem_fpm: Path = typer.Option("run/outputs/mlem/SRR3997504/L1HS-L1PA_FPM.txt"),
    salmon_quant: Path = typer.Option("run/outputs/salmon/SRR3997504/only/quant.sf"),
    albertsalmon_quant: Path = typer.Option(
        "run/outputs/salmon/SRR3997504/filtered/only/quant.sf"
    ),
    out_dir: Path = typer.Option("results/srr_experiment"),
):
    """Join the per-locus tables on region and write the tidy long table + manifest."""
    logger.info("loading per-locus quantifications")
    base = load_race5(race5_counts)  # region + metadata + race5 (the x-axis)
    method_tables = [
        load_salmon(albertsalmon_quant, "AlbertSalmon")[["region", "AlbertSalmon"]],
        load_salmon(salmon_quant, "Salmon")[["region", "Salmon"]],
        load_fpm(mlem_fpm, "AlbertEM")[["region", "AlbertEM"]],
        load_fpm(l1em_fpm, "L1EM")[["region", "L1EM"]],
    ]

    wide = reduce(
        lambda left, right: pd.merge(left, right, on="region", how="outer"), [base] + method_tables
    )
    wide["category"] = pd.to_numeric(wide["category"], errors="coerce")

    long = wide.melt(
        id_vars=[
            "region",
            "chrom",
            "chromStart",
            "chromEnd",
            "target",
            "category",
            "strand",
            "race5",
        ],
        value_vars=["AlbertSalmon", "Salmon", "AlbertEM", "L1EM"],
        var_name="method",
        value_name="value",
    )
    long = long.sort_values(["method", "region"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "l1em_salmon_fpm.csv"
    long.to_csv(csv_path, index=False)
    manifest = {
        "trap_version": "unknown",
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 3469,
        "inputs": {
            "race5_counts": str(race5_counts),
            "l1em_fpm": str(l1em_fpm),
            "mlem_fpm": str(mlem_fpm),
            "salmon_quant": str(salmon_quant),
            "albertsalmon_quant": str(albertsalmon_quant),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.success(f"wrote {len(long):,} rows ({long.method.nunique()} methods) -> {csv_path}")


if __name__ == "__main__":
    app()
