"""Post-processing: filter read IDs by NEGATIVE class score.

Phase-1 change: ``filter_ids`` now reads Parquet score files (written by the
new ``quantify`` command) in addition to the legacy pickle format.  The Parquet
path is preferred; the pickle path is kept for backward compatibility with
existing runs.
"""

from __future__ import annotations

import glob
import pickle
from pathlib import Path

import typer
from Bio import SeqIO
from datasets import Dataset
from loguru import logger

from trap.utils.io import genome_file_handle

app = typer.Typer(help="Filter read IDs based on class scores from trap quantification.")


def sanitize_paths(*args):
    """Convert input arguments to pathlib.Path objects if possible."""
    paths = []
    for arg in args:
        if isinstance(arg, Path) or arg is None:
            paths.append(arg)
        else:
            try:
                paths.append(Path(arg))
            except Exception as e:
                raise ValueError(f"Invalid path argument: {arg}") from e
    return paths


def processing_ids_from_fastq(input_file: Path, num_workers: int = 4):
    """Extract read IDs from a FASTQ file."""
    (input_file,) = sanitize_paths(input_file)

    def generator_from_iterator():
        with genome_file_handle(input_file) as r1_handle:
            for r1 in SeqIO.parse(r1_handle, "fastq"):
                yield {"id": str(r1.id)}

    logger.info("Creating ID dataset from FASTQ file")
    return Dataset.from_generator(generator_from_iterator, num_proc=num_workers)


def processing_ids_from_list(id_file: Path):
    """Load read IDs from a plain text file (one per line)."""
    (id_file,) = sanitize_paths(id_file)
    with open(id_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({"id": ids})


def _load_scores_parquet(output_path: Path):
    """Load all Parquet score shards from ``output_path`` into a DataFrame."""
    import pandas as pd

    parquet_files = sorted(glob.glob(str(output_path / "scores_*.parquet")))
    if not parquet_files:
        return None
    logger.info(f"Reading {len(parquet_files)} Parquet shard(s)")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


def _load_scores_pickle(pkl_path: Path):
    """Load the legacy pickle score file and return a DataFrame."""
    import pandas as pd

    with open(pkl_path, "rb") as f:
        scores = pickle.load(f)

    # Legacy format: list of lists of dicts [{label, score}, ...]
    rows = []
    for entry in scores:
        row = {d["label"]: d["score"] for d in entry}
        rows.append(row)
    return pd.DataFrame(rows)


@app.command()
def filter_ids(
    fastq: Path = typer.Option(None, help="Input FASTQ file"),
    id_list: Path = typer.Option(None, help="Text file with read IDs, one per line"),
    output_path: Path = typer.Option(..., help="Directory containing score files"),
    threshold: float = typer.Option(0.5, help="Max NEGATIVE score to keep a read"),
    output_filtered_ids: Path = typer.Option(None, help="File to write filtered IDs"),
    num_workers: int = typer.Option(4),
):
    """Filter reads whose NEGATIVE class score is below ``threshold``.

    Reads Parquet shards (``scores_*.parquet``) produced by ``quantify``.
    Falls back to the legacy ``class_scores.pkl`` if no Parquet files exist.

    The Parquet path uses polars ``scan_parquet`` (DR-5) so no full file is
    loaded into RAM — the filter and ``id`` projection run lazily.  For legacy
    pickle files the IDs are joined by position from ``--fastq`` or
    ``--id-list``.
    """
    if not fastq and not id_list:
        typer.echo("Error: provide --fastq or --id-list.")
        raise typer.Exit(code=1)

    (output_path,) = sanitize_paths(output_path)
    parquet_files = sorted(glob.glob(str(output_path / "scores_*.parquet")))

    if parquet_files:
        # DR-5: streaming polars path — no full load into RAM.
        try:
            import polars as pl

            lazy = pl.scan_parquet(parquet_files)
            # collect_schema().names() is the non-deprecated way to read column
            # names from a LazyFrame without materializing it (LazyFrame.columns
            # is deprecated / removed in recent polars).
            if "NEGATIVE" not in lazy.collect_schema().names():
                typer.echo("Error: Parquet files lack a 'NEGATIVE' column.")
                raise typer.Exit(code=1)
            result_ids = (
                lazy.filter(pl.col("NEGATIVE") < threshold).select("id").collect()["id"].to_list()
            )
        except ImportError:
            # polars not installed — fall back to pandas
            import pandas as pd

            df = _load_scores_parquet(output_path)
            if "NEGATIVE" not in df.columns:
                typer.echo("Error: Parquet files lack a 'NEGATIVE' column.")
                raise typer.Exit(code=1)
            result_ids = df.loc[df["NEGATIVE"] < threshold, "id"].tolist()

        logger.info(f"Retained {len(result_ids)} reads (NEGATIVE < {threshold})")

    else:
        # Legacy pickle path
        pkl_path = output_path / "class_scores.pkl"
        if not pkl_path.exists():
            typer.echo(f"Error: no Parquet shards or class_scores.pkl in {output_path}")
            raise typer.Exit(code=1)

        if fastq:
            id_dataset = processing_ids_from_fastq(fastq, num_workers=num_workers)
        else:
            id_dataset = processing_ids_from_list(id_list)

        scores_df = _load_scores_pickle(pkl_path)
        if len(id_dataset) != len(scores_df):
            raise ValueError(
                f"Length mismatch: {len(id_dataset)} IDs vs {len(scores_df)} score rows"
            )
        scores_df.index = id_dataset["id"]
        filtered = scores_df[scores_df["NEGATIVE"] < threshold]
        logger.info(f"Retained {len(filtered)} / {len(scores_df)} reads (NEGATIVE < {threshold})")
        result_ids = list(filtered.index)

    if output_filtered_ids:
        with open(output_filtered_ids, "w") as f:
            f.write("\n".join(result_ids))
        typer.echo(f"Filtered IDs saved to {output_filtered_ids}")
    else:
        typer.echo("\n".join(result_ids))


if __name__ == "__main__":
    app()
