#!/usr/bin/env python3
"""Stage-A translator for the SRR "aggregate by chromosome" figures (5.04 / 5.05).

It reads the raw quantification artifacts (per-read Albert class scores, aligner BAMs,
RepeatMasker L1 loci, the L1EM reference BED and its BWA re-alignment) and emits ONE tidy
long table per sample plus a ``manifest.json``:

    results/<sample>/aggregate_by_chr.csv
        columns: sample, figure, chrom, method, class, value

Keyed by ``figure`` because the same-named methods (e.g. FILTERED_STAR) carry different
per-chromosome sums across figures: each figure combines its method pieces under its own
join semantics (outer with fill vs inner) before summing, so each figure is its own
aggregation block. The Stage-B R notebooks subset this table by ``figure``.

Runs on Grace where the raw artifacts live (all gitignored). It CANNOT run off-HPC — the
durable, reproducible artifact is the emitted CSV, which the notebooks read.

    python scripts/aggregate_srr_by_chr.py srr4099955
    python scripts/aggregate_srr_by_chr.py srr3997504
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import subprocess

from datasets import load_from_disk
from loguru import logger
import numpy as np
import pandas as pd
import pysam
import typer

app = typer.Typer(add_completion=False, help=__doc__)

# ---------------------------------------------------------------------------
# Chromosome ordering / labels
# ---------------------------------------------------------------------------
STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrMT", "chrM"]


def chr_sort_key(chrom: str) -> int:
    """chr1..chr22 -> 1..22, chrX/Y/M(T) -> 23/24/25, else 26."""
    tail = chrom[3:] if chrom.startswith("chr") else chrom
    if tail.isdigit():
        return int(tail)
    return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(tail, 26)


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
def load_rm_line1(rm_outfile: Path) -> dict:
    """RepeatMasker ``.out`` -> {id: [locus dicts]}."""
    rmtarget: dict = {}
    with open(rm_outfile) as infile:
        for line in infile.readlines()[3:]:  # skip the 3 header lines
            c = line.strip().split()
            if len(c) < 15:
                continue
            rmtarget.setdefault(c[14], []).append(
                {
                    "query_name": c[4],
                    "query_start": c[5],
                    "query_end": c[6],
                    "orientation": "-" if c[8] == "C" else "+",
                    "id": c[14],
                    "target": c[9],
                    "family": c[10],
                }
            )
    return rmtarget


def alignments_handle(path: Path) -> pysam.AlignmentFile:
    mode = {".bam": "rb", ".sam": "r", ".cram": "rc"}.get(Path(path).suffix)
    if mode is None:
        raise ValueError(f"Unsupported alignment extension: {path}")
    return pysam.AlignmentFile(str(path), mode)


def read_pair_generator(bam, contig=None, start=None, stop=None):
    """Yield proper read pairs."""
    read_dict = defaultdict(lambda: [None, None])
    try:
        for read in bam.fetch(contig=contig, start=start, stop=stop):
            if not read.is_proper_pair or read.is_secondary or read.is_supplementary:
                continue
            qname = read.query_name
            if qname not in read_dict:
                read_dict[qname][0 if read.is_read1 else 1] = read
            else:
                if read.is_read1:
                    yield read, read_dict[qname][1]
                else:
                    yield read_dict[qname][0], read
                del read_dict[qname]
    except (OSError, ValueError) as exc:
        logger.debug(f"fetch {contig}:{start}-{stop}: {exc}")


def parse_genomic_region(region: str) -> dict:
    m = re.match(r"(?P<query>chr\w+):(?P<start>\d+)-(?P<end>\d+)\((?P<strand>[+-])\)", region)
    if not m:
        raise ValueError(f"bad region string: {region}")
    return m.groupdict()


# ---------------------------------------------------------------------------
# Albert per-read class scores
# ---------------------------------------------------------------------------
def average_scores(processed_dataset, class_scores) -> pd.DataFrame:
    """Average Albert score per (read id, class) over the read's fragments."""
    by = defaultdict(lambda: defaultdict(list))
    for data, scores in zip(processed_dataset, class_scores):
        rid = data["id"]
        for sd in scores:
            by[rid][sd["label"]].append(sd["score"])
    rows = {rid: {lab: float(np.mean(v)) for lab, v in labs.items()} for rid, labs in by.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def one_hot(avg: pd.DataFrame) -> pd.DataFrame:
    """One-hot the argmax class per read."""
    return avg.eq(avg.max(axis=1), axis=0).astype(float)


def load_scored_perid(scored_ds: Path) -> pd.DataFrame:
    """Per-id Albert scores from the saved HF 'scored' dataset."""
    df = load_from_disk(str(scored_ds)).to_pandas().set_index("id")
    return df


# ---------------------------------------------------------------------------
# RepeatMasker-overlap fraction (BWA / bowtie2 / STAR)
# ---------------------------------------------------------------------------
def rm_fraction(
    rmtarget: dict,
    bam,
    fetch_contig,
    classes: list,
    paired: bool,
    unique_only: bool,
    divide_classes: list,
) -> pd.DataFrame:
    """Per-read subfamily membership fraction from reads overlapping RM L1 loci.

    ``classes`` are the emitted columns; a read's overlaps are bucketed L1PA2 / L1HS /
    L1PA (target startswith L1PA) / LINE/L1 (family startswith LINE) / NEGATIVE (else).

    ``divide_classes`` lists the emitted classes normalized by the read's total overlap
    count. It is deliberately allowed to be a strict subset of ``classes``: the SRR3997504
    aggregation emits ``L1PA`` but omits it from ``divide_classes``, so that column carries
    raw overlap counts while the others carry per-read fractions. Published figures depend
    on this scaling, so the two are not interchangeable. The SRR4099955 aggregation passes
    ``divide_classes=classes`` and is unaffected.

    Buckets outside ``classes`` are dropped rather than redistributed, so an emitted row
    sums to at most one.
    """
    annotations: dict = defaultdict(list)
    for loci in rmtarget.values():
        for a in loci:
            contig = fetch_contig(a["query_name"])
            start, stop = int(a["query_start"]), int(a["query_end"])
            if paired:
                for read_1, _read_2 in read_pair_generator(
                    bam, contig=contig, start=start, stop=stop
                ):
                    annotations[read_1.query_name].append(a)
            else:
                for read in bam.fetch(contig=contig, start=start, stop=stop):
                    annotations[read.query_name].append(a)

    rows: dict = {}
    for rid, targets in annotations.items():
        if unique_only and len(targets) != 1:
            continue
        score = {c: 0.0 for c in classes}
        for t in targets:
            if t["target"] == "L1PA2":
                score["L1PA2"] = score.get("L1PA2", 0.0) + 1.0
            elif t["target"] == "L1HS":
                score["L1HS"] = score.get("L1HS", 0.0) + 1.0
            elif t["target"].startswith("L1PA"):
                score["L1PA"] = score.get("L1PA", 0.0) + 1.0
            elif t["family"].startswith("LINE"):
                score["LINE/L1"] = score.get("LINE/L1", 0.0) + 1.0
            else:
                score["NEGATIVE"] = score.get("NEGATIVE", 0.0) + 1.0
        for c in divide_classes:
            score[c] = score[c] / len(targets)
        rows[rid] = {c: score.get(c, 0.0) for c in classes}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=classes)


# ---------------------------------------------------------------------------
# L1EM fractions
# ---------------------------------------------------------------------------
def load_l1em_bed(bed: Path) -> pd.DataFrame:
    df = pd.read_csv(bed, sep="\t", header=None)
    df.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "query_strand"]
    df[["target", "category", "region", "target_strand"]] = df["name"].str.split(".", expand=True)
    return df


def l1em_star_fraction(l1em_df: pd.DataFrame, filtered_bam, classes: list) -> pd.DataFrame:
    """Per-read subfamily fraction from reads under each L1EM.400 interval."""
    ann: dict = defaultdict(list)
    for _, row in l1em_df.iterrows():
        for read_1, _r2 in read_pair_generator(
            filtered_bam,
            contig=row["chrom"],
            start=int(row["chromStart"]),
            stop=int(row["chromEnd"]),
        ):
            ann[read_1.query_name].append({"target": row["target"], "family": "LINE/L1"})
    rows: dict = {}
    for rid, targets in ann.items():
        if len(targets) != 1:
            continue
        score = {c: 0.0 for c in classes}
        for t in targets:
            if t["target"] == "L1PA2":
                score["L1PA2"] += 1.0
            elif t["target"] == "L1HS":
                score["L1HS"] += 1.0
            elif t["target"].startswith("L1PA"):
                score["L1PA"] += 1.0
            elif t["family"].startswith("LINE"):
                score["LINE/L1"] += 1.0
        for c in ("L1PA2", "L1HS", "LINE/L1"):
            score[c] = score[c] / len(targets)
        rows[rid] = {c: score.get(c, 0.0) for c in classes}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=classes)


def l1em_bwa_fraction(l1em_bwa_bam, classes: list) -> pd.DataFrame:
    """Per-read subfamily fraction + QUERY chrom from the L1EM BWA re-alignment.

    The reference name encodes ``<target.category.region.strand>::chrN:start-end(strand)``;
    only category==1 (active) hits contribute. Returns ``classes`` + a QUERY column.
    """
    frac: dict = defaultdict(list)
    for read_1, _r2 in read_pair_generator(l1em_bwa_bam):
        target_info, query_info = read_1.reference_name.split("::")
        target, category, _region, _strand = target_info.split(".")
        q = parse_genomic_region(query_info)
        frac[read_1.query_name].append(
            {
                "query_name": q["query"],
                "target": target,
                "category": int(category),
                "family": "LINE/L1",
            }
        )
    rows: dict = {}
    for rid, targets in frac.items():
        if len(targets) != 1:
            continue
        score = {c: 0.0 for c in classes}
        for t in targets:
            score["QUERY"] = t["query_name"]
            if t["category"] != 1:
                continue
            if t["target"] == "L1PA2":
                score["L1PA2"] += 1.0
            elif t["target"] == "L1HS":
                score["L1HS"] += 1.0
            elif t["target"].startswith("L1PA"):
                score["L1PA"] += 1.0
            elif t["family"].startswith("LINE"):
                score["LINE/L1"] += 1.0
        for c in ("L1PA2", "L1HS", "LINE/L1"):
            if c in score and isinstance(score[c], float):
                score[c] = score[c] / len(targets)
        rows[rid] = {**{c: score.get(c, 0.0) for c in classes}, "QUERY": score.get("QUERY")}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=classes + ["QUERY"])


# ---------------------------------------------------------------------------
# Per-chromosome aggregation + tidy melt
# ---------------------------------------------------------------------------
def aggregate_by_chrom(
    bam, scores_annotations: pd.DataFrame, chrom_of, paired: bool
) -> pd.DataFrame:
    """Sum each read's (method, class) vector into the chromosome it aligns to.

    Iterate the chromosome-source BAM, map
    each read to a chromosome via ``chrom_of(reference_name)`` (None -> skip), and add the
    read's row from the MultiIndex ``scores_annotations`` (index=read id).
    """
    cols = list(scores_annotations.columns)
    mat = scores_annotations.to_numpy(dtype=float, na_value=0.0)
    pos = {rid: i for i, rid in enumerate(scores_annotations.index)}
    acc: dict = defaultdict(lambda: np.zeros(len(cols)))

    def add(qname, ref):
        chrom = chrom_of(ref)
        if chrom is None:
            return
        i = pos.get(qname)
        if i is not None:
            acc[chrom] += mat[i]

    if paired:
        for read_1, _r2 in read_pair_generator(bam):
            add(read_1.query_name, read_1.reference_name)
    else:
        for read in bam.fetch(until_eof=True):
            add(read.query_name, read.reference_name)

    out = pd.DataFrame.from_dict(
        {c: v for c, v in acc.items()}, orient="index", columns=pd.MultiIndex.from_tuples(cols)
    )
    return out


def melt_aggregated(agg: pd.DataFrame, sample: str, figure: str) -> pd.DataFrame:
    """MultiIndex (method, class) columns x chrom index -> long rows."""
    recs = []
    for chrom, row in agg.iterrows():
        for (method, klass), value in row.items():
            recs.append((sample, figure, str(chrom), method, klass, float(value)))
    df = pd.DataFrame(recs, columns=["sample", "figure", "chrom", "method", "class", "value"])
    return df


def concat_pieces(pieces: dict, fill_na: bool, join: str) -> pd.DataFrame:
    """Legacy ``pd.concat(pieces, axis=1[, join])`` + optional 0-fill of the L1 class cols."""
    sa = pd.concat(pieces, axis=1, join=join)
    if fill_na:
        for method in pieces:
            klasses = [c for c in pieces[method].columns if c != "QUERY"]
            na = {k: (1.0 if k == "NEGATIVE" else 0.0) for k in klasses}
            sa[method] = sa[method].fillna(value=na)
    return sa


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def trap_version() -> str:
    try:
        import trap

        return getattr(trap, "__version__", "unknown")
    except Exception:
        return "unknown"


def write_outputs(rows: pd.DataFrame, out_dir: Path, inputs: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows["_k"] = rows["chrom"].map(chr_sort_key)
    rows = rows.sort_values(["sample", "figure", "_k", "chrom", "method", "class"]).drop(
        columns="_k"
    )
    csv_path = out_dir / "aggregate_by_chr.csv"
    rows.to_csv(csv_path, index=False)
    manifest = {
        "trap_version": trap_version(),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 3469,
        "inputs": {k: str(v) for k, v in inputs.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.success(f"wrote {len(rows):,} rows -> {csv_path}  (+ manifest.json)")


# ===========================================================================
# SRR4099955  ->  notebook 5.04
# ===========================================================================
@app.command()
def srr4099955(
    bwa_bam: Path = typer.Option("results/bwa/SRR4099955/sorted_alignments.bam"),
    short_reads: Path = typer.Option("run/outputs/quantify_k18_SRR4099955/short_reads"),
    scores_12m: Path = typer.Option("run/outputs/quantify_k18_SRR4099955/class_scores.pkl"),
    scores_4m: Path = typer.Option(
        "run/outputs/distilled4M_quantify_k18_SRR4099955/class_scores.pkl"
    ),
    scores_1m: Path = typer.Option(
        "run/outputs/distilled1M_quantify_k18_SRR4099955/class_scores.pkl"
    ),
    rm_out: Path = typer.Option("data/annotations/GCF_000001405.40_GRCh38.p14_rm.LINE1.out"),
    translator: Path = typer.Option(
        "data/annotations/GCF_000001405.40_GRCh38.p14_assembly_report.trans"
    ),
    out_dir: Path = typer.Option("results/srr4099955"),
):
    """Aggregate SRR4099955 (Albert OHE + BWA RepeatMasker) and the distilled variants."""
    classes = ["L1HS", "L1PA2", "NEGATIVE"]

    trans = pd.read_csv(translator, sep="\t", header=None, names=["NCBI_name", "common_name"])
    ncbi2common = dict(zip(trans["NCBI_name"], trans["common_name"]))

    def chrom_of(ref):  # BWA references are NCBI accessions (NC_...)
        if ref is None or not ref.startswith("NC"):
            return None
        return ncbi2common.get(ref)

    logger.info("loading Albert per-read scores (12M / 4M / 1M)")
    ds = load_from_disk(str(short_reads))
    with open(scores_12m, "rb") as f:
        cs12 = pickle.load(f)
    avg12 = average_scores(ds, cs12)[classes]
    ohe12 = one_hot(avg12)[classes]

    rm = load_rm_line1(rm_out)
    bam = alignments_handle(bwa_bam)
    logger.info("BWA RepeatMasker-overlap fractions (single reads)")
    bwa_frac = rm_fraction(
        rm,
        bam,
        fetch_contig=lambda q: q,
        classes=classes,
        paired=False,
        unique_only=False,
        divide_classes=classes,
    )

    # Figure 1: Albert (OHE) vs Alignment (BWA fraction, NEGATIVE-default fill).
    sa = concat_pieces({"Albert": ohe12, "Alignment": bwa_frac}, fill_na=True, join="outer")
    agg = aggregate_by_chrom(alignments_handle(bwa_bam), sa, chrom_of, paired=False)
    rows = [melt_aggregated(agg, "SRR4099955", "SRR4099955_agreggate_mlvs_bwa_by_chr")]

    # Figure 2: distilled Albert 12M / 4M / 1M (averaged scores).
    with open(scores_4m, "rb") as f:
        avg4 = average_scores(ds, pickle.load(f))[classes]
    with open(scores_1m, "rb") as f:
        avg1 = average_scores(ds, pickle.load(f))[classes]
    sa_d = concat_pieces(
        {"Albert_12M": avg12, "Albert_4M": avg4, "Albert_1M": avg1}, fill_na=False, join="outer"
    )
    agg_d = aggregate_by_chrom(alignments_handle(bwa_bam), sa_d, chrom_of, paired=False)
    rows.append(melt_aggregated(agg_d, "SRR4099955", "SRR4099955_agreggate_distilled_by_chr"))

    write_outputs(
        pd.concat(rows, ignore_index=True),
        out_dir,
        {
            "bwa_bam": bwa_bam,
            "short_reads": short_reads,
            "scores_12m": scores_12m,
            "scores_4m": scores_4m,
            "scores_1m": scores_1m,
            "rm_out": rm_out,
            "translator": translator,
        },
    )


# ===========================================================================
# SRR3997504  ->  notebook 5.05
# ===========================================================================
@app.command()
def srr3997504(
    scored_ds: Path = typer.Option("run/outputs/quantify_k18_SRR3997504/scored"),
    bowtie2_bam: Path = typer.Option(
        "run/outputs/align_quant/bowtie2/SRR3997504/sorted_reads.bam"
    ),
    star_filtered_bam: Path = typer.Option(
        "run/outputs/align_quant/star/SRR3997504_filter/sorted.bam"
    ),
    star_whole_bam: Path = typer.Option("run/outputs/align_quant/star/SRR3997504/sorted.bam"),
    l1em_bwa_bam: Path = typer.Option(
        "run/outputs/align_quant/bwa/SRR3997504_L1EM_filter/sorted_reads.bam"
    ),
    rm_out: Path = typer.Option("data/annotations/GCF_000001405.40_GRCh38.p14_rm.LINE1.out"),
    l1em_bed: Path = typer.Option("data/annotations/L1EM/L1EM.400.bed"),
    translator: Path = typer.Option(
        "data/annotations/GCF_000001405.40_GRCh38.p14_assembly_report.txt"
    ),
    tau: float = typer.Option(0.3, help="keep reads with P(NEGATIVE) < tau for the 'filtered' ML"),
    out_dir: Path = typer.Option("results/srr3997504"),
):
    """Aggregate SRR3997504 (ML vs bowtie2 / STAR / L1EM), the four by-chr figures."""
    star_classes = ["L1HS", "L1PA2", "L1PA", "LINE/L1"]

    rep = pd.read_csv(
        translator,
        sep="\t",
        comment="#",
        header=None,
        usecols=[4, 6, 9],
        names=["GenBank_name", "RefSeq_name", "UCSC_name"],
    )
    refseq2ucsc = dict(zip(rep["RefSeq_name"], rep["UCSC_name"]))

    def fetch_contig(refseq):  # RM query_name is a RefSeq accession -> UCSC 'chrN'
        return refseq2ucsc.get(refseq, "Unknown Chromosome")

    def chrom_of(ref):  # bowtie2/STAR references are already 'chrN'
        return ref if (ref is not None and ref.startswith("chr")) else None

    logger.info("loading Albert per-id scores")
    scores = load_scored_perid(scored_ds)
    ml = scores[["L1HS", "L1PA2"]]
    ml_filtered = scores[scores["NEGATIVE"] < tau][["L1HS", "L1PA2"]]

    rm = load_rm_line1(rm_out)
    bowtie2 = alignments_handle(bowtie2_bam)

    rows = []

    # Figure 1: ML vs Bowtie2 (unfiltered, RM fraction on the bowtie2 BAM).
    logger.info("[fig1] bowtie2 RepeatMasker fractions")
    bowtie_frac = rm_fraction(
        rm,
        bowtie2,
        fetch_contig,
        classes=star_classes,
        paired=True,
        unique_only=True,
        divide_classes=["L1PA2", "L1HS", "LINE/L1"],
    )
    sa1 = concat_pieces({"ML": ml, "Bowtie2": bowtie_frac}, fill_na=True, join="outer")
    agg1 = aggregate_by_chrom(alignments_handle(bowtie2_bam), sa1, chrom_of, paired=True)
    rows.append(melt_aggregated(agg1, "SRR3997504", "SRR3997504_agreggate_mlvs_bowtie_by_chr"))

    # STAR filtered / whole RM fractions (shared by figs 2 & 3).
    logger.info("[fig2/3] STAR filtered/whole RepeatMasker fractions")
    star_filtered = alignments_handle(star_filtered_bam)
    star_whole = alignments_handle(star_whole_bam)
    filt_frac = rm_fraction(
        rm, star_filtered, fetch_contig, star_classes, True, True, ["L1PA2", "L1HS", "LINE/L1"]
    )
    whole_frac = rm_fraction(
        rm, star_whole, fetch_contig, star_classes, True, True, ["L1PA2", "L1HS", "LINE/L1"]
    )

    # Figure 2: ML(filtered) vs FILTERED_STAR vs WHOLE_STAR (outer join, fill).
    sa2 = concat_pieces(
        {
            "ML": ml_filtered,
            "FILTERED_STAR": filt_frac[star_classes],
            "WHOLE_STAR": whole_frac[star_classes],
        },
        fill_na=True,
        join="outer",
    )
    agg2 = aggregate_by_chrom(alignments_handle(bowtie2_bam), sa2, chrom_of, paired=True)
    rows.append(
        melt_aggregated(agg2, "SRR3997504", "SRR3997504_filtered_agreggate_mlvs_star_by_chr")
    )

    # Figure 3: + L1EM_STAR (inner join across all four pieces).
    logger.info("[fig3] L1EM.400 interval fractions on filtered STAR")
    l1em_df = load_l1em_bed(l1em_bed)
    l1em_star = l1em_star_fraction(l1em_df, alignments_handle(star_filtered_bam), star_classes)
    sa3 = concat_pieces(
        {
            "ML": ml_filtered,
            "FILTERED_STAR": filt_frac[star_classes],
            "WHOLE_STAR": whole_frac[star_classes],
            "L1EM_STAR": l1em_star[star_classes],
        },
        fill_na=False,
        join="inner",
    )
    # ML/FILTERED_STAR fill applies before the inner join; approximated here by 0-fill.
    for m in ("ML", "FILTERED_STAR"):
        sa3[m] = sa3[m].fillna(0.0)
    agg3 = aggregate_by_chrom(alignments_handle(bowtie2_bam), sa3, chrom_of, paired=True)
    rows.append(
        melt_aggregated(agg3, "SRR3997504", "SRR3997504_filtered_agreggate_l1em_mlvs_star_by_chr")
    )

    # Figure 4: ML(filtered) vs L1EM_BWA, aggregated by the L1EM QUERY chromosome.
    logger.info("[fig4] L1EM BWA re-alignment fractions")
    l1em_classes = ["L1HS", "L1PA2", "L1PA"]
    l1em_bwa = l1em_bwa_fraction(alignments_handle(l1em_bwa_bam), l1em_classes + ["LINE/L1"])
    sa4 = pd.concat({"ML": ml_filtered, "L1EM_BWA": l1em_bwa}, axis=1, join="inner")
    grouped = sa4.groupby(("L1EM_BWA", "QUERY")).sum()
    grouped.index.name = "chrom"
    grouped = grouped.drop(columns=[("L1EM_BWA", "QUERY")], errors="ignore")
    # Keep only the plotted (method, class) series.
    keep = [
        ("ML", "L1HS"),
        ("L1EM_BWA", "L1HS"),
        ("ML", "L1PA2"),
        ("L1EM_BWA", "L1PA2"),
        ("L1EM_BWA", "L1PA"),
    ]
    grouped = grouped.reindex(columns=pd.MultiIndex.from_tuples(keep), fill_value=0.0)
    rows.append(
        melt_aggregated(grouped, "SRR3997504", "SRR3997504_filtered_agreggate_l1em_bwa_by_chr")
    )

    write_outputs(
        pd.concat(rows, ignore_index=True),
        out_dir,
        {
            "scored_ds": scored_ds,
            "bowtie2_bam": bowtie2_bam,
            "star_filtered_bam": star_filtered_bam,
            "star_whole_bam": star_whole_bam,
            "l1em_bwa_bam": l1em_bwa_bam,
            "rm_out": rm_out,
            "l1em_bed": l1em_bed,
            "translator": translator,
            "tau": tau,
        },
    )


if __name__ == "__main__":
    app()
