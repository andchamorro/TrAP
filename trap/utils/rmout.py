"""RepeatMasker ``.out`` → L1 BED, and strict-overlap fragment labelling (Track B2).

The sequence-anchored build (`build_dataset_seqlabel.sh`) replaces the noisy
``pairtobed -type either`` (any overlap, per pair) labelling with **strict
read-fraction overlap** against the LINE-1 RepeatMasker annotation:

  * ``to-bed`` — parse the ``_rm.LINE1.out`` into a BED of L1 instances
    (chrom, start, end, family, %div, strand), optionally divergence-filtered.
  * ``label-fragments`` — given bedtools' strict (``-f`` ≥ fraction) per-read hits
    and the any-overlap read set, assign each fragment (read pair, by QNAME) a
    label: an L1 task class if either mate is ≥fraction L1 (subfamily by
    ``TASK_PRIORITY``), ``NEGATIVE`` if neither mate overlaps L1 at all, else
    ``DROP`` (partial 0<overlap<fraction — the ambiguous boundary, excluded).

Because STAR aligned reads at high identity, ≥fraction read-overlap with an L1
instance ≈ that fraction of the read IS L1 sequence — so strict overlap is a
sequence-content proxy (cross-checked directly by ``relabel_by_sequence``).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger
import typer

from trap.utils.io import genome_file_handle
from trap.utils.labels import TASK_PRIORITY, task_label

app = typer.Typer(add_completion=False, help="RepeatMasker .out → L1 BED + strict labelling.")


@app.callback()
def _main() -> None:
    """Keep subcommand names (Typer collapses single-command apps)."""


# --------------------------------------------------------------------------- #
# Pure core (CPU, testable)                                                    #
# --------------------------------------------------------------------------- #
def parse_rmout_line(line: str) -> Optional[Tuple[str, int, int, str, float, str]]:
    """Parse one RepeatMasker ``.out`` row → ``(chrom, start, end, family, div, strand)``.

    Returns ``None`` for header / blank / non-data lines. ``.out`` query coords are
    1-based inclusive → converted to BED 0-based half-open. Strand ``C`` → ``-``.
    """
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        div = float(parts[1])  # header rows have a non-numeric col 1
        start = int(parts[5]) - 1
        end = int(parts[6])
    except ValueError:
        return None
    chrom, strand, family = parts[4], ("-" if parts[8] == "C" else "+"), parts[9]
    return chrom, start, end, family, div, strand


def is_l1(family: str, class_family: str = "") -> bool:
    """Whether a RepeatMasker family is LINE-1 (by class ``LINE/L1`` or name ``L1*``)."""
    return class_family.upper().startswith("LINE/L1") or family.upper().startswith("L1")


def best_task_class(families: Iterable[str]) -> str:
    """Highest-priority task class among the L1 families a fragment hit.

    L1HS > L1PA > NEGATIVE (``TASK_PRIORITY``); families that collapse to neither
    (ancient ``L1M*`` → ``OTHER``) yield ``OTHER`` (caller drops from the 3-class).
    """
    classes = {task_label(f"x|{fam}") for fam in families}
    for c in TASK_PRIORITY:
        if c in classes:
            return c
    return "OTHER"


def resolve_fragment_label(strict_families: List[str], any_overlap: bool) -> str:
    """Per-fragment label from its mates' strict hits + any-overlap flag.

    ``strict_families`` = L1 families a mate overlaps ≥fraction (empty if none).
    Returns an L1 task class, ``NEGATIVE`` (no L1 overlap at all), or ``DROP``
    (partial overlap only — the ambiguous boundary, excluded from training).
    """
    if strict_families:
        return best_task_class(strict_families)
    return "NEGATIVE" if not any_overlap else "DROP"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
@app.command("to-bed")
def to_bed(
    rmout: Path = typer.Option(..., help="RepeatMasker .out(.gz) (LINE1-only is fine)."),
    out: Path = typer.Option(..., help="Output BED of L1 instances."),
    max_div: float = typer.Option(
        100.0, help="Keep instances with %div ≤ this (e.g. 10 for young-only L1)."
    ),
):
    """Stream a RepeatMasker .out → BED (chrom, start, end, family, div, strand)."""
    n_in = n_out = 0
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with genome_file_handle(str(rmout)) as fh, out.open("w") as bed:
        for line in fh:
            rec = parse_rmout_line(line)
            if rec is None:
                continue
            n_in += 1
            chrom, start, end, family, div, strand = rec
            if not is_l1(family) or div > max_div:
                continue
            bed.write(f"{chrom}\t{start}\t{end}\t{family}\t{div:.1f}\t{strand}\n")
            n_out += 1
    logger.success(f"wrote {n_out:,} L1 BED intervals (of {n_in:,} parsed) -> {out}")


@app.command("label-fragments")
def label_fragments(
    strict_hits: Path = typer.Option(
        ..., help="TSV 'qname<TAB>family' of reads overlapping L1 ≥fraction (bedtools -f -wb)."
    ),
    negative_qnames: Path = typer.Option(
        ..., help="One QNAME per line of reads with NO L1 overlap (bedtools -v)."
    ),
    out: Path = typer.Option(..., help="Output TSV 'qname<TAB>label' (OTHER/DROP excluded)."),
):
    """Resolve per-fragment labels: strict→L1 class, no-overlap→NEGATIVE, partial→DROP."""
    strict: Dict[str, List[str]] = defaultdict(list)
    with genome_file_handle(str(strict_hits)) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                strict[f[0]].append(f[1])
    with genome_file_handle(str(negative_qnames)) as fh:
        negatives = {line.strip() for line in fh if line.strip()}

    counts: Dict[str, int] = defaultdict(int)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as oh:
        # A read is in strict (≥fraction L1), in negatives (no overlap), or neither
        # (partial overlap → DROP). any_overlap = NOT a no-overlap read.
        for q in set(strict) | negatives:
            label = resolve_fragment_label(strict.get(q, []), any_overlap=q not in negatives)
            counts[label] += 1
            if label in ("L1HS", "L1PA", "NEGATIVE"):
                oh.write(f"{q}\t{label}\n")
    logger.success(f"fragment labels (OTHER/DROP excluded from output): {dict(counts)} -> {out}")


@app.command("emit-labeled")
def emit_labeled(
    r1_in: Path = typer.Option(..., help="samtools-fastq R1 of the whole BAM."),
    r2_in: Path = typer.Option(..., help="samtools-fastq R2 of the whole BAM."),
    labels: Path = typer.Option(..., help="qname<TAB>label TSV (from label-fragments)."),
    r1_out: Path = typer.Option(...),
    r2_out: Path = typer.Option(...),
):
    """Stream paired FASTQ, keep labelled fragments, append ``|LABEL`` to both mates."""
    lab: Dict[str, str] = {}
    with genome_file_handle(str(labels)) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                lab[f[0]] = f[1]
    logger.info(f"emit-labeled: {len(lab):,} labelled fragments")

    kept = 0
    with genome_file_handle(str(r1_in)) as h1, genome_file_handle(str(r2_in)) as h2, Path(
        r1_out
    ).open("w") as o1, Path(r2_out).open("w") as o2:
        while True:
            rec1 = [h1.readline() for _ in range(4)]
            rec2 = [h2.readline() for _ in range(4)]
            if not rec1[0] or not rec2[0]:
                break
            qname = rec1[0][1:].split()[0].rsplit("/", 1)[0]
            label = lab.get(qname)
            if label is None:
                continue
            tag = f"|{label}"
            o1.write(rec1[0].rstrip("\n") + tag + "\n" + "".join(rec1[1:]))
            o2.write(rec2[0].rstrip("\n") + tag + "\n" + "".join(rec2[1:]))
            kept += 1
    logger.success(f"emitted {kept:,} labelled read pairs")


if __name__ == "__main__":
    app()
