"""Track B1 — relabel the corpus by L1 *sequence* content (minimap2 → L1 library).

The coordinate labels (`pairtobed -type either` over transcriptome reads) tag a
read "L1" by genomic overlap, not L1 sequence — ~95% of "L1" reads carry no L1
sequence (see `.trap/plans/l1-sequence-anchored-labels.md` §0). This redefines the
positive class by alignment to the L1 subfamily consensus library:

  read is L1 ⇔ best minimap2 alignment has identity ≥ id_thr over ≥ cov_thr of the
  read; its subfamily = the best-aligning consensus (→ L1HS / L1PA / OTHER via
  ``task_label``); no qualifying alignment ⇒ NEGATIVE.

``calibrate`` (Track B0): sweep thresholds on a read sample and report
alignment rate + old-vs-new label agreement, to pick id_thr / cov_thr.
``run`` (Track B1): relabel the full corpus and emit the read→label table + the
old-vs-new confusion (the rigorous §4.4 label-noise quantification).

Requires minimap2 (Grace: ``module load GCCcore/13.2.0 minimap2/2.29``).
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import typer

from trap.analysis._l1_align import L1Alignment, align_sequences, minimap2_available
from trap.config.config import RESULTS_DIR
from trap.utils.io import genome_file_handle
from trap.utils.labels import task_label

app = typer.Typer(add_completion=False, help="Relabel reads by L1 sequence (Track B1).")


@app.callback()
def _main() -> None:
    """Keep subcommand names (Typer collapses single-command apps)."""


# --------------------------------------------------------------------------- #
# Pure core (CPU, testable)                                                    #
# --------------------------------------------------------------------------- #
def subfamily_from_header(target_name: str) -> str:
    """Subfamily token from a consensus FASTA id (``L1PA3#LINE/L1`` → ``L1PA3``)."""
    return target_name.split("#")[0].split()[0].strip() if target_name else ""


def sequence_label(
    aln: Optional[L1Alignment], identity_thr: float, coverage_thr: float
) -> Tuple[str, str]:
    """``(task_class, subfamily)`` for a read from its best L1 alignment.

    Below threshold or unaligned ⇒ ``("NEGATIVE", "NEGATIVE")``. Otherwise the
    subfamily is the aligned consensus and the class is its ``task_label`` collapse
    (L1HS / L1PA / OTHER — OTHER = ancient L1M*, dropped from the 3-class).
    """
    if aln is None or aln.identity < identity_thr or aln.aligned_frac < coverage_thr:
        return "NEGATIVE", "NEGATIVE"
    sub = subfamily_from_header(aln.target_name)
    return task_label(f"x|{sub}"), sub


def _best(a: Optional[L1Alignment], b: Optional[L1Alignment]) -> Optional[L1Alignment]:
    """Higher identity×coverage of two (mate) alignments."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a.identity * a.aligned_frac >= b.identity * b.aligned_frac else b


# --------------------------------------------------------------------------- #
# Shared IO / alignment                                                        #
# --------------------------------------------------------------------------- #
def _read_pairs(r1: str, r2: str, max_reads: int):
    """Yield ``(read_id, s1, s2, old_label)`` from paired FASTQ (4-line records)."""
    n = 0
    with genome_file_handle(r1) as h1, genome_file_handle(r2) as h2:
        while max_reads <= 0 or n < max_reads:
            i1 = h1.readline()
            s1 = h1.readline()
            h1.readline()
            h1.readline()
            h2.readline()
            s2 = h2.readline()
            h2.readline()
            h2.readline()
            if not i1 or not s1 or not s2:
                break
            rid = i1[1:].split()[0]
            yield rid, s1.strip(), s2.strip(), task_label(rid)
            n += 1


def _relabel(reads, library: str, id_thr: float, cov_thr: float, threads: int):
    """Align both mates of each fragment to the library; return per-read new labels.

    Returns lists ``(old_labels, new_labels, subfamilies, best_alns)`` aligned to
    the input order.
    """
    s1 = [r[1] for r in reads]
    s2 = [r[2] for r in reads]
    a1 = align_sequences(s1, library, threads=threads)
    a2 = align_sequences(s2, library, threads=threads)
    best = [_best(x, y) for x, y in zip(a1, a2)]
    new_labels, subs = [], []
    for aln in best:
        lab, sub = sequence_label(aln, id_thr, cov_thr)
        new_labels.append(lab)
        subs.append(sub)
    return [r[3] for r in reads], new_labels, subs, best


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
@app.command()
def calibrate(
    library: Path = typer.Option(..., help="L1 subfamily consensus FASTA (Track B0)."),
    r1: Path = typer.Option(..., help="Sample R1 FASTQ."),
    r2: Path = typer.Option(..., help="Sample R2 FASTQ."),
    max_reads: int = typer.Option(20000, help="Reads to sample for calibration."),
    threads: int = typer.Option(8),
):
    """Sweep id/coverage thresholds; report alignment rate + old-vs-new agreement."""
    if not minimap2_available():
        raise typer.BadParameter("minimap2 not found (module load GCCcore/13.2.0 minimap2/2.29).")
    reads = list(_read_pairs(str(r1), str(r2), max_reads))
    logger.info(f"calibrating on {len(reads):,} read pairs vs {library}")
    s1, s2 = [r[1] for r in reads], [r[2] for r in reads]
    a1 = align_sequences(s1, str(library), threads=threads)
    a2 = align_sequences(s2, str(library), threads=threads)
    best = [_best(x, y) for x, y in zip(a1, a2)]
    old = [r[3] for r in reads]
    old_is_l1 = [o in ("L1HS", "L1PA") for o in old]

    lines = ["\nThreshold sweep (id × cov): new-L1 rate, agreement with old L1-vs-NEG:"]
    lines.append(f"  {'id':>4} {'cov':>4}  {'new_L1%':>8} {'old∧new_L1%':>12} {'aligned%':>9}")
    for id_thr in (0.70, 0.80, 0.90):
        for cov_thr in (0.40, 0.60):
            new = [sequence_label(b, id_thr, cov_thr)[0] for b in best]
            new_is_l1 = [n in ("L1HS", "L1PA") for n in new]
            n = len(reads)
            new_l1 = sum(new_is_l1) / n
            agree = sum(o and x for o, x in zip(old_is_l1, new_is_l1)) / max(1, sum(old_is_l1))
            aligned = sum(b is not None for b in best) / n
            lines.append(
                f"  {id_thr:>4.2f} {cov_thr:>4.2f}  {100 * new_l1:>7.2f}% {100 * agree:>11.1f}% {100 * aligned:>8.1f}%"
            )
    logger.info("\n".join(lines))
    logger.success("Pick id/cov where new-L1 reads are real L1 (sanity: they align ~100%).")


@app.command()
def run(
    library: Path = typer.Option(..., help="L1 subfamily consensus FASTA (Track B0)."),
    r1: Path = typer.Option(..., help="Corpus R1 FASTQ."),
    r2: Path = typer.Option(..., help="Corpus R2 FASTQ."),
    identity_threshold: float = typer.Option(0.80),
    coverage_threshold: float = typer.Option(0.50),
    max_reads: int = typer.Option(0, help="0 = whole corpus."),
    threads: int = typer.Option(8),
    out: Path = typer.Option(RESULTS_DIR / "relabel_by_sequence.tsv"),
):
    """Relabel the corpus by sequence; emit read→label table + old-vs-new confusion."""
    if not minimap2_available():
        raise typer.BadParameter("minimap2 not found (module load GCCcore/13.2.0 minimap2/2.29).")
    reads = list(_read_pairs(str(r1), str(r2), max_reads))
    old, new, subs, best = _relabel(
        reads, str(library), identity_threshold, coverage_threshold, threads
    )

    confusion: Counter = Counter(zip(old, new))
    lines = ["\nOld (coordinate) → New (sequence) label confusion:"]
    classes = ["L1HS", "L1PA", "NEGATIVE", "OTHER"]
    for o in classes:
        row = {c: confusion.get((o, c), 0) for c in classes}
        tot = sum(row.values())
        if tot:
            lines.append(
                f"  {o:<9} → " + "  ".join(f"{c}:{row[c]}" for c in classes) + f"   (n={tot})"
            )
    logger.info("\n".join(lines))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["read_id", "old_label", "new_label", "subfamily", "identity", "aligned_frac"])
        for r, o, nlab, sub, b in zip(reads, old, new, subs, best):
            w.writerow(
                [
                    r[0],
                    o,
                    nlab,
                    sub,
                    f"{b.identity:.4f}" if b else "",
                    f"{b.aligned_frac:.4f}" if b else "",
                ]
            )
    logger.success(f"Wrote {len(reads):,} relabelled reads -> {out}")


if __name__ == "__main__":
    app()
