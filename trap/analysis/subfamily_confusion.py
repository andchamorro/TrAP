"""Per-raw-subfamily prediction breakdown for the L1 classifier.

Diagnoses the dominant error axis of the 3-class task (L1PA <-> NEGATIVE). The
collapsed label hides *which* RepeatMasker subfamily each read came from; this
runs the trained classifier over reads of KNOWN raw subfamily (parsed from the
FASTQ id) and cross-tabulates the predicted task class against that raw subfamily.

It answers the two questions the confusion matrix can't:

  * Of the L1PA reads called NEGATIVE, are they concentrated in the OLD edge of
    the L1PA bin (L1PREC2 / L1PB / L1PA13-17)? -> the bin is too broad, and the
    model is biologically right to call its degenerate members background.
  * Where do the DROPPED ancient L1s (L1M*, L1ME*, OTHER) land — NEGATIVE (clean)
    or L1PA (they leak L1 signal into the positive class)?

Tokenisation mirrors trap.utils.preprocessing_sequences.classification exactly,
so it is valid for BOTH the SPM (raw-read) and Salmon (k-mer) models.

Usage (1 GPU):
    TRAP_SPM_EXPERIMENTAL=1 python -m trap.analysis.subfamily_confusion \
        --model-path models/albert.l1hs_l1pa2.v48.k17.spm/final \
        --tokenizer-path models/tokenizer.gencode.v48.k17.spm \
        --r1 data/external/l1hs_l1pa2_negative.5x_R1.fq \
        --r2 data/external/l1hs_l1pa2_negative.5x_R2.fq \
        --per-subfamily 4000 --max-position-embeddings 128 \
        --out results/subfamily_confusion.spm.csv
"""

from collections import defaultdict
import csv
from pathlib import Path
import re

from Bio import SeqIO
from loguru import logger
import typer

from trap.config.config import RESULTS_DIR
from trap.utils.io import genome_file_handle
from trap.utils.kmer import kmer_split
from trap.utils.labels import task_label

app = typer.Typer(add_completion=False)


def _raw_subfamily(read_id: str) -> str:
    """Parse the RepeatMasker subfamily from a TrAP record id (last ``|`` field)."""
    parts = read_id.split("|")
    raw = parts[-1].split("-")[0] if len(parts) >= 2 else read_id
    return raw.upper()


def _age_rank(sf: str):
    """Coarse evolutionary-age sort key (young -> old -> background) for display."""
    if sf == "L1HS":
        return (0, 0, sf)
    if sf.startswith("L1PA"):
        nums = re.findall(r"\d+", sf)
        return (1, int(nums[0]) if nums else 99, sf)
    if sf.startswith("L1P"):  # L1PB, L1PREC2, L1P1-5 — old primate edge of the bin
        return (2, 0, sf)
    if sf == "NEGATIVE":
        return (9, 0, sf)
    return (5, 0, sf)  # L1M*, L1ME*, … (ancient; OTHER/dropped)


def _read_pairs(r1: str, r2: str, per_subfamily: int, max_scan: int):
    """Scan paired FASTQ in lockstep; keep up to ``per_subfamily`` reads/subfamily."""
    kept = defaultdict(list)
    counts = defaultdict(int)
    n = 0
    with genome_file_handle(r1) as h1, genome_file_handle(r2) as h2:
        for rec1, rec2 in zip(SeqIO.parse(h1, "fastq"), SeqIO.parse(h2, "fastq")):
            n += 1
            if n > max_scan:
                break
            sf = _raw_subfamily(str(rec1.id))
            if counts[sf] >= per_subfamily:
                continue
            counts[sf] += 1
            kept[sf].append((str(rec1.seq), str(rec2.seq)))
    return kept


@app.command()
def run(
    model_path: Path = typer.Option(..., help="Trained classifier dir (…/final)"),
    tokenizer_path: Path = typer.Option(..., help="Tokenizer dir (SPM or Salmon)"),
    r1: Path = typer.Option(..., help="L1 R1 FASTQ (raw subfamily in the read ids)"),
    r2: Path = typer.Option(..., help="L1 R2 FASTQ"),
    k: int = typer.Option(17, help="K-mer size (k-mer tokenizers only)."),
    max_position_embeddings: int = typer.Option(128, help="Pad/truncate length."),
    per_subfamily: int = typer.Option(4000, help="Max reads sampled per subfamily."),
    max_scan: int = typer.Option(4_000_000, help="Cap on FASTQ pairs scanned."),
    batch_size: int = typer.Option(256),
    out: Path = typer.Option(RESULTS_DIR / "subfamily_confusion.csv"),
):
    import torch
    from transformers import AlbertForSequenceClassification

    from trap.loaders.tokenizer import SalmonKmerTokenizer, load_kmer_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_kmer_tokenizer(str(tokenizer_path), max_position_embeddings)
    is_salmon = isinstance(tokenizer, SalmonKmerTokenizer)
    is_raw = getattr(tokenizer, "trap_raw_read", False)
    model = AlbertForSequenceClassification.from_pretrained(str(model_path)).to(device).eval()
    id2label = {i: model.config.id2label[i] for i in range(model.config.num_labels)}
    classes = [id2label[i] for i in range(len(id2label))]
    logger.info(
        f"device={device} model={model_path} classes={classes} "
        f"salmon={is_salmon} raw_read={is_raw} k={k} pad={max_position_embeddings}"
    )

    kept = _read_pairs(str(r1), str(r2), per_subfamily, max_scan)
    logger.info(
        f"sampled {sum(len(v) for v in kept.values()):,} read pairs over "
        f"{len(kept)} raw subfamilies"
    )

    def _encode(b1, b2):
        if is_salmon:
            enc = tokenizer.batch_encode_sequences(
                b1,
                b2,
                max_length=max_position_embeddings,
                padding="max_length",
                pad_to_multiple_of=8,
                truncation=True,
            )
        else:
            t1 = list(b1) if is_raw else [kmer_split(k, r) for r in b1]
            t2 = list(b2) if is_raw else [kmer_split(k, r) for r in b2]
            enc = tokenizer(
                t1,
                t2,
                padding="max_length",
                pad_to_multiple_of=8,
                truncation=True,
                max_length=max_position_embeddings,
                return_token_type_ids=True,
            )
        return {
            kk: torch.as_tensor(enc[kk]).to(device)
            for kk in ("input_ids", "attention_mask", "token_type_ids")
            if kk in enc
        }

    # counts[subfamily][predicted_class] and summed class-probabilities per subfamily.
    counts = defaultdict(lambda: defaultdict(int))
    prob_sum = defaultdict(lambda: [0.0] * len(classes))
    for sf, pairs in kept.items():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            inp = _encode([a for a, _ in chunk], [b for _, b in chunk])
            with torch.no_grad():
                probs = model(**inp).logits.softmax(-1)
            for row in probs:
                counts[sf][classes[int(row.argmax())]] += 1
                ps = prob_sum[sf]
                for j in range(len(classes)):
                    ps[j] += float(row[j])

    # --- per-subfamily table (CSV + log) -------------------------------------
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["subfamily", "collapsed", "n"]
    header += [f"pct_{c}" for c in classes] + [f"meanP_{c}" for c in classes]
    rows = []
    for sf in sorted(kept, key=_age_rank):
        n = sum(counts[sf].values())
        if not n:
            continue
        pct = [100 * counts[sf].get(c, 0) / n for c in classes]
        meanp = [prob_sum[sf][j] / n for j in range(len(classes))]
        rows.append(
            [
                sf,
                task_label(f"x|{sf}"),
                n,
                *[round(x, 1) for x in pct],
                *[round(x, 3) for x in meanp],
            ]
        )
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    pidx = classes.index("NEGATIVE") if "NEGATIVE" in classes else -1
    lines = [
        "\nPer-subfamily prediction breakdown (young -> old -> NEGATIVE):",
        f"  {'subfamily':<12} {'coll':<9} {'n':>7}  "
        + "  ".join(f"%->{c:<8}" for c in classes)
        + "  meanP(NEG)",
    ]
    for sf in sorted(kept, key=_age_rank):
        n = sum(counts[sf].values())
        if not n:
            continue
        pct = "  ".join(f"{100 * counts[sf].get(c, 0) / n:>8.1f}" for c in classes)
        pneg = prob_sum[sf][pidx] / n if pidx >= 0 else float("nan")
        lines.append(f"  {sf:<12} {task_label(f'x|{sf}'):<9} {n:>7}  {pct}  {pneg:>9.3f}")
    logger.info("\n".join(lines))
    logger.success(f"Wrote per-subfamily breakdown -> {out}")


if __name__ == "__main__":
    app()
