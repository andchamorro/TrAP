"""K-mer-level attribution for the L1 classifier + biological validation.

Backs manuscript §2.6 ("Transformers Model Interpret") and answers reviewer
**R2-Major-13**, which asked us to (1) state *precisely* how the "relevance
scores" are computed — they are **Layer Integrated Gradients** (Sundararajan et
al. 2017), *not* raw attention weights — and (2) map the top-attributed k-mers
back to the L1HS/L1.3 consensus to check whether they cluster in functional
domains (5'UTR promoter, ORF1, ORF2, ORF2 reverse-transcriptase).

Attribution method
------------------
``transformers-interpret``'s ``SequenceClassificationExplainer`` computes **Layer
Integrated Gradients on the word-embedding layer**: it integrates the gradient of
the target-class logit with respect to the token embeddings along a straight path
from a ``[PAD]`` baseline (special tokens preserved) to the real input, then sums
over the embedding dimension to give one signed attribution per token. Positive =
pushes toward the predicted class; negative = pushes against it. We reproduce that
exact quantity here with ``inputs_embeds`` interpolation (no ``captum`` runtime
dependency; the computation is identical). For the contrast the reviewer asks for,
we also record the naive **last-layer self-attention received by each token**
(head-averaged) — the "attention as importance" proxy that is *not* a reliable
attribution — so the notebook can show the two rankings disagree.

Biological mapping
------------------
Every non-special token is a stride-1 canonical k-mer at a known read offset
(``[CLS] R1_kmers [SEP] R2_kmers [SEP]``; token ``1 + j`` of mate R1 is
``read[j : j + k]``). Each read is aligned to the L1.3 consensus (L19088.1) with
minimap2 (``trap.analysis._l1_align``); a k-mer's L1.3 coordinate is the alignment
midpoint shifted by its offset from the read centre (a linear approximation — L1
functional domains are ~kb-scale, so this is robust), and its ``domain`` is the
functional region containing that coordinate.

Stage A (this module, Grace GPU + minimap2). Stage B renders the figures in
``notebooks/5.07-ach-model-interpretability.ipynb``.

Usage (1 GPU, seqlabel model)::

    python -m trap.analysis.interpret attribute \\
        --model-path models/albert.l1hs_l1pa2.v48.k17.salmon.seqlabel/final \\
        --tokenizer-path models/tokenizer.gencode.v48.k17.salmon \\
        --r1 data/external/l1hs_l1pa2_negative.seqlabel.5x_R1.fq \\
        --r2 data/external/l1hs_l1pa2_negative.seqlabel.5x_R2.fq \\
        --l1-consensus data/external/L19088.1.fa \\
        --out-dir results/interpret
"""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

from loguru import logger
import typer

from trap.config import manifest as manifest_mod
from trap.config.config import EXTERNAL_DATA_DIR, MODELS_DIR, RESULTS_DIR
from trap.utils.io import genome_file_handle
from trap.utils.labels import task_label

app = typer.Typer(add_completion=False, help="K-mer attribution + L1.3 domain mapping.")


@app.callback()
def _main():
    """Force sub-command dispatch so ``… interpret attribute`` is invoked by name.

    (A single-command Typer app otherwise collapses and rejects the command name.)
    """


# Reads whose primary L1.3 alignment is at least this good get a consensus
# coordinate; weaker ones are labelled 'unaligned' (mostly NEGATIVE + divergent L1).
_MIN_IDENTITY = 0.70
_MIN_ALIGNED_FRAC = 0.50


def _raw_subfamily(read_id: str) -> str:
    """RepeatMasker subfamily from a TrAP record id (last ``|`` field)."""
    parts = read_id.split("|")
    raw = parts[-1].split("-")[0] if len(parts) >= 2 else read_id
    return raw.upper()


def _sample_reads(r1: str, r2: str, per_class: int, max_scan: int):
    """Scan paired FASTQ; keep up to ``per_class`` reads for each task class.

    Returns ``{class: [(read_id, s1, s2, subfamily), ...]}``. NEGATIVE is the last
    (multi-million-read) block, so we stop once every class has its quota.
    """
    kept = defaultdict(list)
    n = 0
    with genome_file_handle(r1) as h1, genome_file_handle(r2) as h2:
        while n < max_scan:
            id1 = h1.readline()
            s1 = h1.readline()
            h1.readline()
            h1.readline()
            h2.readline()
            s2 = h2.readline()
            h2.readline()
            h2.readline()
            if not id1 or not s1 or not s2:
                break
            n += 1
            tok = id1[1:].split()
            read_id = tok[0] if tok else f"read{n}"
            sf = _raw_subfamily(read_id)
            cls = task_label(f"x|{sf}")
            if cls == "OTHER" or len(kept[cls]) >= per_class:
                if all(len(kept[c]) >= per_class for c in ("L1HS", "L1PA", "NEGATIVE")):
                    break
                continue
            kept[cls].append((read_id, s1.strip(), s2.strip(), sf))
    return kept


def _token_layout(input_ids, sep_id: int, pad_id: int):
    """Return ``[(pos, mate, offset)]`` for every non-special token in a row.

    ``[CLS] R1_kmers [SEP] R2_kmers [SEP] <pad>...`` — the j-th R1/R2 k-mer token
    is ``read[j : j+k]``; truncation (if any) drops tail k-mers, so offsets stay
    contiguous from 0.
    """
    sep_positions = [i for i, t in enumerate(input_ids) if t == sep_id]
    if not sep_positions:
        return []
    first_sep = sep_positions[0]
    layout = [(i, "R1", i - 1) for i in range(1, first_sep)]
    if len(sep_positions) >= 2:
        second_sep = sep_positions[1]
        layout += [(i, "R2", i - first_sep - 1) for i in range(first_sep + 1, second_sep)]
    return layout


@app.command()
def attribute(
    model_path: Path = typer.Option(..., help="Trained classifier dir (…/final)."),
    tokenizer_path: Path = typer.Option(
        MODELS_DIR / "tokenizer.gencode.v48.k17.salmon", help="Salmon k-mer tokenizer dir."
    ),
    r1: Path = typer.Option(..., help="L1 R1 FASTQ (raw subfamily in the read ids)."),
    r2: Path = typer.Option(..., help="L1 R2 FASTQ."),
    l1_consensus: Path = typer.Option(
        EXTERNAL_DATA_DIR / "L19088.1.fa", help="L1.3 consensus FASTA (L19088.1)."
    ),
    k: int = typer.Option(17, help="K-mer size."),
    max_position_embeddings: int = typer.Option(280, help="Pad/truncate length."),
    per_class: int = typer.Option(300, help="Reads sampled per task class for the domain stats."),
    n_examples: int = typer.Option(2, help="Example reads PER class saved for the heatmap."),
    ig_steps: int = typer.Option(50, help="Riemann steps for Integrated Gradients."),
    target: str = typer.Option(
        "predicted", help="Attribution target class: 'predicted' or L1HS/L1PA/NEGATIVE."
    ),
    max_scan: int = typer.Option(20_000_000, help="Hard cap on FASTQ pairs scanned."),
    seed: int = typer.Option(3469, help="Global seed."),
    out_dir: Path = typer.Option(RESULTS_DIR / "interpret", help="Output directory."),
):
    """Compute per-k-mer IG attribution + attention, map to L1.3 domains, write CSVs."""
    import torch
    from transformers import AlbertForSequenceClassification

    from trap.analysis import _l1_align
    from trap.loaders.tokenizer import load_kmer_tokenizer
    from trap.utils.seeding import set_global_seed

    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_kmer_tokenizer(str(tokenizer_path), max_position_embeddings)
    model = AlbertForSequenceClassification.from_pretrained(str(model_path)).to(device).eval()
    id2label = {i: model.config.id2label[i] for i in range(model.config.num_labels)}
    classes = [id2label[i] for i in range(len(id2label))]
    label2id = {v: i for i, v in id2label.items()}
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    word_emb = model.albert.embeddings.word_embeddings
    logger.info(
        f"device={device} model={model_path} classes={classes} "
        f"k={k} pad={max_position_embeddings} target={target} ig_steps={ig_steps}"
    )

    if target != "predicted" and target not in label2id:
        raise typer.BadParameter(f"target must be 'predicted' or one of {classes}")

    def _encode_one(s1: str, s2: str):
        enc = tokenizer.batch_encode_sequences(
            [s1],
            [s2],
            max_length=max_position_embeddings,
            padding="max_length",
            pad_to_multiple_of=8,
            truncation=True,
        )
        return {
            kk: torch.as_tensor(enc[kk], device=device)
            for kk in ("input_ids", "attention_mask", "token_type_ids")
            if kk in enc
        }

    def _lig_and_attention(inp):
        """Return (per-token IG attribution, per-token attention, pred_id, probs)."""
        input_ids = inp["input_ids"]
        attn_mask = inp["attention_mask"]
        tt_ids = inp["token_type_ids"]

        # Prediction + naive last-layer attention (head-averaged, received per token).
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=tt_ids,
                output_attentions=True,
            )
            probs = out.logits.softmax(-1)[0]
            pred_id = int(probs.argmax())
            last_attn = out.attentions[-1][0].mean(0)  # (q, kv) head-averaged
            attention = last_attn.mean(0)  # attention received by each token (mean over queries)
        tgt = pred_id if target == "predicted" else label2id[target]

        # Baseline: [PAD] everywhere except the preserved special tokens (CLS/SEP/pad).
        special = (input_ids == cls_id) | (input_ids == sep_id) | (input_ids == pad_id)
        baseline_ids = torch.where(special, input_ids, torch.full_like(input_ids, pad_id))

        emb_in = word_emb(input_ids)
        emb_base = word_emb(baseline_ids)
        alphas = torch.linspace(0.0, 1.0, ig_steps, device=device).view(-1, 1, 1)
        grads = torch.zeros_like(emb_in)
        for a in alphas:
            emb_a = (emb_base + a * (emb_in - emb_base)).detach().requires_grad_(True)
            logits = model(
                inputs_embeds=emb_a, attention_mask=attn_mask, token_type_ids=tt_ids
            ).logits
            model.zero_grad(set_to_none=True)
            logits[0, tgt].backward()
            grads += emb_a.grad
        avg_grad = grads / ig_steps
        attributions = ((emb_in - emb_base) * avg_grad).sum(-1)[0]  # (seq,)
        return (
            attributions.detach().cpu().numpy(),
            attention.detach().cpu().numpy(),
            pred_id,
            probs.detach().cpu().numpy(),
        )

    kept = _sample_reads(str(r1), str(r2), per_class, max_scan)
    logger.info(
        "sampled "
        + ", ".join(f"{c}={len(kept.get(c, []))}" for c in classes)
        + f" ({sum(len(v) for v in kept.values())} reads)"
    )

    # ---- attribution pass ---------------------------------------------------
    # per_kmer rows collect every non-special token; read_meta holds read-level
    # info needed to align + place k-mers on the consensus afterwards.
    per_kmer = []  # dicts, filled with l1_pos/domain after alignment
    read_seqs = []  # (mate-specific sequence) parallel list for minimap2
    read_key_of_seq = []  # (read_uid, mate) for each entry in read_seqs
    example_rows = []  # full per-token attribution for a few example reads
    read_uid = 0
    n_ex_per_class = defaultdict(int)

    minimap_ok = _l1_align.minimap2_available()
    if not minimap_ok:
        logger.warning("minimap2 not on PATH — k-mers will not receive L1.3 domains.")

    for cls in classes:
        reads = kept.get(cls, [])
        for read_id, s1, s2, sf in reads:
            inp = _encode_one(s1, s2)
            attr, attention, pred_id, probs = _lig_and_attention(inp)
            input_ids = inp["input_ids"][0].tolist()
            layout = _token_layout(input_ids, sep_id, pad_id)
            uid = read_uid
            read_uid += 1
            mate_seq = {"R1": s1, "R2": s2}
            is_example = n_ex_per_class[cls] < n_examples
            for pos, mate, off in layout:
                seq = mate_seq[mate]
                if off < 0 or off + k > len(seq):
                    continue
                kmer = seq[off : off + k]
                row = {
                    "read_uid": uid,
                    "read_id": read_id,
                    "label": cls,
                    "subfamily": sf,
                    "pred": classes[pred_id],
                    "mate": mate,
                    "offset": off,
                    "kmer": kmer,
                    "attribution": float(attr[pos]),
                    "attention": float(attention[pos]),
                    "read_len": len(seq),
                }
                per_kmer.append(row)
                if is_example:
                    example_rows.append(
                        {**row, "token_index": pos, f"p_{classes[pred_id]}": float(probs[pred_id])}
                    )
            # queue both mates for alignment (positive reads carry the L1 signal).
            read_seqs.append(s1)
            read_key_of_seq.append((uid, "R1"))
            read_seqs.append(s2)
            read_key_of_seq.append((uid, "R2"))
            if is_example:
                n_ex_per_class[cls] += 1

    # ---- align reads to L1.3, place each k-mer on the consensus -------------
    aln_by_key = {}
    if minimap_ok and read_seqs:
        logger.info(f"aligning {len(read_seqs)} mate sequences to {l1_consensus.name} …")
        alns = _l1_align.align_sequences(read_seqs, str(l1_consensus))
        for key, aln in zip(read_key_of_seq, alns):
            aln_by_key[key] = aln

    def _place(row):
        aln = aln_by_key.get((row["read_uid"], row["mate"]))
        if aln is None or aln.identity < _MIN_IDENTITY or aln.aligned_frac < _MIN_ALIGNED_FRAC:
            return None, "unaligned", float("nan"), float("nan")
        # k-mer centre offset relative to the read centre, mapped onto the target.
        delta = (row["offset"] + k / 2) - row["read_len"] / 2
        l1_pos = aln.target_mid + (delta if aln.strand == "+" else -delta)
        return l1_pos, _l1_align.l1_region(l1_pos), aln.identity, aln.aligned_frac

    for row in per_kmer:
        l1_pos, domain, ident, afrac = _place(row)
        row["l1_pos"] = "" if l1_pos is None else round(l1_pos, 1)
        row["domain"] = domain
        row["aln_identity"] = "" if ident != ident else round(ident, 3)  # NaN check
        row["aln_frac"] = "" if afrac != afrac else round(afrac, 3)

    # ---- write outputs ------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kmer_cols = [
        "read_uid",
        "read_id",
        "label",
        "subfamily",
        "pred",
        "mate",
        "offset",
        "kmer",
        "attribution",
        "attention",
        "l1_pos",
        "domain",
        "aln_identity",
        "aln_frac",
    ]
    kmer_csv = out_dir / "per_kmer_attribution.csv"
    with kmer_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=kmer_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_kmer)

    ex_cols = [
        "read_uid",
        "read_id",
        "label",
        "pred",
        "mate",
        "token_index",
        "offset",
        "kmer",
        "attribution",
        "attention",
    ]
    ex_csv = out_dir / "read_examples.csv"
    with ex_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ex_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(example_rows)

    n_aligned = sum(1 for r in per_kmer if r["domain"] != "unaligned")
    manifest_mod.write(
        out_dir,
        seed=seed,
        attribution={
            "method": "Layer Integrated Gradients (word-embedding layer)",
            "reference_baseline": "[PAD] (special tokens preserved)",
            "ig_steps": ig_steps,
            "target": target,
            "library_equivalent": "transformers-interpret SequenceClassificationExplainer",
            "attention_recorded_as_contrast": True,
        },
        model={"path": str(model_path), "classes": classes},
        tokenizer={"path": str(tokenizer_path), "k": k},
        l1_consensus={
            "path": str(l1_consensus),
            "regions": [r[0] for r in _l1_align.L1_3_REGIONS],
        },
        counts={
            "reads": read_uid,
            "kmers": len(per_kmer),
            "kmers_aligned": n_aligned,
            "example_reads": len(set(r["read_uid"] for r in example_rows)),
            "per_class": {c: len(kept.get(c, [])) for c in classes},
        },
    )
    logger.success(
        f"wrote {kmer_csv.name} ({len(per_kmer):,} k-mers, {n_aligned:,} L1.3-placed), "
        f"{ex_csv.name} ({len(example_rows):,} rows), manifest.json → {out_dir}"
    )


if __name__ == "__main__":
    app()
