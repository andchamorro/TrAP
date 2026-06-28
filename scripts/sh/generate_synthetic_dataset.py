#!/usr/bin/env python3
"""Generate one synthetic L1-insertion sample (deterministic port of notebook 4.07).

Inserts full-length L1 *element* sequences into chr1 GENCODE transcripts with
biologically-motivated truncation / internal deletion / point mutation / TSDs, then
writes the modified transcript FASTA + an insertion BED (the ground truth). ART
(run by the shell wrapper) simulates reads from the modified FASTA; salmon quantifies
against the same L1 elements.

Adapted from the legacy generator for data/external inputs:
  * reference  = chr1 GENCODE transcripts (was data/ref/gencode.v47 + GTF subset);
  * L1 inserts = full-length L1 extracted from the genome at the LINE1.promoter.bed
    coords (was L1Base IntactL1ElementsFLI). No per-element ORF annotation, so the
    5′/3′-UTR boundaries for truncation use the L1.3 (L19088.1) structure as
    *fractions* of element length (5′UTR≈0–15%, 3′UTR≈96–100%) — 5′-biased truncation,
    matching L1 biology.
  * DETERMINISTIC: a fixed --seed (the legacy set none), so the dataset is reproducible.

Usage (one grid cell; the .sh wrapper loops power × del_prob and runs ART):
  python generate_synthetic_dataset.py \
      --transcripts chr1_transcripts.fa --l1-elements l1_fulllength.fa \
      --power 8 --del-prob 0.025 --seed 3469 \
      --out-fasta out.fa --out-bed out.bed
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# L1.3 (L19088.1) structure as fractions of element length — used for UTR-aware,
# 5′-biased truncation when per-element ORF annotation is unavailable.
UTR5_FRAC = 0.15  # 5′UTR ≈ first 15% (L1.3 5′UTR 1–909 / 6064)
UTR3_FRAC = 0.96  # 3′UTR ≈ last 4%  (L1.3 3′UTR 5815–6064 / 6064)


def simulate_l1_insertion(l1_seq, rng, del_prob=0.1, mutation_prob=0.01):
    """One biologically-realistic L1 insertion: 5′-truncate, del, mutate, TSD, strand.

    ``l1_seq`` is a full-length L1 element in sense orientation. Returns
    ``(insertion_seq, strand)`` where strand is the random insertion orientation.
    """
    n = len(l1_seq)
    strand = rng.choice(["+", "-"])
    # 5′-biased truncation (incomplete reverse transcription): keep a 3′ portion.
    utr5_end = int(n * UTR5_FRAC)
    truncation_point = rng.randint(0, max(0, utr5_end))
    truncated = l1_seq[truncation_point:]

    # Occasional internal deletion (RT error).
    if rng.random() < del_prob and len(truncated) > 1:
        ds = rng.randint(0, len(truncated) - 1)
        de = rng.randint(ds, len(truncated) - 1)
        truncated = truncated[:ds] + truncated[de:]

    # Point mutations.
    bases = list(str(truncated))
    for i in range(len(bases)):
        if rng.random() < mutation_prob:
            bases[i] = rng.choice(["A", "T", "C", "G"])
    mutated = Seq("".join(bases))

    if strand == "-":
        mutated = mutated.reverse_complement()

    tsd_len = rng.randint(5, 15)
    tsd = "".join(rng.choices(["A", "T", "C", "G"], k=tsd_len))
    return Seq(tsd) + mutated + Seq(tsd), strand


def simulate_insertions(reference_records, l1_records, insertions_count, rng,
                        del_prob=0.1, mutation_prob=0.01, min_distance=100):
    """Insert ``insertions_count`` L1 elements into the reference transcripts.

    Returns ``(modified_records, bed_rows)`` where bed_rows is a list of
    ``(seqname, start, end, l1_name, score, strand)`` over the *modified* coords.
    """
    ref_ids = list(reference_records)
    seqs = {rid: list(str(reference_records[rid].seq)) for rid in ref_ids}
    lengths = {rid: len(seqs[rid]) for rid in ref_ids}
    concat_length = sum(lengths.values())
    max_distance = concat_length // (insertions_count + 1)
    if max_distance <= min_distance:
        raise ValueError(
            f"reference too small ({concat_length} bp) for {insertions_count} insertions "
            f"at min_distance={min_distance}"
        )

    # Insertion positions as cumulative random gaps along the ORIGINAL concatenated
    # reference (so locate() maps to original-coordinate (record, local_pos) pairs).
    positions, cur = [], 0
    for _ in range(insertions_count):
        cur += rng.randint(min_distance, max_distance)
        if cur >= concat_length:
            break
        positions.append(cur)

    def locate(global_pos):
        acc = 0
        for rid in ref_ids:
            if global_pos < acc + lengths[rid]:
                return rid, global_pos - acc
            acc += lengths[rid]
        return ref_ids[-1], lengths[ref_ids[-1]]

    # Plan each insertion at its ORIGINAL local position.
    l1_keys = list(l1_records)
    plan = {rid: [] for rid in ref_ids}
    for global_pos in positions:
        rid, local_pos = locate(global_pos)
        key = rng.choice(l1_keys)
        ins_seq, strand = simulate_l1_insertion(
            l1_records[key].seq, rng, del_prob=del_prob, mutation_prob=mutation_prob
        )
        plan[rid].append((local_pos, key, str(ins_seq), strand))

    bed_rows = []
    for rid in ref_ids:
        items = sorted(plan[rid])  # ascending by original local position
        # Splice in DESCENDING order so each insertion's list index stays valid.
        for local_pos, _key, ins, _strand in sorted(items, key=lambda x: -x[0]):
            seqs[rid][local_pos:local_pos] = list(ins)
        # Record FINAL coordinates: ascending, shifting by the cumulative length of
        # all earlier (more 5′) insertions in this record.
        offset = 0
        for local_pos, key, ins, strand in items:
            start = local_pos + offset
            bed_rows.append((rid, start, start + len(ins), key, ".", strand))
            offset += len(ins)

    modified = [
        SeqRecord(Seq("".join(seqs[rid])), id=rid, description="") for rid in ref_ids
    ]
    bed_rows.sort()
    return modified, bed_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcripts", required=True, help="chr1 transcript FASTA (insertion reference).")
    ap.add_argument("--l1-elements", required=True, help="Full-length L1 element FASTA (inserts).")
    ap.add_argument("--power", type=int, required=True, help="insertion_count = 2**power.")
    ap.add_argument("--del-prob", type=float, required=True)
    ap.add_argument("--mutation-prob", type=float, default=0.01)
    ap.add_argument("--min-distance", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3469, help="Determinism (legacy set none).")
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--out-bed", required=True)
    args = ap.parse_args()

    # Seed is salted by (power, del_prob) so each grid cell is a distinct but
    # reproducible draw.
    rng = random.Random((args.seed, args.power, round(args.del_prob, 3)).__hash__())

    reference = {r.id: r for r in SeqIO.parse(args.transcripts, "fasta")}
    l1 = {r.id: r for r in SeqIO.parse(args.l1_elements, "fasta")}
    if not reference or not l1:
        raise SystemExit("empty reference or L1 element FASTA")

    count = 2 ** args.power
    modified, bed_rows = simulate_insertions(
        reference, l1, count, rng,
        del_prob=args.del_prob, mutation_prob=args.mutation_prob, min_distance=args.min_distance,
    )

    Path(args.out_fasta).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(modified, args.out_fasta, "fasta")
    with open(args.out_bed, "w") as fh:
        for row in bed_rows:
            fh.write("\t".join(str(x) for x in row) + "\n")
    print(f"[gen] power={args.power} del_prob={args.del_prob:.3f}: "
          f"{len(bed_rows)} insertions of {len(l1)} L1 elements → {args.out_fasta}")


if __name__ == "__main__":
    main()
