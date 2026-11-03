# 0003 — Label reads by sequence content, not by coordinate overlap

**Status:** accepted.

## Context

Reads are simulated from the GENCODE transcriptome and originally labelled `L1HS`,
`L1PA` or `NEGATIVE` by genomic overlap: a pair counted as L1HS when *either* mate
overlapped an L1HS RepeatMasker interval by any number of bases (`bedtools pairtobed
-type either`), after STAR placement with up to 100 multimapping loci, and the
non-overlapping mate inherited the label.

## Evidence

Aligning those reads back to the L1.3 consensus (L19088.1, minimap2,
`trap/analysis/_l1_align.py`):

- 150 bp windows cut from L1.3 itself align at about 100 %, so the tooling is sound.
- Reads labelled L1HS or L1PA2 align to L1.3 for only about **5 %** of reads
  (256 of 5,000); older labelled families align for about 0.5 %.

Most reads carrying an "L1" label therefore contain no detectable L1 sequence. The
fuzziness at the L1PA–NEGATIVE boundary follows from the labelling rule, which makes
the boundary a coordinate technicality rather than a sequence property, and no decision
rule or loss applied to those labels could resolve it.

## Decision

Define the positive class by sequence content (`scripts/build_dataset_seqlabel.sh`).
Positives are simulated from the LINE-1 corpus and kept only when at least half of the
read lies inside an L1 instance (`STRICT_FRAC`), labelled by the annotated subfamily.
Negatives come from the transcriptome with no L1 overlap. A final minimap2 pass
re-derives the label from the read sequence and reports agreement, which catches
multimapping artefacts. The classifier is retrained from random initialisation on this
dataset (`config/runs/salmon_seqlabel.yaml`) so that nothing is inherited from the
coordinate-labelled weights.

## Consequences

- Metrics from the two labelings are not comparable; the coordinate-labelled models
  measure agreement with a labelling rule, the sequence-labelled models measure L1
  content.
- The coordinate-labelled code state is tagged `coord-labels-baseline`, and the new
  outputs use separate `seqlabel` names, so the earlier framing stays reproducible.
- The 0.5 threshold is a proxy: STAR aligned at high identity, so overlap fraction stands
  in for identity. The minimap2 cross-check exists to bound that proxy.

## Reopen if

The cross-check shows the strict-overlap label and the sequence-derived label disagreeing
on more than a small fraction of positives.
