# 0001 — Keep k = 17 although the selection analysis recommends 19

**Status:** accepted, with a known gap in the evidence.

## Context

Every read is tokenised into canonical k-mers hashed to a fixed vocabulary, so k fixes the
vocabulary, the sequence length the model sees (about 270 tokens for a pair of 150 bp
reads) and every downstream number. The published models use k = 17. The question was
whether the entropy and redundancy of the sequence data justify that value, or merely
fail to contradict it.

## Evidence

`python -m trap.analysis run --config config/analysis/entropy_redundancy.yaml` computes
the bias-corrected k-mer entropy (Miller–Madow, Chao–Shen) of the GENCODE v48
transcriptome for k = 2–20, with rarefaction, and applies a three-gate criterion
(`trap/analysis/selection.py`). Its output, `results/entropy_redundancy/selection.json`:

| Gate | Result |
|---|---|
| Saturation: estimate stable under subsampling | passes for every k up to 20 |
| Marginal gain below τ = 0.05 bit | first at **k = 19** (gain 0.033); k = 17 gains 0.111, k = 18 gains 0.054 |
| Task: k-mer features separate the classes | **uninformative** (see below) |

The data therefore place the onset of diminishing returns at **k = 19, not 17**, and the
plateau is real rather than a sampling artefact. The analysis flags
`corroborates_manuscript_k16_17: false`.

The task gate cannot be used. The ablation on disk scores 0.4546 macro-F1 at *every* k,
which is what a classifier predicting the majority class scores; it was run on 1,412
reads in two classes rather than the full three-class benchmark.

## Decision

Keep k = 17. Changing it would retrain every model and move every reported number, and
no evidence here shows 17 to be wrong: entropy at k = 17 is within 0.11 bit of its value
at k = 20 on a scale of 26.5 bits. The choice is a pragmatic one, not an optimum.

## Consequences

- The manuscript should describe 17 as a choice inside the plateau, not as the point where
  the plateau begins.
- The task gate has not yet been run on the full benchmark, so "k does not limit
  classification" is untested, not shown.

## Reopen if

The ablation on the three-class benchmark shows macro-F1 still rising between k = 17 and
19, or a retraining at k = 19 changes a reported comparison.
