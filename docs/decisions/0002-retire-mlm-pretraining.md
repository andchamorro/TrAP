# 0002 — Train the classifier directly; retire MLM pre-training

**Status:** accepted.

## Context

The pipeline began as masked-language-model pre-training of an ALBERT encoder followed by
classification fine-tuning. A 35-hour pre-training run on the k = 17 tokenizer learned
nothing: loss sat at ln(65 541) = 11.09 for all 40 epochs with accuracy near zero.

## Evidence

`scripts/diagnose_mlm_learnability.py` ruled out the usual causes. The learning rate
(5e-4 and 5e-5 behaved identically), mixed precision (fp32 froze too), the library version
(a CPU smoke run learns) and the masking and label construction were all sound.

What remained is structural:

- Overfitting 8–16 real sequences drops the loss from 11.09 to about 7.48 and then
  plateaus. e^7.48 ≈ 1,770 matches the number of distinct masked tokens, so 7.48 *is* the
  marginal token entropy: the model learns token frequency and no context.
- A fixed-mask run (the same mask every step, 600 steps) also stalls at the marginal and
  its gradient norm collapses from 1.01 to 4×10⁻⁴. Memorisation needs no generalisation,
  so the masking draw is not the cause.
- The vocabulary is 65,541 buckets of a splitmix64 feature hash. The hash is an avalanche
  function: neighbouring k-mers share 16 of 17 bases yet their token ids correlate at
  r ≈ −0.03, so the target cannot be predicted from context without inverting the hash.

The k-mers already overlap (stride 1), so overlap is not the missing ingredient.

## Decision

Drop MLM. Fine-tune the classifier from a random initialisation. The shelved stages are
kept under `scripts/slurm/legacy/mlm/` and the `masking` sub-command remains in
`trap/modeling/train.py` for the record; neither is part of the active pipeline.

## Consequences

- The manuscript's description of pre-training no longer matches the released pipeline;
  the README states the departure.
- k = 17 is unaffected: the failure is the hash plus the objective, not k (see 0001).

## Reopen if

An objective that predicts the nucleotide content of a masked k-mer, or a compositional
k-mer embedding, is tried. Enlarging the hash, shrinking it, or re-enabling the
sentencepiece tokenizer would not address the cause.
