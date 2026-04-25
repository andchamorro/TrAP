# Tokenizer — Unigram training debug notes

This page documents the **four-bug stack** that caused a Rust panic in the
SentencePiece Unigram trainer on Grace (tokenizers ≥ 0.19 build
`1764694752139`, November 2025). All four surface as the **same** opaque
message — `called Result::unwrap() on an Err value: Internal` — which is why
they had to be peeled off one at a time. The corresponding fix is in
`trap/loaders/tokenizer.py::train_sentencepiece` and the regression tests
live in `tests/loaders/test_tokenizer_debug.py`.

:::{important}
Bugs 1–3 are reproducible with a tiny corpus. **Bug 4 only triggers at scale**
(a full transcriptome) — the unit tests pass but the SLURM job still panics.
If you have applied fixes 1–3 and still see the panic on real data, it is
almost certainly bug 4.
:::

## Symptom

Running `python -m trap.loaders.tokenizer train ... --k 17 --vocab-size 32000`
on any tokenizers ≥ 0.19 build crashes with:

```
thread '<unnamed>' panicked at
  tokenizers/src/models/unigram/trainer.rs:228:53:
called `Result::unwrap()` on an `Err` value: Internal
```

The panic location is in the Viterbi forward pass of the EM algorithm. It is
an upstream bug (unwrap instead of graceful error return), but the three
conditions that trigger it can be avoided at the call site.

## Bug 1 — Iterator format (primary cause)

`kmer_split_batch` yields `List[str]` — batches of k-mer sentences. In
tokenizers < 0.19, `Iterator[List[str]]` was treated as a batch of
sentences (pre-tokenizer applied to each string). In tokenizers ≥ 0.19 the
semantics changed: each `List[str]` is now treated as a **pre-tokenised
sequence** where each element of the list is a single **word**.

Each "word" is therefore a full k-mer sentence such as:

```
"ACTGACTGACTGACTGA CTGACTGACTGACTGAC TGACTGACTGACTGACT ..."
```

This string contains **space characters**. Space is not in `initial_alphabet
= list("ACGTN")`, so the Viterbi forward pass cannot cover the space position
and returns `None`. The EM step calls `.unwrap()` on that `None` → panic.

**Fix:** flatten the batch iterator:

```python
# Before (broken in >=0.19)
tokenizer._tokenizer.train_from_iterator(
    kmer_split_batch(raw_datasets, batch_size, k),
    ...
)

# After (fixed)
import itertools
tokenizer._tokenizer.train_from_iterator(
    itertools.chain.from_iterable(kmer_split_batch(raw_datasets, batch_size, k)),
    ...
)
```

With `chain.from_iterable`, the iterator yields individual strings; the
pre-tokenizer splits each on spaces into k-mer words.

## Bug 2 — MetaSpace prefix

`SentencePieceUnigramTokenizer` defaults to a `MetaSpace` pre-tokenizer,
which prepends `▁` (U+2581, 3 UTF-8 bytes) to every word. Even after
fixing Bug 1, every k-mer becomes `▁ACTGACTGACTGACTGA`. The `▁` character
is not in `initial_alphabet = list("ACGTN")` → Viterbi can't cover the
first character of any word → same panic.

**Fix:** replace the pre-tokenizer before training:

```python
from tokenizers import pre_tokenizers as _pre

tokenizer = SentencePieceUnigramTokenizer()
tokenizer._tokenizer.pre_tokenizer = _pre.Whitespace()
```

`Whitespace` splits on spaces and adds no prefix. For k-mer sequences this
is semantically correct — k-mers have no natural-language word-boundary
semantics.

## Bug 3 — `max_piece_length` default

`SentencePieceUnigramTokenizer.train_from_iterator` creates a
`UnigramTrainer(max_piece_length=16)` internally. For k = 17, the full
k-mers (17 characters) exceed the limit and are silently skipped during
vocabulary seeding → near-empty initial vocabulary → EM fails.

**Fix:** bypass the wrapper and call `train_from_iterator` directly with a
manually constructed trainer:

```python
from tokenizers import trainers as _tr

trainer = _tr.UnigramTrainer(
    vocab_size=vocab_size,
    special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
    unk_token="<unk>",
    initial_alphabet=list("ACGTN"),
    max_piece_length=k,          # exact for k-mers, no ▁ prefix
    show_progress=True,
)
tokenizer._tokenizer.train_from_iterator(iterator, trainer=trainer, length=n)
```

## Bug 4 — Corpus size / esaxx i32 overflow (the one that survives scale)

This is the bug that the small unit tests cannot reach. The Unigram trainer's
`make_seed_sentence_pieces` concatenates **every** training sentence into one
flat string and builds a suffix array over it via `esaxx_rs::suffix(...).unwrap()`.
That library indexes the string with **`i32`** (max 2,147,483,647 chars). When
the input exceeds the limit, esaxx returns an error and the `.unwrap()` panics
with `Internal`.

k-mer splitting makes this trivial to hit: overlapping k-mers inflate the
corpus by ~(k+1)×. Each base of a length-`L` sequence becomes part of `L-k+1`
overlapping k-mers, each rendered as `k` characters plus a separating space:

```
flat_string ≈ (k + 1) × total_nucleotides
```

| GENCODE size | Flat string (k=17) | vs i32 limit |
|--------------|--------------------|--------------|
| 100 M nt | 1.8 B chars | 0.8× — ok |
| 200 M nt | 3.6 B chars | 1.7× — **overflow** |
| 400 M nt | 7.2 B chars | 3.4× — **overflow** |

The full GENCODE v48 transcriptome is well into the overflow range. The
original `google` SentencePiece branch already guarded against this with
`input_sentence_size=1000000` — the HuggingFace path simply lacked it.

**Fix:** seeded subsampling so the post-k-mer footprint stays under a budget
(`max_training_chars`, default 500 M — safely under i32 and ~6 GB peak memory).
`_select_training_indices` shuffles deterministically and greedily accepts
sequences until the budget is reached, skipping sequences shorter than `k`
(they k-mer-split to `""`):

```python
def _select_training_indices(raw_datasets, k, max_chars, seed):
    idx = list(range(len(raw_datasets)))
    random.Random(seed).shuffle(idx)
    selected, total = [], 0
    for i in idx:
        seq_len = len(raw_datasets[i])
        if seq_len < k:
            continue
        cost = (seq_len - k + 1) * (k + 1)
        if selected and total + cost > max_chars:
            break
        selected.append(i)
        total += cost
    return selected, total
```

## Combined fix (all four)

```python
import itertools
import random
from tokenizers import SentencePieceUnigramTokenizer
from tokenizers import pre_tokenizers as _pre
from tokenizers import trainers as _tr

# Bug 4: cap the post-k-mer corpus under the esaxx i32 limit
indices, _ = _select_training_indices(raw_datasets, k, max_training_chars, seed)
subset = [raw_datasets[i] for i in indices]

tokenizer = SentencePieceUnigramTokenizer()
tokenizer._tokenizer.pre_tokenizer = _pre.Whitespace()   # Bug 2

trainer = _tr.UnigramTrainer(
    vocab_size=vocab_size,
    special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
    unk_token="<unk>",
    initial_alphabet=list("ACGTN"),
    max_piece_length=k,                                   # Bug 3
    show_progress=True,
)
tokenizer._tokenizer.train_from_iterator(
    itertools.chain.from_iterable(                        # Bug 1
        kmer_split_batch(subset, batch_size, k)           # Bug 4 (subset)
    ),
    trainer=trainer,
    length=len(subset),
)
```

The `--max-training-chars` CLI flag exposes the budget; raise it (up to
~1.8 B to stay under i32) if you have the memory and want broader coverage.

## Regression tests

`tests/loaders/test_tokenizer_debug.py` contains fourteen focused unit tests
written with 50 bp synthetic reads that:

1. **Document the iterator shape** — prove `kmer_split_batch` yields
   `List[str]` and that flattening gives `Iterator[str]`.
2. **Reproduce the panic scenario** — show that the list-of-sentences path
   embeds spaces, which are outside the DNA alphabet.
3. **Verify fixes 1–3** — training with the flat iterator completes without
   panic and the resulting vocabulary contains only pure `ACGTN` sub-k-mers
   with no `▁` prefix.
4. **Verify the corpus budget (bug 4)** — `TestCorpusSizeBudget` proves the
   ~(k+1)× inflation, that `_select_training_indices` caps the total under the
   budget, is seeded/reproducible, skips short/empty sequences, keeps all when
   under budget, and trains without panic on a corpus combining every entry
   type (normal reads, a long sequence, near-k, short, and empty).

Run them locally:

```bash
pytest tests/loaders/test_tokenizer_debug.py -v
```

## Affected versions

| Component | Behaviour |
|-----------|-----------|
| tokenizers < 0.19 | `Iterator[List[str]]` = batch of sentences → pre-tokenizer applied to each → works |
| tokenizers ≥ 0.19 | `Iterator[List[str]]` = pre-tokenised sequence → each list element is a word → panics |
| Grace build `1764694752139` (≈ Nov 2025) | ≥ 0.19 semantics, `max_piece_length=16` default |

## See also

- `trap/loaders/tokenizer.py::train_sentencepiece` — production fix
- `tests/loaders/test_tokenizer.py::TestTokenizerTrainCLI` — CLI-level regression test for k=17
- [HuggingFace tokenizers issue tracker](https://github.com/huggingface/tokenizers/issues) — upstream `unwrap` vs. graceful error
