# Decision records

One page per decision that shaped a reported result and is not obvious from the code.
Each record states what was decided, the evidence for it (with the path that regenerates
it), what it costs, and what would reopen it. They exist so that a reader of the code or
the manuscript can tell a deliberate choice from an accident.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-kmer-length-17.md) | Keep k = 17 although the selection analysis recommends 19 | accepted, with a known gap |
| [0002](0002-retire-mlm-pretraining.md) | Train the classifier directly; retire MLM pre-training | accepted |
| [0003](0003-sequence-anchored-labels.md) | Label reads by sequence content, not by coordinate overlap | accepted |
| [0004](0004-figure-11-filter.md) | Keep the published Figure 11 locus filter and add a reference-detected set | accepted |
| [0005](0005-5race-negative-screen.md) | Report that no second 5′RACE library exists, and that no substitute protocol works | accepted |

New records take the next number, are never renumbered, and are superseded by a later
record rather than edited into a different decision.
