# 0005 — Report that no second 5′RACE library exists, and that no substitute protocol works

**Status:** accepted.

## Context

The locus-level comparison rests on one 5′RACE library, `SRR4099955` (HEK293-FRT,
PRJNA341059). A second library in a line with more active L1 loci would strengthen it.

## Evidence

**Enumeration.** `RACE` is a controlled `library_selection` value in SRA, so human RACE
libraries can be listed: 7,903 runs in 146 studies, of which exactly one is L1-related,
`SRR4099955`. The most L1-active lines (2102Ep, NCCIT, PA-1, NTERA-2) have only short-read
RNA-seq. A metadata enumeration speaks only about deposits, so it was paired with a
Europe PMC citation sweep of the source method paper; that found one further L1-targeted
5′RACE experiment (Rangwala et al. 2009), which was never deposited and is a weak
reference (L1HS is 1.2 % of tags, hg18, median read 111 nt).

**Substitute protocols**, each quantified over the same `L1EM.400` annotation with the
same `featureCounts` flags (`scripts/analyses/5race_substitution/`):

| Route | Protocol | Young (L1HS+L1PA2) share | Verdict |
|---|---|---|---|
| reference | L1-targeted 5′RACE, SRR4099955 | 74.0 % | — |
| A | PacBio Iso-Seq, same cell line | 4.6 % (ρ = 0.28) | rejected |
| O | cap-trapped long-read CAGE | 29.0 % vs 44.9 % in its own random-primed control (ratio 0.65) | rejected by its control |
| N | L1-ORF1p RNA-IP, nanopore | 50.3 % (95 % CI 45.9–54.7), ρ = 0.807 at 94 % of the noise ceiling | composition passes; not 5′-anchored by design; kept as a descriptive reference |

The lesson is that 5′ anchoring and L1 targeting are independent, and only targeting
matters: untargeted long-read libraries sample L1 passengers inside abundant host
transcripts, and more depth does not change that.

## Decision

Report the screen and its negative outcome in place of a second benchmark line, and do not
propose any untargeted library as a 5′RACE substitute. Route N is retained as a
descriptive locus-ranking reference, not as a quantitative benchmark.

## Consequences

The only route to new data is generating 5′RACE in a high-activity line. HEK293 is near
the biological norm for L1 activity, so the single-line benchmark is a limitation to be
stated, not a defect to be repaired.

## Reopen if

A new L1-targeted 5′RACE deposit appears, or the Rangwala supplement is shown to carry
usable locus-level data at the relevant resolution.
