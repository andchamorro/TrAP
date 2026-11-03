# 5′RACE substitution routes

The locus-level comparison in Figure 11 rests on a single 5′RACE library: the published
HEK293-FRT reference `SRR4099955` (PRJNA341059). A second L1-targeted 5′RACE library, in
a cell line with more active L1 loci, would strengthen it.

**No such library exists.** A controlled-vocabulary enumeration of SRA and ENA, paired
with a Europe PMC citation sweep, returned exactly one L1-targeted 5′RACE *deposit*:
`SRR4099955` itself. The code here is what was run to answer the next question — whether
some other protocol can stand in for 5′RACE — and the answer in all three cases was no.

The three substitution attempts are each a calibration arm, not a result. Each one asks
the same question: *does this protocol measure what 5′RACE measures?* A substitute is only
admissible if it does, and all three were discarded on their own measurements rather than
on judgement. That is why the code is kept: the negative is the reported finding, and a
negative that cannot be regenerated is not evidence.

## The three routes

| Dir | Route | Dataset | Protocol | Verdict |
|---|---|---|---|---|
| `route_a_isoseq/` | A | PRJNA1176011 (HEK-293 arm) | PacBio Revio Iso-Seq | rejected by measurement |
| `route_n_l1rip/` | N | PRJNA657989 (VM-Cub-1) | L1-ORF1p RNA-IP, ONT GridION | passed composition, promoted, then discarded on the query side |
| `route_o_lrcage/` | O | PRJNA936447 (NCI-H1299) | cap-trapped long-read CAGE, PacBio Sequel II | rejected by its own internal control |

Route A is the cleanest comparison and the clearest negative: PRJNA1176011 contains an
Iso-Seq arm on the **same cell line** as the published 5′RACE reference, so the
substitution could be measured directly instead of argued. It put 4.6 % of L1 signal on
L1HS+L1PA2 where 5′RACE puts 74.0 %, at ρ = 0.28 across shared loci. Untargeted
long-read cDNA does not sample the L1 5′ end the way a targeted 5′RACE does.

Route O is the most instructive, because it carries its own control. LRCAGE and LRhex
libraries from the **same biosample** (`SAMN33348225`) differ only in cap selection. If
cap-trapping enriched young full-length L1 5′ ends, the LRCAGE young share would exceed
the LRhex one. It is *lower*: 29.0 % against 44.9 %, a ratio of 0.65. The route fails on
the contrast it was designed to pass, which is a stronger negative than any comparison
against a different cell line could be.

Route N went furthest. ORF1p RIP is L1-targeted, and on composition it passed: a 50.3 %
(95 % CI 45.9–54.7) 5′-restricted young share over 19,316 reads, with ρ = 0.807 on
full-length young loci carrying ≥ 10 reads (n = 165) — 94 % of the counting-noise
ceiling, so most of the residual is sampling noise rather than disagreement. It was
discarded for a reason unrelated to composition; see `route_n_l1rip/FREEZE.md` for the
decisions that were fixed before any query-side number existed.

## Shared conventions

All three arms are quantified over the identical `L1EM.400` SAF with identical
`featureCounts` flags (`-F SAF -M --fraction`); only the aligner differs, because long
reads require `minimap2` where the published short-read reference used `bwa mem`. The
`race5` statistic is `100 × count / total` over all `L1EM.400` intervals — a share of
total L1 signal, not of the library — defined in `scripts/aggregate_srr_l1em_exp.py`
and reused unchanged here, so the two sides of every comparison are commensurable.

"Young" means L1HS + L1PA2 throughout, matching the subfamilies the manuscript
quantifies.

## Running the report steps locally

The report scripts are the part that can be re-run without an HPC allocation: they
consume cached `featureCounts` tables and CSVs under `results/`, and write CSVs plus a
printed summary. Fetching, alignment and counting need Grace and are driven by the
`*.slurm` files through each route's `submit_*.sh` wrapper, which reads the charge
account from the gitignored `config/hpc/grace.yaml` so it never enters a tracked file.

```bash
conda activate trap

# Route A — needs results/isoseq_calibration/ and results/srr_experiment/
python scripts/analyses/5race_substitution/route_a_isoseq/isoseq_calibration_report.py

# Route O — needs results/lrcage_calibration/ and the published 5'RACE counts
python scripts/analyses/5race_substitution/route_o_lrcage/lrcage_calibration_report.py

# Route N — needs results/l1rip_calibration/
python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_calibration_report.py
```

Each writes into its own `results/<route>_calibration/` directory by default; pass
`--out-dir` to send output elsewhere, which is how these are verified without touching
the cached inputs.

`L1EM` is a third-party checkout and is not shipped with TrAP. The SLURM scripts expect
it at `external/L1EM`, overridable with `L1EM_DIR`.

## What is deliberately not here

The long narrative record of each route — the day-by-day reasoning, the cost ledgers,
the blocker evaluations, the gotchas — stays in the author's private working notes. The
conclusions that the manuscript depends on are in `docs/decisions/` and in
`docs/guides/manuscript_provenance.md`. What is tracked here is
the code needed to regenerate the numbers, and `FREEZE.md`, which is evidence about
*when* decisions were made and so cannot be rewritten after the fact.
