# Route N — L1-ORF1p RIP nanopore (VM-Cub-1) as a 5′RACE substitute

**Verdict: passed composition and was promoted to a query-side experiment; retained as a
descriptive locus-ranking reference, not as a 5′-initiation benchmark.** This is the route
that went furthest, and the only one that produced quantitative agreement with the
reference rather than just a rejection.

## Why this route was worth the compute

Routes A and O both fail for the same underlying reason: they are untargeted with respect
to L1, so most of what they capture is passenger L1 sequence inside other transcripts.
PRJNA657989 is different. It is an L1-ORF1p RNA immunoprecipitation — ORF1p binds its own
transcript, so pulling on ORF1p selects for genuinely L1-derived RNA. It is L1-targeted
by mechanism rather than by primer design, which is the closest available analogue to
what 5′RACE does.

The cost is the cell line: VM-Cub-1, not HEK-293, so no same-line comparison is
available. That is why the arm is judged on *composition* — does the read population
look like a 5′RACE population — rather than on correlation against the published
reference.

## The honest weakness, stated up front

Only 22–33 % of reads start in the first 1 % of their element. The library is
**partly** 5′-anchored, not fully. Everything below therefore runs on the
5′-end-restricted subset, and the whole-interval column is reported alongside it
precisely because it is passenger-dominated: 14.0 % young share against 50.3 % for the
5′-restricted population drawn from the same libraries. That dissociation *within one
library* is the central measurement of this arm.

## The result

| Quantity | Value |
|---|---|
| 5′-restricted young (L1HS+L1PA2) share | **50.3 %** (95 % CI 45.9–54.7) |
| Reads behind it | 19,316 over 5,542 detected loci |
| Whole-interval young share, same libraries | 14.0 % |
| ρ vs the reference on full-length young loci, ≥ 10 reads (n = 165) | **0.807** |
| That ρ as a fraction of the counting-noise ceiling | **94 %** |
| Published 5′RACE anchor | 74.0 % young share |

The ceiling fraction is the number that matters and the reason this arm passed. A
multinomial resampling of the observed counts cannot exceed ρ ≈ 0.86 at this depth, so
0.807 leaves almost no room for genuine disagreement — the residual is sampling noise,
not a different answer. Reporting ρ alone would have understated the agreement badly.

Three blockers were raised against the arm and then measured rather than accepted. The
headline replicate agreement of ρ = 0.41 proved to be an artifact of the locus set: it
was computed over all 122,626 intervals, of which only 1,486 carry a read in both
libraries, so it largely measured "do the same loci get *any* read" on a vector that is
~98 % ties at zero. The pre-registered verdict was deliberately left untouched when this
was found; the correction is recorded next to it rather than substituted for it.

## Pre-registration

`FREEZE.md` was written on 2026-10-15, after the references were rebuilt and **before any
query-side estimate existed anywhere on disk**. It fixes the x-axis definition, the
filters, the stop conditions and the content hashes of the analysis scripts. Its purpose
is that nothing in it can have been chosen to fit a result. It also carries a
re-verification table covering what changed when this code was promoted out of the
author's private working directory into this tracked one.

## Files

Reference side — does this library look like 5′RACE?

| File | Role |
|---|---|
| `submit_l1rip_calibration.sh` | sbatch wrapper; reads `slurm.account` from the gitignored `config/hpc/grace.yaml`, passes mail as CLI flags |
| `l1rip_calibration_fetch.slurm` | array 0–1; prefetch → `vdb-validate` → `fasterq-dump` → gzip, with read-count and read-length verification |
| `l1rip_calibration_align_count.slurm` | `minimap2 -ax splice --secondary=no` → `featureCounts` ×2 → L1 assignment rate |
| `l1rip_fulllength.py` | per-run `five_prime` / `body_covered` / `full_length` classification |
| `l1rip_calibration_fulllength.slurm` | runs the above on the two BAMs |
| `l1rip_calibration_report.py` | **the report step**: targeting, protocol shape, published-result check, replicate agreement, G3b, verdict |
| `l1rip_replicate_power.py` | floor-restricted agreement, the multinomial noise ceiling, the Iso-Seq technical-replicate yardstick, locus-cluster bootstrap CIs — local, no allocation |
| `l1rip_fiveprime_profile.py` | where along an element reads start, with the published 5′RACE library as positive control |
| `l1rip_blocker_checks.slurm` | aligns the reference control, runs the positional profile, recounts with unique assignment |
| `l1rip_blocker_checks.py` | turns those outputs into verdicts, including an 18-definition sweep against the published result |

Query side — can the arm carry a quantification comparison?

| File | Role |
|---|---|
| `l1rip_g3a_orf_defined.py` | survival, activity and leverage under each L1Base2 restriction — local, no allocation |
| `l1rip_query_reference.py` | writes the 5′-restricted reference counts in `featureCounts` shape, so the aggregator's `--race5-counts` is this route's own x-axis |
| `l1hs_l1pa_fpm.py` | reconstructs `L1HS-L1PA_FPM.txt`, which no script in the repo produced; the filter was recovered by measuring the published HEK293 file |
| `query_build_refs.slurm` | salmon `only` index, validated target-for-target against the published `quant.sf`, plus the L1EM reference and its bwa index |
| `query_prefetch.sh` | **login node, zero SU**: SRA Tools `prefetch` of the three replicates, ≤ 3 concurrent transfers |
| `query_dump_match.slurm` | `vdb-validate` → `fasterq-dump` → seeded depth-match to 41,368,288 pairs |
| `query_strandedness.slurm` | annotation-free genome-wide strandedness check; decides which L1EM model is correct |
| `query_probe.slurm` | the gate before the expensive runs: does L1EM run on python 3.12 / numpy 2, what does it cost, what library type is this |
| `query_salmon.slurm` | array 0–2; the Salmon arm on all three DMSO replicates |
| `query_align.slurm` | the single `bwa mem` genome BAM every alignment-dependent method consumes |
| `query_l1em.slurm` | `L1EM` (`ARM=plain`) and `AlbertEM` (`ARM=filtered`) from one script, so the arms differ only in which reads went in |
| `query_trap.slurm` | tokenise → classify → `P(NEGATIVE) < τ` filter → `AlbertSalmon` |
| `query_salmon_version.slurm` | is the latest salmon usable here? 1.10.3 vs 2.1.2 defaults vs 2.1.2 with chain-pruning flags, same `only.fa` |
| `salmon_version_compare.py` | per-locus comparison of those three: correlation plus what correlation hides — max relative difference, detection disagreement, efflen drift |

## Inputs and outputs of the report step

Inputs (cached under `results/`, so these re-run locally with no allocation):

- `results/l1rip_calibration/counts.l1em.txt` and `.summary` — `featureCounts` for both libraries
- `results/l1rip_calibration/counts.l1em.unique.txt` — unique-assignment recount
- `results/l1rip_calibration/l1rip_fulllength_counts.csv` — from `l1rip_fulllength.py`
- `results/l1rip_calibration/l1rip_position_hist.csv`, `l1rip_position_summary.csv`
- `results/isoseq_calibration/counts.l1em.SRR4099955.txt` — published 5′RACE counts
- `results/isoseq_calibration/counts.l1em.isoseq_hek293.txt` — the technical-replicate yardstick
- `data/ref/l1base/` — L1Base2 element tracks

Outputs, into `--out-dir` (default `results/l1rip_calibration/`):

- `l1rip_arm_profiles.csv`, `l1rip_targeting.csv`, `l1rip_loci.csv`, `l1rip_top_loci.csv`
- `l1rip_agreement.csv`, `l1rip_g3b.csv`, `l1rip_published_result_check.csv`
- `l1rip_replicate_power.csv`, `l1rip_share_reliability.csv`, `l1rip_signal_concentration.csv`
- `l1rip_g3a_restrictions.csv`, `l1rip_g3a_matched_sets.csv`, `l1rip_g3a_dynamic_range.csv`
- `l1rip_blocker2_position.csv`, `l1rip_blocker3_published_check.csv`

```bash
conda activate trap
python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_calibration_report.py
python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_replicate_power.py
python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_g3a_orf_defined.py
```

## Gotchas

No `module purge` on Grace. No `zcat | head` under `pipefail`. The charge account never
enters a tracked file and mail must be passed as CLI flags, not `#SBATCH` directives.
Verify a deposit's protocol before spending compute — that check is what killed routes A
and O cheaply. `L1EM` is a third-party checkout expected at `external/L1EM`, overridable
with `L1EM_DIR`.
