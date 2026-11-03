# Route N query side — pre-registration freeze

Written **2026-10-15**, after the reference side was fixed and the references were
rebuilt, and **before any query-side estimate existed**. Plan §7a asks for this so the
record shows what was decided before the numbers arrived; the point of the timestamp is
that nothing below can have been chosen to fit a result.

At this moment: `query_build_refs` had completed (job 20160099); `query_fetch` was in
flight; no Salmon, L1EM, AlbertSalmon or AlbertEM value for any VM-Cub-1 library
existed anywhere on disk.

Repository state at the freeze: HEAD `582cec15d3f9b2914779d58ddd7a8f0b55c9ad32`. That SHA no longer exists: the history
was rewritten in 2026-11 to purge two unpublished draft files, which renumbered every
commit. The tree it named is preserved under the tag `v1.0.0`.

## Files whose content is frozen

Paths are **as of the freeze**, when the route code lived in the author's private
working directory. The second table below re-verifies the same content at the paths it
occupies now.

| File, as frozen | sha256 |
|---|---|
| `trap/analysis/robustness.py` | `02f6a958afe7b3037712cf1520fc8478dff7275fe46856ebeb943a29a260099c` |
| `.trap/scripts/aggregate_srr_l1em_exp.py` | `fb87bfc6f65d4f12f41f3c9e4ba5c641a6ccb43eab2b8ca4ff5cc1c1a7b2485d` |
| `.trap/route-n-l1rip/l1rip_fulllength.py` | `37b6d551977e475ac150b81c3ab9c9259e48440102f6506f77823abbac761789` |
| `.trap/route-n-l1rip/l1rip_query_reference.py` | `5272cb9e3468a9810ef0363d6553b066b8bb2e836a0cf9b6c79cb81f16ee2d91` |
| `.trap/route-n-l1rip/l1rip_g3a_orf_defined.py` | `009b5310f1fd7d095daff234e149cb263a68c1e5a7e6a6265a0361b98db20771` |
| `.trap/route-n-l1rip/l1hs_l1pa_fpm.py` | `9cf13d732af016cb1b85c0372174d2593432fa384729de5a7c3da83cd1ed37b8` |
| `results/l1rip_calibration/counts.l1em.route-n.pooled.txt` | `44431d6cc740959d5ce22b7dceff86b638d400c114d28c13f5434ef1b8df8d2b` |

## Re-verification after promotion to tracked paths (2026-11-02)

The route code was promoted out of the private working directory into this tracked
directory. Three of the seven files no longer hash to the frozen value, so each was
checked for behavioural change rather than assumed intact. The counts file and two
analysis scripts are byte-identical, which is the part that matters most: the frozen
x-axis is literally the same bytes.

| File, now | Byte-identical to freeze? | If not, what changed |
|---|---|---|
| `results/l1rip_calibration/counts.l1em.route-n.pooled.txt` | **yes** | — |
| `route_n_l1rip/l1rip_fulllength.py` | **yes** | — |
| `route_n_l1rip/l1rip_g3a_orf_defined.py` | no | Docstring only: two pointers into the private plan directory replaced by the facts they pointed at. AST-equivalent with docstrings excluded. |
| `route_n_l1rip/l1rip_query_reference.py` | no | Docstring paths, plus one f-string constant: the provenance line written into the output header now names the script's new path. AST-equivalent apart from that one string. |
| `route_n_l1rip/l1hs_l1pa_fpm.py` | no | Edited in the private directory after the freeze, before promotion, and that directory is not version-controlled, so the intermediate edit cannot be recovered or dated. **Treat this one file as not covered by the freeze.** |
| `trap/analysis/robustness.py` | no | Two documentation commits after the freeze (`3d58430`, `a625956`). Proven AST-equivalent to the frozen version with docstrings excluded; the 13/12-line diff is comments and docstrings. |
| `scripts/aggregate_srr_l1em_exp.py` | no | The frozen copy lived at `.trap/scripts/` and was never committed; promoting it applied isort. Verified against the surviving `.pyc` of the frozen source: all 7 code objects present, every function's `co_names` and constants identical, differing only in module-level import order. The `race5 = 100 × count / total` definition is unchanged. |

Reproduce this check with `shasum -a 256` against the first table.

## Decisions fixed here

1. **x-axis** = the pooled **5′-restricted** per-locus counts of the two ORF1p-RIP
   libraries, `counts.l1em.route-n.pooled.txt` (19,316 reads over 5,542 detected loci),
   converted by the aggregator to `race5 = 100 × count / total`. **Not** the
   whole-interval column, which is passenger-dominated (14.0 % young share vs 50.3 %).
2. **Declared sensitivity analyses**, both written before the query side ran:
   the two single-library x-axes (`Library_8`, `Library_10`), because §11 established
   the libraries differ in composition beyond counting noise.
3. **Primary locus set** = `trap.analysis.robustness.PRIMARY_LOCUS_SET` — common ∩
   reference-detected (finite estimate from all four methods and `race5 > 0`),
   unchanged from the HEK293 analysis.
4. **Statistics**, unchanged from HEK293: Pearson R² and Spearman ρ; paired bootstrap
   ΔR² at 5,000 resamples, seed 3469; leave-one-out for the dominant and
   largest-residual loci; primary endpoints are the paired **within-quantifier**
   contrasts (AlbertSalmon − Salmon, AlbertEM − L1EM).
5. **Restrictions reported**: per track (FLI, ORF2, FLnI) and per composite (`union`,
   `orf_defined`), each with `leverage_share_pct`, > 6 kb body floor,
   per-track flanks from `L1BASE_TRACKS`.
6. **G3a passed before the freeze** and its numbers are already on disk
   (`results/l1rip_calibration/l1rip_g3a_*.csv`): `orf_defined` leaves 195
   detected loci, 69 at ≥ 10 reads, leverage 8.5 %. Because 69 > 50, §11's stop
   condition is cleared and **the paired bootstrap is licensed on the ORF-defined set**.
   `ORF2` alone is declared too thin to fit (12 loci at ≥ 10 reads, 0 clearing 1 %) and
   will be reported descriptively whatever it shows.
7. **Depth**: every replicate subsampled to SRR3997504's 41,368,288 pairs,
   `seqkit sample -p`, seed 3469. Chosen from the published run's read count.
8. **Classifier**: `albert.l1hs_l1pa2.v48.k17.salmon.seqlabel`, k = 17, tokenizer
   `tokenizer.gencode.v48.k17.salmon`, **τ = 0.5** on P(NEGATIVE), per plan §7b. The
   retained-read fraction will be recorded in `trap/<ACC>/filter_summary.tsv`.
9. **Salmon library type** is pinned to whatever `query_probe.slurm` measures with
   `-l A`, and the same value is used for both the Salmon and AlbertSalmon arms. It is
   not re-inferred per quantification.

## Known asymmetries, declared rather than discovered

- The published HEK293 **AlbertSalmon** column came from the older k = 18 repeatmasker
  classifier (`run/outputs/quantify_k18_SRR3997504`), not the k = 17 salmon seqlabel
  model fixed above. The primary endpoints are within-library contrasts and never
  compare a VM-Cub-1 Albert value to a HEK293 Albert value, so this does not affect
  them — but HEK293 is **not** a comparator for the Albert columns, only for locus
  counts and dynamic range. This belongs in Methods.
- ~~The rebuilt salmon index reproduces all 10,643 published targets and adds **23** the
  legacy build dropped. If any of the 23 enters the matched set it is a locus HEK293
  could not have contributed, and it will be reported, not silently included.~~
  **Withdrawn 2026-10-16: there is no asymmetry.** `salmon index` deduplicates —
  *"Removed 23 transcripts that were sequence duplicates of indexed transcripts"* — so
  10,666 input sequences yield 10,643 targets, and the published build did the same. The
  published and rebuilt `quant.sf` agree on `(Name, Length)` for **10,643 of 10,643, with
  zero on either side**. Nothing needs excluding from the matched set. The residue is a
  reference property, not a comparability problem: 23 category-1 loci (mostly chrY
  palindrome pairs) are byte-identical in sequence and so are not individually resolvable
  by any sequence-based quantifier; salmon pools each pair into the retained copy.
  See `salmon_ref/only.Index/duplicate_clusters.tsv`.
- Route N remains, per §11: a **descriptive locus-ranking reference with its CI
  attached** — pooled young share 50.3 % (95 % CI 45.9–54.7) — not a 5′-initiation
  reference, and it validates no pipeline independently (0 of 18 locus-set definitions
  reproduce the published 90 loci / > 60 % top-10). Nothing in this phase can change
  those three statements; they are properties of the reference arm.

## Stop conditions

- Matched-locus count under ~50 after the four-method intersection → report descriptive
  concordance and **decline** the paired bootstrap (§11).
- `query_probe.slurm` failing, or L1EM producing counts that do not reconcile with
  `baminfo.txt` → stop; the python 3.12 / numpy 2 stack is then not usable for L1EM and
  the EM arms cannot be run at all.
- Salmon's observed library type disagreeing with the pinned value in any arm → stop and
  re-pin, because that difference would otherwise masquerade as a method effect.

---

## Addendum, 2026-10-15, after `query_probe.slurm` — appended, nothing above rewritten

Item 9 above left the Salmon library type to be measured. It was, and the measurement
turned out to decide something larger than a flag, so it is recorded here rather than
folded silently into the scripts.

**The library is unstranded: `IU`.** Pinned for both the Salmon and AlbertSalmon arms.
Not taken from salmon's `expected_format` verdict — from the fragment counts behind it,
on 2 M pairs of each replicate:

| run | ISF | ISR | strand_mapping_bias | verdict |
|---|---|---|---|---|
| SRR14758878 | 873 | 868 | 0.501 | IU |
| SRR14758879 | 931 | 1,047 | 0.471 | IU |
| SRR14758880 | 811 | 890 | 0.496 | IU |

That is inside the 49.65–50.35 % band the sibling RobaleTE/Sertorius project uses as its
own unstranded criterion, and it is consistent across all three libraries.

**Consequence not anticipated in the freeze above: L1EM must run its unstranded model.**
This is not a flag. `run_L1EM_unstranded.sh` substitutes `G_of_R_unstranded.py` for
`G_of_R.py` and emits a different schema — `full_counts.txt` carries
`only / 3prunon / passive` instead of the five stranded columns — because sense and
antisense cannot be separated without strand information. Running the stranded driver on
this library would have manufactured a sense/antisense split out of the EM's
initialisation, which is the failure Sertorius's ADR-0005 D-1 declines to ship and the
reason it dropped its antisense arm entirely. Both L1EM jobs now take
`L1EM_STRAND` (default `unstranded`), and `l1hs_l1pa_fpm.py` reads the schema off the
file rather than assuming the stranded column names, which it previously did and which
would have raised `KeyError` on the first real run.

**A new declared asymmetry, alongside the k=18 classifier one.** The published HEK293
query library `SRR3997504` is **`ISR`** (SMARTer Stranded); this one is **`IU`**. So the
two benchmarks differ in library chemistry as well as in cell line and protocol. The
primary endpoints are within-library contrasts and are unaffected, but **HEK293 is not a
comparator for anything strand-dependent**, and the EM arms of the two benchmarks are
produced by two different L1EM models. This must be stated in Methods.

**Scope limit on the measurement — stated, then CLOSED 2026-10-16.** The `IU` call was
first taken on ~3,000 fragments against the L1-only index, which is a narrow basis for a
decision that selects an EM model. It has now been corroborated genome-wide and
**without any gene annotation** (`query_strandedness.slurm`, job 20164846, 11 s, 2 cores,
~0.006 SU).

The test bins chr1 at 10 kb and takes the fraction of primary R1 alignments on the +
strand per bin. A stranded library reads each gene in one orientation, so the across-bin
distribution is bimodal at 0 and 1; an unstranded library puts every sufficiently-covered
bin near 0.5. Gene strand cancels, so no annotation is required.

| over 4,262 bins with ≥ 50 R1 reads | |
|---|---|
| mean R1-forward fraction | **0.4985** |
| sd across bins | 0.0577 |
| bins in [0.40, 0.60] | **4,050 / 4,262 = 95.0 %** |
| bins ≤ 0.15 or ≥ 0.85 | **24 / 4,262 = 0.6 %** |

Sharply unimodal and central: `0.4-0.5` holds 2,005 bins and `0.5-0.6` holds 2,039, with
single digits in every bucket beyond 0.3-0.7. **The library is unstranded, and
`run_L1EM_unstranded.sh` is the correct model.** Full histogram:
`strandedness/SRR14758878.chr1.summary.txt`.

**Cost, re-measured and materially different from the record.** `query_probe.slurm`
timed L1EM on this library at 8 threads: **316 s for 200,000 pairs unstranded** (571 s
stranded), extrapolating to **~18 h** for 41,368,288 pairs — against the ~4.6 h the
repo's own note records. The EM is not the cost: only **212 of 200,000 pairs** reached
it. The bill is `bwa aln` in STEP 1 and STEP 3, both read-count-proportional, which is
why the linear extrapolation should roughly hold. `query_l1em.slurm` is therefore sized
at 48 h on the `long` partition; `medium` caps at 24 h.

**One stop condition is already being approached.** At probe depth **0 of 76** loci carry
≥ 100 proper read pairs — McKerrow's own validity floor, below which L1EM's r² = 0.83
agreement with simulation is not claimed. Probe depth is 1/207 of full depth so this
should clear, but it is now printed on every run and it is the quantity most likely to
shrink the matched set. If the full-depth matched set lands under ~50 loci, §"Stop
conditions" above applies unchanged: descriptive concordance, no paired bootstrap.
