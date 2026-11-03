# Route O — long-read CAGE as a 5′RACE substitute (NCI-H1299)

**Verdict: rejected by its own internal control.**

## Why this arm needs no cross-line comparison

Cap-trapped long-read CAGE (PRJNA936447, PacBio Sequel II) is 5′-anchored by
construction, which is what makes it a candidate. The attraction is that it carries its
own control: the deposit contains **LRCAGE** (cap-selected) and **LRhex** (random-primed)
libraries from the *same biosample*, `SAMN33348225`. The two differ only in cap
selection.

That gives a prediction that does not depend on the cell line at all. If cap-trapping
enriches young full-length L1 5′ ends — the thing a 5′RACE substitute has to do — then
the LRCAGE young share must exceed the LRhex young share. Testing a route against its
own matched control is stronger than testing it against a different cell line, because
no locus-repertoire difference can explain the result away.

## The result

| 5′-end-restricted, position-matched | Assigned reads | Young share | Full-length share |
|---|---|---|---|
| LRCAGE DMSO (cap-selected) | 1,040 | **29.0 %** | 29.6 % |
| LRhex DMSO (random-primed control) | 4,301 | **44.9 %** | 48.0 % |
| LRCAGE DACSB (L1-activated) | 1,847 | 23.2 % | 29.0 % |

The cap-selection contrast is a **ratio of 0.65** — cap selection *depletes* young L1
relative to its own random-primed control, where the route requires enrichment well
above 1. The L1-activation contrast (DACSB vs DMSO within LRCAGE) is 0.80, so activating
L1 did not raise the young share either.

The top 5′ locus makes the mechanism concrete: a 358 bp L1PA8 fragment with 85 LRCAGE
reads against 2 LRhex reads. What cap-trapping enriches here is a short old fragment
sitting at a real TSS — a promoter that happens to contain L1 sequence, not an L1
promoter. For comparison, the published 5′RACE anchor sits at 74.0 % young share and the
rejected Iso-Seq arm at 4.6 %.

A route that fails the contrast it was designed to pass has no further use as a
substitute.

## Files

| File | Role |
|---|---|
| `submit_lrcage_calibration.sh` | sbatch wrapper; reads `slurm.account` from the gitignored `config/hpc/grace.yaml` |
| `lrcage_calibration_fetch.slurm` | prefetch → `vdb-validate` → `fasterq-dump` → gzip, with verification |
| `lrcage_calibration_align_count.slurm` | `minimap2` → `featureCounts` ×2 → L1 assignment rate |
| `lrcage_calibration_5p_ends.slurm` | runs the 5′-end restriction on the BAMs |
| `lrcage_5p_ends.py` | per-run 5′-end-restricted counts and the TSS-precision offset histogram |
| `lrcage_calibration_report.py` | **the report step**: arm profiles, the internal contrasts, the confounded cross-line comparison, G3b, verdict |

## Inputs and outputs of the report step

Inputs (all cached under `results/`):

- `results/lrcage_calibration/counts.l1em.txt` — `featureCounts` for the three Sequel II runs
- `results/lrcage_calibration/lrcage_5p_end_counts.csv` — per-run 5′-end-restricted counts
- `results/lrcage_calibration/lrcage_5p_end_offsets.csv` — TSS-precision histogram
- `results/isoseq_calibration/counts.l1em.SRR4099955.txt` — published 5′RACE counts, used
  only for the explicitly-labelled confounded comparison
- `data/ref/l1base/` — L1Base2 element tracks

Outputs, into `--out-dir` (default `results/lrcage_calibration/`):

- `lrcage_internal_contrasts.csv` — **the decisive table**: cap selection and L1 activation
- `lrcage_arm_profiles.csv` — per-run shape, whole-interval and 5′-restricted
- `lrcage_concordance_confounded.csv` — the cross-line comparison, named as confounded
- `lrcage_g3b.csv`, `lrcage_loci.csv`, `lrcage_top_loci.csv`

```bash
conda activate trap
python scripts/analyses/5race_substitution/route_o_lrcage/lrcage_calibration_report.py
```

Reproduces 29.038 % vs 44.943 %, ratio 0.646, as reported.
