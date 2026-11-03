# Route A — PacBio Iso-Seq as a 5′RACE substitute (HEK-293 calibration arm)

**Verdict: rejected by measurement.** Untargeted long-read cDNA does not sample the L1
5′ end the way L1-targeted 5′RACE does.

## Why this arm is the decisive one

The confound in every substitution attempt is the cell line: a different line has
different active loci, so a different locus profile proves nothing. PRJNA1176011 removes
that confound. It contains a PacBio Revio Iso-Seq arm on **HEK-293** — the same line as
the published 5′RACE reference `SRR4099955` (HEK293-FRT). Any remaining difference is
attributable to protocol, so the substitution is measured rather than assumed.

Both sides go through the identical `L1EM.400` SAF with identical `featureCounts` flags
(`-F SAF -M --fraction`). Only the aligner differs — `minimap2 -ax splice:hq
--secondary=no` instead of `bwa mem` — because long reads require it.

## The result

| Quantity | 5′RACE (SRR4099955) | Iso-Seq HEK-293 (4 Revio runs) |
|---|---|---|
| Assigned reads over `L1EM.400` | 32,028 | 165,299 |
| L1HS+L1PA2 share of L1 signal | **74.0 %** | **4.6 %** |
| Loci detected | 1,989 | 7,054 |
| Spearman ρ with 5′RACE, L1HS+L1PA2 | — | **0.28** |

Iso-Seq has five times the L1-assigned depth and still puts a sixteenth of the young-L1
share on it. The disagreement is not depth, and it is not noise: the G3b depth gate
passes comfortably, so the library is deep enough for the comparison to mean something.
Iso-Seq's own top loci are old, truncated L1PA copies inside expressed transcripts —
passengers, not L1 promoter activity.

The 5′-end-restricted counts tell the same story more sharply: 289 Iso-Seq 5′-end reads
against 21,660 for 5′RACE over 1,278 matched elements.

## Files

| File | Role |
|---|---|
| `submit_isoseq_calibration.sh` | sbatch wrapper; reads `slurm.account` from the gitignored `config/hpc/grace.yaml`, passes mail as CLI flags |
| `isoseq_calibration_fetch.slurm` | prefetch → `vdb-validate` → `fasterq-dump` → gzip, with read-count and read-length verification |
| `isoseq_calibration_align_count.slurm` | `minimap2 -ax splice:hq --secondary=no` → `featureCounts` → L1 assignment rate |
| `isoseq_5p_ends.py` | per-run 5′-end-restricted counts from the BAMs (needs the BAMs, so Grace only) |
| `isoseq_calibration_report.py` | **the report step**: arm profiles, protocol concordance, the G3b depth gate, Iso-Seq at the loci 5′RACE calls dominant |

## Inputs and outputs of the report step

Inputs (all cached under `results/`, so the report re-runs locally with no allocation):

- `results/isoseq_calibration/counts.l1em.isoseq_hek293.txt` — Iso-Seq `featureCounts`
- `results/isoseq_calibration/counts.l1em.SRR4099955.txt` — published 5′RACE `featureCounts`
- `results/srr_experiment/l1em_salmon_fpm.csv` — the Figure 11 locus table, for the common-locus set
- `data/ref/l1base/` — L1Base2 element tracks, for the FLI / FLnI / ORF2 restrictions

Outputs, into `--out-dir` (default `results/isoseq_calibration/`):

- `isoseq_arm_profiles.csv` — per-arm shape under three denominator conventions
- `isoseq_concordance.csv` — ρ, Jaccard and log₁₀ R² over ten locus sets
- `isoseq_g3b.csv` — the depth gate
- `isoseq_top_loci.csv`, `isoseq_at_race5_dominant_loci.csv`, `isoseq_vs_race5_loci.csv`

```bash
conda activate trap
python scripts/analyses/5race_substitution/route_a_isoseq/isoseq_calibration_report.py
```

Reproduces ρ = 0.277 on L1HS+L1PA2 and the 74.0 % / 4.6 % young shares above.
