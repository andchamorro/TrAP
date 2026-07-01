# Synthetic e2e validation — figure pipeline

R/ggplot figures for the classifier-filter → salmon abundance-recovery benchmark
(see `.trap/plans/synthetic-e2e-validation.md`). Built with the
`scientific-visualization` skill; language **R** (`plans/_language.md`).

## Inputs / outputs
- **Input**: `results/synthetic_validation/abundance.csv` — from
  `python scripts/sh/synthetic_abundance.py` (run after the validation grid lands).
- **Notebook**: `notebooks/5.02-ach-synthetic-e2e-validation.ipynb` (`ir-trap` kernel).
- **Figures** (this dir): `synthetic_albertsalmon_by_level` (aggregation ladder — the
  headline), `synthetic_albertsalmon_scatter` (total-level recovery), and
  `abundace_by_insertion_rate` (recovered vs 2^p truth, legacy-4.08 style); png + pdf, 300 dpi.
- **R² tables**: `results/synthetic_validation/r2_by_aggregation.csv`,
  `r2_by_insertion_rate.csv`, `r2_by_del_probability.csv`.

The central result: recovery R² rises per-locus → per-subfamily → total, because young
full-length L1 are >99% identical (not per-locus resolvable by short reads); the model
recovers L1 abundance exactly at the family/total level.

## Execute
```bash
# 1. build abundance.csv (Grace — needs the insertion BEDs + quant.sf)
python scripts/sh/synthetic_abundance.py
# 2. run the notebook on the ir-trap kernel (jupyter from the base env)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/5.02-ach-synthetic-e2e-validation.ipynb
```

`plans/` holds the per-figure pipeline specs; `cc.json` the caption manifest.
