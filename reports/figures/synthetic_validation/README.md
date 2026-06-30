# Synthetic e2e validation — figure pipeline

R/ggplot figures for the classifier-filter → salmon abundance-recovery benchmark
(see `.trap/plans/synthetic-e2e-validation.md`). Built with the
`scientific-visualization` skill; language **R** (`plans/_language.md`).

## Inputs / outputs
- **Input**: `results/synthetic_validation/abundance.csv` — from
  `python scripts/sh/synthetic_abundance.py` (run after the validation grid lands).
- **Notebook**: `notebooks/5.02-ach-synthetic-e2e-validation.ipynb` (`ir-trap` kernel).
- **Figures** (this dir): `abundace_by_insertion_rate`, `synthetic_albertsalmon_scatter`,
  `synthetic_albertsalmon_by_subfamily` (png + pdf, 300 dpi).
- **R² tables**: `results/synthetic_validation/r2_by_insertion_rate.csv`,
  `r2_by_del_probability.csv`, `r2_by_subfamily.csv`.

## Execute
```bash
# 1. build abundance.csv (Grace — needs the insertion BEDs + quant.sf)
python scripts/sh/synthetic_abundance.py
# 2. run the notebook on the ir-trap kernel (jupyter from the base env)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/5.02-ach-synthetic-e2e-validation.ipynb
```

`plans/` holds the per-figure pipeline specs; `cc.json` the caption manifest.
