# synthetic_albertsalmon_scatter

**Title**: Total-level abundance recovery
**Plot type**: plot
**Objective**: Show the total (family-level) correspondence between recovered salmon abundance and simulated abundance per grid cell, with the linear fit and total-level R² (≈1.0) — the level at which the model recovers abundance.
**Aspect ratio**: 1:1
**Language**: R
**Data sources**: abundance_csv

---

## Data

**Inputs used**: `results/synthetic_validation/abundance.csv`

**Load snippet**:

```r
ab <- data.table::fread(file.path(results_dir, "abundance.csv"))
```

**Output variable**: `ab`

---

## Preprocessing

**Transformations**: sum to the total per grid cell; the total-level R² is the annotation.

```r
ab_tot <- ab[, .(sim = sum(simulated), est = sum(albertsalmon_seqlabel)), by = .(power, del_prob)]
r2_tot <- r2(ab_tot$sim, ab_tot$est)
```

**Output variable(s)**: `ab_tot`, `r2_tot`

---

## Visualization

**Chart type**: scatter_regression
**Palette**: Okabe-Ito (points `#56B4E9`, fit `#D55E00`)
**Style notes**: 45 per-cell points; lm fit with SE band; R² annotated top-left.

```r
fig_scatter <- ggplot(ab_tot, aes(sim, est)) +
  geom_point(size = 2.6, alpha = 0.75, colour = OKABE_ITO[2]) +
  geom_smooth(method = "lm", se = TRUE, colour = OKABE_ITO[6], fill = OKABE_ITO[6]) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.15, vjust = 1.6,
           label = sprintf("R^2 == %.3f", r2_tot), parse = TRUE, size = 5) +
  labs(x = "Simulated total L1 expression (transcript copies)",
       y = "Recovered total (salmon NumReads)")
```

---

## Export

**Output path**: `reports/figures/synthetic_validation/synthetic_albertsalmon_scatter.png` (+ .pdf)
**DPI**: 300

```r
save_fig(fig2, "synthetic_albertsalmon_scatter", 5, 5)
```

---

## Notes

**Caption**: Per-element scatter of classifier-filtered salmon abundance against the simulated insertion count, pooled across all grid cells, with the linear fit and overall coefficient of determination.

**Critic passes**: 1
**Revisions**: none
