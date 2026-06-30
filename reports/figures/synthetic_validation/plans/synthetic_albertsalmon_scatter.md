# synthetic_albertsalmon_scatter

**Title**: Per-element abundance recovery (pooled)
**Plot type**: plot
**Objective**: Show the element-level correspondence between estimated (salmon) and simulated abundance across all grid cells, with the linear fit and overall R².
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

**Transformations**: none beyond the pooled overall R² annotation (all rows used).

```r
overall_r2 <- r2(ab$simulated, ab$albertsalmon_seqlabel)
```

**Output variable(s)**: `ab`, `overall_r2`

---

## Visualization

**Chart type**: scatter_regression
**Palette**: Okabe-Ito (points `#56B4E9`, fit `#D55E00`)
**Style notes**: dense scatter → alpha 0.25, point size 0.7; lm fit with SE band; R² annotated top-left.

```r
fig2 <- ggplot(ab, aes(simulated, albertsalmon_seqlabel)) +
  geom_point(alpha = 0.25, size = 0.7, colour = OKABE_ITO[2]) +
  geom_smooth(method = "lm", se = TRUE, colour = OKABE_ITO[6], fill = OKABE_ITO[6]) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.15, vjust = 1.5,
           label = sprintf("R^2 == %.3f", overall_r2), parse = TRUE, size = 5) +
  labs(x = "Simulated abundance (insertions)", y = "AlbertSalmon (salmon NumReads)")
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
