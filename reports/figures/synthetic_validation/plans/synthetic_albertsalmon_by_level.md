# synthetic_albertsalmon_by_level

**Title**: Abundance-recovery R² by aggregation level
**Plot type**: plot
**Objective**: Show that recovered-vs-simulated R² rises from per-locus to total, i.e. the model recovers L1 abundance at the aggregate level while near-identical loci are not individually identifiable.
**Aspect ratio**: 3:2
**Language**: R
**Data sources**: abundance_csv

---

## Data

**Inputs used**: `results/synthetic_validation/abundance.csv`

**Load snippet**:

```r
ab <- data.table::fread(file.path(results_dir, "abundance.csv"))
ab[, power := as.integer(power)]
```

**Output variable**: `ab`

---

## Preprocessing

**Transformations**: pooled R² at three levels — per-locus (raw rows), per-subfamily (sum within subfamily per cell), total (sum all per cell).

```r
r2 <- function(x, y) if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) NA_real_ else summary(lm(y ~ x))$r.squared
ab_sub <- ab[, .(sim = sum(simulated), est = sum(albertsalmon_seqlabel)), by = .(power, del_prob, subfamily)]
ab_tot <- ab[, .(sim = sum(simulated), est = sum(albertsalmon_seqlabel)), by = .(power, del_prob)]
overall <- data.table(level = c("per-locus", "per-subfamily", "total (family)"),
                      r2 = c(r2(ab$simulated, ab$albertsalmon_seqlabel), r2(ab_sub$sim, ab_sub$est), r2(ab_tot$sim, ab_tot$est)))
```

**Output variable(s)**: `overall`

---

## Visualization

**Chart type**: bar_simple (labelled)
**Palette**: Okabe-Ito (`#D55E00`, `#E69F00`, `#009E73`)
**Style notes**: value labels above bars; y ∈ [0, 1.08]; no legend.

```r
fig_level <- ggplot(overall, aes(level, r2, fill = level)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.2f", r2)), vjust = -0.4, size = 4.2) +
  scale_fill_manual(values = OKABE_ITO[c(6, 1, 3)], guide = "none") +
  coord_cartesian(ylim = c(0, 1.08)) +
  labs(x = "Aggregation level", y = expression(R^2 ~ "(recovered vs simulated)"))
```

---

## Export

**Output path**: `reports/figures/synthetic_validation/synthetic_albertsalmon_by_level.png` (+ .pdf)
**DPI**: 300

```r
save_fig(fig_level, "synthetic_albertsalmon_by_level", 6, 4)
```

---

## Notes

**Caption**: Coefficient of determination between recovered classifier-filtered salmon abundance and simulated L1 abundance at three aggregation levels — per-locus, per-subfamily, and total — showing recovery is identifiable at the aggregate level while individual near-identical loci are not.

**Critic passes**: 1
**Revisions**: refactored from per-subfamily bars to the aggregation ladder after the locus-identifiability finding.
