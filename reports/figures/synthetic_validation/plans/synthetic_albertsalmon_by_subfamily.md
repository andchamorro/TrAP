# synthetic_albertsalmon_by_subfamily

**Title**: Per-subfamily aggregated abundance recovery
**Plot type**: plot
**Objective**: Show which L1 subfamilies are recovered well after aggregating near-identical elements to the subfamily level (a stable counterpart to the hard per-element metric).
**Aspect ratio**: 3:2
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

**Transformations**: sum simulated + estimated to subfamily level within each cell, then R² per subfamily across cells; write `r2_by_subfamily.csv`.

```r
ab_sub <- ab[, .(simulated = sum(simulated),
                 albertsalmon_seqlabel = sum(albertsalmon_seqlabel)),
             by = .(power, del_prob, subfamily)]
r2_sub <- ab_sub[, .(r2 = r2(simulated, albertsalmon_seqlabel), n_elem = .N),
                 by = subfamily][order(-r2)]
```

**Output variable(s)**: `r2_sub`

---

## Visualization

**Chart type**: bar_simple (horizontal, sorted)
**Palette**: Okabe-Ito (`#009E73`)
**Style notes**: horizontal bars (long subfamily labels), sorted by R², y ∈ [0,1].

```r
fig3 <- ggplot(r2_sub[!is.na(r2)], aes(reorder(subfamily, r2), r2)) +
  geom_col(fill = OKABE_ITO[3]) +
  coord_flip(ylim = c(0, 1)) +
  labs(x = NULL, y = expression(R^2 ~ "(per-subfamily, aggregated over cells)"))
```

---

## Export

**Output path**: `reports/figures/synthetic_validation/synthetic_albertsalmon_by_subfamily.png` (+ .pdf)
**DPI**: 300

```r
save_fig(fig3, "synthetic_albertsalmon_by_subfamily", 6, 4.5)
```

---

## Notes

**Caption**: Per-subfamily coefficient of determination between classifier-filtered salmon abundance and the simulated insertion count, after aggregating elements to the subfamily level within each grid cell.

**Critic passes**: 1
**Revisions**: none
