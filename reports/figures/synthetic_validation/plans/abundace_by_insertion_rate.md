# abundace_by_insertion_rate

**Title**: Abundance-recovery R² by insertion level
**Plot type**: plot
**Objective**: Show how well classifier-filtered salmon abundance tracks the simulated truth as the L1 insertion level rises from 2^5 to 2^13.
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

**Transformations**: per insertion level (`power`), fit `lm(albertsalmon_seqlabel ~ simulated)` and take R²; write `r2_by_insertion_rate.csv`.

```r
r2 <- function(x, y) if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) NA_real_ else summary(lm(y ~ x))$r.squared
r2_by_level <- ab[, .(r2 = r2(simulated, albertsalmon_seqlabel)), by = power][order(power)]
```

**Output variable(s)**: `r2_by_level`

---

## Visualization

**Chart type**: line_basic (single series)
**Palette**: Okabe-Ito (`#0072B2`)
**Style notes**: theme_bw(14); x = insertion level as 2^p; y ∈ [0,1]; no minor grid.

```r
fig1 <- ggplot(r2_by_level, aes(power, r2)) +
  geom_line(colour = OKABE_ITO[5], linewidth = 1) +
  geom_point(colour = OKABE_ITO[5], size = 2.5) +
  scale_x_continuous(breaks = 5:13, labels = function(p) parse(text = paste0("2^", p))) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "Insertion level", y = expression(R^2 ~ "(AlbertSalmon vs Simulated)"))
```

---

## Export

**Output path**: `reports/figures/synthetic_validation/abundace_by_insertion_rate.png` (+ .pdf)
**DPI**: 300

```r
save_fig(fig1, "abundace_by_insertion_rate", 6, 4)
```

---

## Notes

**Caption**: Coefficient of determination between classifier-filtered salmon abundance and the simulated insertion count, as a function of the synthetic L1 insertion level from 2^5 to 2^13.

**Critic passes**: 1
**Revisions**: none
