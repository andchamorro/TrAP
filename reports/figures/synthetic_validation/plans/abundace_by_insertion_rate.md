# abundace_by_insertion_rate

**Title**: Recovered abundance vs insertion level (legacy 4.08 style)
**Plot type**: plot
**Objective**: Show recovered total L1 abundance (bars) tracking the 2^p simulated truth (line) across insertion levels 2^5–2^13, reproducing the legacy 4.08 abundance-by-insertion-level figure for the seqlabel pipeline.
**Aspect ratio**: 16:9
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

**Transformations**: sum to the total per cell, put recovered NumReads on the copy-count scale by one global factor (R² ≈ 1), mean over del_prob per insertion level; the truth is 2^power.

```r
ab_tot <- ab[, .(sim = sum(simulated), est = sum(albertsalmon_seqlabel)), by = .(power, del_prob)]
scale  <- sum(ab_tot$sim) / sum(ab_tot$est)
rate   <- ab_tot[, .(recovered = mean(est) * scale), by = power][order(power)]
rate[, truth := 2^power]
```

**Output variable(s)**: `rate`

---

## Visualization

**Chart type**: bar_simple + overlaid truth line
**Palette**: Okabe-Ito (bars `#0072B2`, truth `#D55E00`)
**Style notes**: log2 y; x = 2^p; bottom legend distinguishing recovered bars from the simulated line.

```r
fig_rate <- ggplot(rate, aes(x = factor(power))) +
  geom_col(aes(y = recovered, fill = "AlbertSalmon (recovered)"), width = 0.8) +
  geom_line(aes(y = truth, colour = "Simulated (2^p)", group = 1), linewidth = 1) +
  geom_point(aes(y = truth, colour = "Simulated (2^p)"), size = 2.4) +
  scale_x_discrete(labels = function(p) parse(text = paste0("2^", p))) +
  scale_y_continuous(trans = "log2") +
  scale_fill_manual(values = c("AlbertSalmon (recovered)" = OKABE_ITO[5]), name = NULL) +
  scale_colour_manual(values = c("Simulated (2^p)" = OKABE_ITO[6]), name = NULL) +
  labs(x = "Insertion level (log2)", y = "Estimated abundance (log2)") +
  theme(legend.position = "bottom")
```

---

## Export

**Output path**: `reports/figures/synthetic_validation/abundace_by_insertion_rate.png` (+ .pdf)
**DPI**: 300

```r
save_fig(fig_rate, "abundace_by_insertion_rate", 8, 5)
```

---

## Notes

**Caption**: Mean recovered total L1 abundance as a function of the simulated expression level from 2^5 to 2^13, tracking the expression gradient across roughly 2.5 orders of magnitude.

**Critic passes**: 1
**Revisions**: refactored to the legacy 4.08 recovered-vs-truth bar+line style at the total level.
