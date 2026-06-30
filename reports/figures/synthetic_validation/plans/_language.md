# Language decision

**Language: R** (caller-specified).

Rationale: the figures join a single tidy CSV (`abundance.csv`) and the project's
manuscript-figure convention is R/ggplot notebooks (exemplar `notebooks/5.01-…`,
`ir-trap` Jupyter kernel, `data.table` + `ggplot2`, `ggsave` png+pdf). The assembled
artifact is therefore an `.ipynb` on the `ir-trap` kernel (not `.Rmd`), to match the
repository's existing 5.x R notebooks.

Style: `theme_bw` (repo convention, per 5.01) + the Okabe-Ito colorblind-safe palette,
300 DPI, no author-identifying content in any label/title/filename (anti-leakage).
