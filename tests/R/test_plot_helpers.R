#!/usr/bin/env Rscript
# Smoke test for scripts/R/trap_plot_helpers.R, the file all seven figure notebooks source.
#
# Exercises every helper on a tiny synthetic table and writes into a temporary directory.
# It does not need results/, so it can run in CI where the gitignored result tables are
# absent.  Any failure stops with a non-zero exit status.

suppressMessages({
  library(ggplot2)
  library(data.table)
})

root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))), "..", ".."))
source(file.path(root, "scripts", "R", "trap_plot_helpers.R"))

check <- function(ok, what) {
  if (!isTRUE(ok)) stop("FAILED: ", what, call. = FALSE)
  cat("ok:", what, "\n")
}

out <- tempfile("trap_plot_helpers_")
dir.create(out)

# trap_save_fig: writes both formats, honours defaults, and lets a call override them.
p <- ggplot(data.frame(x = 1:3, y = 1:3), aes(x, y)) + geom_point()
save_fig <- trap_save_fig(out, default_width = 6, default_height = 4)
save_fig(p, "default")
save_fig(p, "wide", width = 12, height = 4)
check(all(file.exists(file.path(out, c("default.png", "default.pdf", "wide.png", "wide.pdf")))),
      "save_fig writes a PNG and a PDF per stem")
check(file.size(file.path(out, "wide.png")) != file.size(file.path(out, "default.png")),
      "a call-site width overrides the default")
check(identical(save_fig(p, "again"), p), "save_fig returns the plot")

# Notebook 5.03 passes no defaults: width and height must then be required.
required <- trap_save_fig(out)
required(p, "required", width = 5, height = 5)
check(inherits(try(required(p, "missing_size"), silent = TRUE), "try-error"),
      "without defaults, width and height are required")

# Colour helpers.
check(identical(lighten("#000000", 0), "#000000"), "lighten(.., 0) leaves the colour alone")
check(identical(toupper(lighten("#123456", 1)), "#FFFFFF"), "lighten(.., 1) gives white")
check(identical(comma_from_5_digits(c(999, 12345)), c("999", "12,345")),
      "thousands separator starts at five digits")

# series_palette reads class_palette from the calling environment.
class_palette <- c(L1HS = "#d62728", L1PA = "#1f77b4")
pal <- series_palette(c("A", "B"), c("L1HS", "L1PA"))
check(identical(names(pal), c("A (L1HS)", "B (L1HS)", "A (L1PA)", "B (L1PA)")),
      "series_palette names every Method (Class) series")
check(identical(pal[["A (L1HS)"]], "#D62728"), "the first method keeps the class colour")
check(!identical(pal[["B (L1HS)"]], pal[["A (L1HS)"]]), "later methods are lightened")

# plot_by_chr: end to end, plus its two guard paths.
class_levels <- c("L1HS", "L1PA")
chrom_levels <- c("chr1", "chr2")
dt <- CJ(figure = "fig", method = c("A", "B"), class = class_levels, chrom = chrom_levels)
dt[, value := seq_len(.N)]
res <- plot_by_chr(dt, "fig", c("A", "B"), width = 6, height = 4)
check(inherits(res, "ggplot"), "plot_by_chr returns a ggplot")
check(file.exists(file.path(out, "fig.png")), "plot_by_chr saves through save_fig")
check(is.null(suppressWarnings(plot_by_chr(dt, "absent", c("A", "B")))),
      "plot_by_chr skips a figure with no rows")
w <- tryCatch(plot_by_chr(dt, "fig", c("A", "B", "C"), width = 6, height = 4),
              warning = function(w) conditionMessage(w))
check(is.character(w) && grepl("missing method", w), "plot_by_chr warns about a missing method")

cat("all helper checks passed\n")
