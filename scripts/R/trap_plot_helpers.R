###############################################################################
# scripts/R/trap_plot_helpers.R
#
# Plotting helpers shared by the TrAP figure notebooks (notebooks/5.0x-*).
#
# Each notebook used to carry its own copy of these, which meant a fix had to be
# applied seven times and a drift between copies was invisible.  Source this file
# instead, immediately after the `figures_dir` / palette definitions:
#
#     source(file.path("..", "scripts/R/trap_plot_helpers.R"))
#     save_fig <- trap_save_fig(figures_dir, default_width = 14, default_height = 8)
#
# `source()` evaluates into the caller's environment, so these functions resolve
# the notebook's own objects at call time and behave exactly as the inlined
# copies did.
#
# Calling contract: `series_palette()` and `plot_by_chr()` read `class_palette`,
# `class_levels` and `chrom_levels` from the calling environment, and
# `plot_by_chr()` also calls `save_fig`.  Define all four before calling them.
#
# Requires: ggplot2, data.table, scales, grDevices.
###############################################################################


#' Build a figure-saving function bound to one notebook's output directory.
#'
#' Every notebook writes a PNG and a PDF of each figure at 300 dpi on a white
#' background, and differs only in its default canvas size.  The factory binds
#' the directory and that notebook's defaults, so the body lives in one place
#' while each notebook keeps the dimensions its figures were composed for.
#'
#' Omit `width`/`height` to make them required at each call site, which is what
#' notebook 5.03 does because its panels have no single natural size.
#'
#' Args:
#'   figures_dir: Directory to write into; must already exist.
#'   default_width, default_height: Default canvas size in inches. Omitting them
#'     makes width/height required at the call site.
#'
#' Returns:
#'   A function(plot, stem, width, height) that writes `<stem>.png` and
#'   `<stem>.pdf` and returns `plot` invisibly.
trap_save_fig <- function(figures_dir, default_width, default_height) {
  force(figures_dir)
  function(plot, stem, width = default_width, height = default_height) {
    for (ext in c("png", "pdf")) {
      f <- file.path(figures_dir, paste0(stem, ".", ext))
      suppressMessages(ggsave(f, plot, width = width, height = height, dpi = 300, bg = "white"))
    }
    invisible(plot)
  }
}


#' Move a colour toward white by `amount`.
#'
#' Used to derive one shade per method inside a class colour, so that class
#' stays readable as the hue while methods remain distinguishable within it.
#'
#' Args:
#'   hex: A colour accepted by `col2rgb`.
#'   amount: 0 = the base colour, 1 = white.
#'
#' Returns:
#'   The lightened colour as a hex string.
lighten <- function(hex, amount) {
  v <- col2rgb(hex) / 255
  v <- v + (1 - v) * amount
  grDevices::rgb(v[1], v[2], v[3])
}


#' Colour every "Method (Class)" series.
#'
#' Class sets the hue and method sets the lightness within it, so a reader
#' compares classes by colour and methods by shade rather than holding a
#' fourteen-entry legend in mind.
#'
#' Reads `class_palette` from the calling environment.
#'
#' Args:
#'   methods, classes: Character vectors, in the order they should be shaded.
#'
#' Returns:
#'   A named character vector mapping "Method (Class)" to a colour.
series_palette <- function(methods, classes) {
  amt <- if (length(methods) <= 1) 0 else seq(0, 0.62, length.out = length(methods))
  pal <- character(0)
  for (cl in classes) for (i in seq_along(methods))
    pal[sprintf("%s (%s)", methods[i], cl)] <- lighten(class_palette[[cl]], amt[i])
  pal
}


#' Thousands separators only once a number reaches five digits.
#'
#' Below 10,000 a separator adds width without aiding comparison, and on a
#' crowded axis that width costs a tick.
comma_from_5_digits <- function(x) {
  ifelse(abs(x) >= 1e4, scales::label_comma()(x), scales::label_number(big.mark = "")(x))
}


#' Grouped per-chromosome bar chart of aggregated L1 reads, saved to disk.
#'
#' Bars are dodged class-major so the classes of one method sit together, while
#' the legend is ordered method-major, which is the order a reader compares
#' methods in.  Series with no rows are dropped rather than drawn empty, and a
#' missing method warns instead of silently vanishing from the comparison.
#'
#' Reads `class_levels`, `chrom_levels`, `class_palette` and `save_fig` from the
#' calling environment.
#'
#' Args:
#'   dt: data.table with columns figure, method, class, chrom, value.
#'   stem: Figure stem; selects `dt[figure == stem]` and names the output file.
#'   method_levels: Methods to plot, in dodge order.
#'   legend: "inside" for an overlaid legend, otherwise a ggplot2 position.
#'   legend_size: Legend text size in points.
#'   width, height: Canvas size in inches.
#'
#' Returns:
#'   The ggplot object, or NULL invisibly when `stem` matched no rows.
plot_by_chr <- function(dt, stem, method_levels, legend = "inside",
                        legend_size = 18.75, width = 12, height = 7) {
  d <- dt[figure == stem]
  if (nrow(d) == 0) { warning(sprintf("%s: no rows — skipped", stem)); return(invisible(NULL)) }
  present <- intersect(method_levels, unique(d$method))
  if (length(present) < length(method_levels))
    warning(sprintf("%s: missing method(s): %s", stem,
                    paste(setdiff(method_levels, present), collapse = ", ")))
  classes <- intersect(class_levels, unique(d$class))

  d <- copy(d)
  d <- d[method %in% present & class %in% classes]
  d[, chrom := factor(chrom, levels = chrom_levels)]; d <- d[!is.na(chrom)]
  d[, series_chr := sprintf("%s (%s)", method, class)]

  # class-major dodge order; method-major legend order; keep only combos with data.
  have         <- unique(d$series_chr)
  dodge_levels <- as.vector(outer(present, classes, function(m, cl) sprintf("%s (%s)", m, cl)))
  legend_lvls  <- as.vector(outer(classes, present, function(cl, m) sprintf("%s (%s)", m, cl)))
  dodge_levels <- intersect(dodge_levels, have); legend_lvls <- intersect(legend_lvls, have)
  d[, series := factor(series_chr, levels = dodge_levels)]
  pal <- series_palette(present, classes)

  p <- ggplot(d, aes(chrom, value, fill = series, group = series)) +
    geom_col(position = position_dodge2(preserve = "single", padding = 0.1),
             colour = "grey30", linewidth = 0.1) +
    scale_fill_manual(values = pal, breaks = legend_lvls, name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = NULL, y = "Aggregated L1 reads") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
          legend.key.size = unit(0.9, "lines"),
          legend.text = element_text(size = legend_size))
  if (identical(legend, "inside")) {
    p <- p + guides(fill = guide_legend(ncol = length(present))) +
      theme(legend.position = c(0.01, 0.99), legend.justification = c(0, 1),
            legend.background = element_rect(fill = alpha("white", 0.75), colour = NA),
            legend.margin = margin(2, 4, 2, 4))
  } else {                                    # outside (legacy 4.02): single column, e.g. "right"
    p <- p + guides(fill = guide_legend(ncol = 1)) + theme(legend.position = legend)
  }
  save_fig(p, stem, width, height)
  p
}
