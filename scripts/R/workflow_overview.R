#!/usr/bin/env Rscript
# TrAP workflow overview — the two-track (build / apply) pipeline diagram.
# Renders §3.1 of .trap/summaries/refactor-update_core_improve_repro.md as a figure.
# Structure adapted from Sertorius/proposal/desk/fig/workflow_overview.R.
#
# Dependencies: ggplot2 + grid only (both already in the conda env; grid is base R).
# Rounded boxes use grid::roundrectGrob so no ggforce is required — ggforce's compiled deps
# (tweenr/polyclip/systemfonts) do not build from source in the conda toolchain.
#
# NOTE: there is deliberately NO MLM pre-training node — the ALBERT classifier is trained
# directly on the supervised task (MLM was dropped; the hashed k-mer vocab is unlearnable
# under a masking objective). Run from the repo root:  Rscript scripts/R/workflow_overview.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

out_dir <- file.path("reports", "figures")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ── Colours (soft palette; one hue per role) ────────────────────────────────
COL_INPUT  <- "#FADADD"  # references / sample input (pink)
COL_TOK    <- "#CCE5FF"  # k-mer tokenizer (blue)
COL_DATA   <- "#FFF2B8"  # synthetic dataset (yellow)
COL_MODEL  <- "#E4D0F4"  # ALBERT classifier (purple)
COL_INFER  <- "#C8EDD8"  # streaming inference (green)
COL_FILTER <- "#FFDAB9"  # NEGATIVE filter (peach)
COL_OUT    <- "#EBEBEB"  # abundance output (grey)
COL_AUX    <- "#F3F3F3"  # supporting analysis (light grey)
COL_TRACK_BUILD <- "#EEF5FF"
COL_TRACK_APPLY <- "#EEFBF3"

# ── Nodes (cx/cy = centre; w/h = half-width/half-height) ─────────────────────
nodes <- data.frame(
  id    = 1:10,
  cx    = c(2.2, 5.0, 7.8, 10.6,  3.0, 6.8, 10.6, 13.4, 15.9,  5.0),
  cy    = c(5.0, 5.0, 5.0, 5.0,   2.0, 2.0, 2.0,  2.0,  2.0,   6.25),
  w     = c(1.25, 1.25, 1.20, 1.30,  1.10, 1.10, 1.30, 1.15, 1.25,  1.40),
  h     = c(0.50, 0.50, 0.50, 0.50,  0.50, 0.50, 0.50, 0.50, 0.50,  0.40),
  label = c(
    "GENCODE v48 transcripts\n+ RepeatMasker L1",
    "Canonical k-mer tokenizer\n(k = 17, 1 token / k-mer)",
    "Synthetic reads\n(ART → STAR → label)",
    "L1HS / L1PA / NEGATIVE\nclassifier (ALBERT, direct)",
    "Sample FASTQ",
    "k-mer tokenize",
    "Streaming GPU inference\n→ per-read scores",
    "Filter NEGATIVE\n(P < tau)",
    "Salmon / EM →\nLINE-1 abundance",
    "k-mer length selection\n(entropy / redundancy · stage 05)"
  ),
  fill  = c(COL_INPUT, COL_TOK, COL_DATA, COL_MODEL, COL_INPUT,
            COL_TOK, COL_INFER, COL_FILTER, COL_OUT, COL_AUX),
  stage = c("stage 00", "stage 10", "stage 20", "stage 40", "", "", "stage 50", "", "", ""),
  stringsAsFactors = FALSE
)
nodes$xmin <- nodes$cx - nodes$w
nodes$xmax <- nodes$cx + nodes$w
nodes$ymin <- nodes$cy - nodes$h
nodes$ymax <- nodes$cy + nodes$h

# Rounded box layers via base grid (one annotation_custom per rectangle) — no ggforce.
box_layers <- function(df, border = "grey35", lwd = 1.7, alpha = 1, radius = unit(8, "pt")) {
  lapply(seq_len(nrow(df)), function(i) {
    r <- df[i, ]
    annotation_custom(
      grob = roundrectGrob(r = radius,
                           gp = gpar(fill = r$fill, col = border, lwd = lwd, alpha = alpha)),
      xmin = r$xmin, xmax = r$xmax, ymin = r$ymin, ymax = r$ymax
    )
  })
}

# ── Track backgrounds ───────────────────────────────────────────────────────
track_bg <- data.frame(
  id   = c("build", "apply"),
  xmin = c(0.55, 1.65), xmax = c(12.05, 17.35),
  # bottom extended to enclose the stage-NN labels (at node ymin - 0.24); top a touch higher.
  ymin = c(3.98, 0.98), ymax = c(5.82, 2.82),
  fill = c(COL_TRACK_BUILD, COL_TRACK_APPLY), stringsAsFactors = FALSE
)

# ── Arrows ──────────────────────────────────────────────────────────────────
arws <- data.frame(
  x    = c(3.45, 6.25, 9.00,  4.10, 7.90, 11.90, 14.55),
  y    = c(5.0,  5.0,  5.0,   2.0,  2.0,  2.0,   2.0),
  xend = c(3.75, 6.60, 9.30,  5.70, 9.30, 12.25, 14.65),
  yend = c(5.0,  5.0,  5.0,   2.0,  2.0,  2.0,   2.0)
)
handoff  <- data.frame(x = 10.6, y = 4.50, xend = 10.6, yend = 2.50)
kselect  <- data.frame(x = 5.0,  y = 5.83, xend = 5.0,  yend = 5.52)

# ── Plot ────────────────────────────────────────────────────────────────────
p <- ggplot() +
  box_layers(track_bg, border = NA, lwd = 0, alpha = 0.6, radius = unit(12, "pt")) +
  annotate("text", x = 0.7, y = 5.60, hjust = 0, size = 6.75, fontface = "italic",
           color = "#336699", label = "Core") +
  annotate("text", x = 1.8, y = 2.60, hjust = 0, size = 6.75, fontface = "italic",
           color = "#2E7D52", label = "Per sample") +
  geom_segment(data = kselect, aes(x = x, y = y, xend = xend, yend = yend),
               linetype = "dashed", linewidth = 0.6, color = "#8A8A8A",
               arrow = arrow(length = unit(0.18, "cm"), type = "closed")) +
  geom_segment(data = arws, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.24, "cm"), type = "closed"),
               linewidth = 0.9, color = "#5A5A5A", lineend = "round") +
  geom_segment(data = handoff, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.26, "cm"), type = "closed"),
               linewidth = 1.1, color = "#7A4FB6", lineend = "round") +
  annotate("text", x = 10.85, y = 3.5, hjust = 0, size = 5.625, fontface = "italic",
           color = "#7A4FB6", label = "trained model") +
  box_layers(nodes, border = "grey35", lwd = 1.7, radius = unit(8, "pt")) +
  geom_text(data = nodes, aes(x = cx, y = cy, label = label),
            size = 5.875, lineheight = 1.05, color = "#202020") +
  geom_text(data = nodes[nodes$stage != "", ],
            aes(x = cx, y = ymin - 0.24, label = stage),
            size = 4.5, color = "#7A7A7A", fontface = "italic") +
  annotate("text", x = 8.9, y = 7.55, size = 9.0, fontface = "bold", color = "#111111",
           label = "TrAP Workflow Overview") +
  annotate("text", x = 8.9, y = 7.15, size = 6.0, color = "#444444",
           label = "Build a k-mer classifier once, then apply it per sample to quantify LINE-1 — no MLM pre-training") +
  annotate("text", x = 8.9, y = 0.55, size = 5.375, color = "#666666", fontface = "italic",
           label = "reproducibility band: manifest.json + seed + SHA256 recorded at every stage") +
  coord_cartesian(xlim = c(0, 17.8), ylim = c(0.2, 7.85), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(16, 16, 16, 16))

# r-cairo (from environment.yml) renders the Unicode arrows cleanly.
for (ext in c("png", "pdf")) {
  dev <- if (ext == "png") "cairo" else cairo_pdf
  args <- list(filename = file.path(out_dir, paste0("workflow_overview.", ext)),
               plot = p, width = 24, height = 9.5, dpi = 300, bg = "white")
  if (ext == "png") args$type <- "cairo"
  do.call(ggsave, args)
}
message("wrote ", file.path(out_dir, "workflow_overview.{png,pdf}"))
