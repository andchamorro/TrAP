#!/usr/bin/env Rscript
# TrAP workflow overview — the two-track (build / apply) pipeline diagram.
# Renders §3.1 of .trap/summaries/refactor-update_core_improve_repro.md as a figure.
# Structure adapted from Sertorius/proposal/desk/fig/workflow_overview.R.
#
# NOTE: there is deliberately NO MLM pre-training node — the ALBERT classifier is trained
# directly on the supervised task (MLM was dropped; the hashed k-mer vocab is unlearnable
# under a masking objective). Run from the repo root:  Rscript scripts/R/workflow_overview.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggforce)
  library(dplyr)
  library(tibble)
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
nodes <- tibble::tribble(
  ~id, ~cx,  ~cy, ~w,   ~h,  ~label,                                        ~fill,       ~stage,
  # BUILD track (y = 5.0)
  1,   2.2,  5.0, 1.25, 0.50, "GENCODE v48 transcripts\n+ RepeatMasker L1", COL_INPUT,  "stage 00",
  2,   5.0,  5.0, 1.25, 0.50, "Canonical k-mer tokenizer\n(k = 17, 1 token / k-mer)", COL_TOK, "stage 10",
  3,   7.8,  5.0, 1.20, 0.50, "Synthetic reads\n(ART → STAR → label)", COL_DATA, "stage 20",
  4,  10.6,  5.0, 1.30, 0.50, "L1HS / L1PA / NEGATIVE\nclassifier (ALBERT, direct)", COL_MODEL, "stage 40",
  # APPLY track (y = 2.0)
  5,   3.0,  2.0, 1.10, 0.50, "Sample FASTQ",                              COL_INPUT,  "",
  6,   6.8,  2.0, 1.10, 0.50, "k-mer tokenize",                            COL_TOK,    "",
  7,  10.6,  2.0, 1.30, 0.50, "Streaming GPU inference\n→ per-read scores", COL_INFER, "stage 50",
  8,  13.4,  2.0, 1.15, 0.50, "Filter NEGATIVE\n(P < tau)",           COL_FILTER, "",
  9,  15.9,  2.0, 1.25, 0.50, "Salmon / EM →\nLINE-1 abundance",      COL_OUT,    "",
  # supporting analysis (feeds the k choice)
  10,  5.0,  6.25, 1.40, 0.40, "k-mer length selection\n(entropy / redundancy · stage 05)", COL_AUX, ""
) %>%
  mutate(xmin = cx - w, xmax = cx + w, ymin = cy - h, ymax = cy + h)

nodes_rect <- nodes %>%
  rowwise() %>%
  do(data.frame(id = .$id, fill = .$fill,
                x = c(.$xmin, .$xmax, .$xmin, .$xmax),
                y = c(.$ymin, .$ymin, .$ymax, .$ymax))) %>%
  ungroup()

# ── Track backgrounds ───────────────────────────────────────────────────────
track_bg <- tibble::tribble(
  ~id,      ~xmin, ~xmax, ~ymin, ~ymax, ~fill,
  "build",   0.55, 12.05, 4.30,  5.75,  COL_TRACK_BUILD,
  "apply",   1.65, 17.35, 1.30,  2.75,  COL_TRACK_APPLY
)
track_rect <- track_bg %>%
  rowwise() %>%
  do(data.frame(id = .$id, fill = .$fill,
                x = c(.$xmin, .$xmax, .$xmin, .$xmax),
                y = c(.$ymin, .$ymin, .$ymax, .$ymax))) %>%
  ungroup()

# ── Solid arrows (main flow) ────────────────────────────────────────────────
arws <- tibble::tribble(
  ~x,    ~y,   ~xend, ~yend,
  # BUILD: references -> tokenizer -> synthetic -> classifier
  3.45,  5.0,  3.75,  5.0,
  6.25,  5.0,  6.60,  5.0,
  9.00,  5.0,  9.30,  5.0,
  # APPLY: FASTQ -> tokenize -> inference -> filter -> abundance
  4.10,  2.0,  5.70,  2.0,
  7.90,  2.0,  9.30,  2.0,
  11.90, 2.0, 12.25,  2.0,
  14.55, 2.0, 14.65,  2.0
)

# hand-off: classifier -> streaming inference (the trained model)
handoff <- tibble::tibble(x = 10.6, y = 4.50, xend = 10.6, yend = 2.50)

# ── Plot ────────────────────────────────────────────────────────────────────
p <- ggplot() +
  geom_mark_rect(data = track_rect, aes(x = x, y = y, group = id, fill = fill),
                 alpha = 0.6, color = NA, radius = unit(12, "pt"), expand = unit(0, "mm")) +
  annotate("text", x = 0.7, y = 5.60, hjust = 0, size = 5.4, fontface = "italic",
           color = "#336699", label = "BUILD — offline, run once (SLURM 00→40)") +
  annotate("text", x = 1.8, y = 2.60, hjust = 0, size = 5.4, fontface = "italic",
           color = "#2E7D52", label = "APPLY — per sample (SLURM 50)") +
  # dashed arrow: k-selection analysis -> tokenizer
  geom_segment(data = tibble::tibble(x = 5.0, y = 5.83, xend = 5.0, yend = 5.52),
               aes(x = x, y = y, xend = xend, yend = yend), linetype = "dashed",
               linewidth = 0.6, color = "#8A8A8A",
               arrow = arrow(length = unit(0.18, "cm"), type = "closed")) +
  # main-flow arrows
  geom_segment(data = arws, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.24, "cm"), type = "closed"),
               linewidth = 0.9, color = "#5A5A5A", lineend = "round") +
  # trained-model hand-off (thicker, coloured)
  geom_segment(data = handoff, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.26, "cm"), type = "closed"),
               linewidth = 1.1, color = "#7A4FB6", lineend = "round") +
  annotate("text", x = 10.85, y = 3.5, hjust = 0, size = 4.5, fontface = "italic",
           color = "#7A4FB6", label = "trained model") +
  # node boxes + text
  geom_mark_rect(data = nodes_rect, aes(x = x, y = y, group = id, fill = fill),
                 color = "grey35", linewidth = 0.85, radius = unit(9, "pt"),
                 expand = unit(0, "mm")) +
  scale_fill_identity() +
  geom_text(data = nodes, aes(x = cx, y = cy, label = label),
            size = 4.7, lineheight = 1.05, color = "#202020") +
  # stage tags under each box
  geom_text(data = dplyr::filter(nodes, stage != ""),
            aes(x = cx, y = ymin - 0.24, label = stage),
            size = 3.6, color = "#7A7A7A", fontface = "italic") +
  # title + subtitle + reproducibility band
  annotate("text", x = 8.9, y = 7.55, size = 7.2, fontface = "bold", color = "#111111",
           label = "TrAP Workflow Overview") +
  annotate("text", x = 8.9, y = 7.15, size = 4.8, color = "#444444",
           label = "Build a k-mer classifier once, then apply it per sample to quantify LINE-1 — no MLM pre-training") +
  annotate("text", x = 8.9, y = 0.55, size = 4.3, color = "#666666", fontface = "italic",
           label = "reproducibility band: manifest.json + seed + SHA256 recorded at every stage") +
  coord_cartesian(xlim = c(0, 17.8), ylim = c(0.2, 7.85), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(16, 16, 16, 16))

for (ext in c("png", "pdf")) {
  ggsave(file.path(out_dir, paste0("workflow_overview.", ext)),
         plot = p, width = 20, height = 9, dpi = 300, bg = "white")
}
message("wrote ", file.path(out_dir, "workflow_overview.{png,pdf}"))
