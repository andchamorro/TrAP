#!/usr/bin/env Rscript
###############################################################################
# envs/r-packages.R
# Install CRAN R packages into the active conda environment and register the
# IRkernel so the base-env Jupyter sees "R (trap)".
#
# Policy:
#   * Build from source inside the conda env so packages link against conda libs.
#   * Install into the conda env's R library (R.home("library")), not ~/Library/R.
#   * Skip packages that are already installed in the conda env library.
#
# Usage (after `conda env create -f envs/environment.yml`):
#   conda activate trap
#   Rscript --vanilla envs/r-packages.R
#
# Or via setup_conda_env.sh (runs this automatically).
###############################################################################

# If this repo is opened as an renv project, `.Rprofile` may activate renv and
# shim installation helpers. Use base installers explicitly.
Sys.setenv(RENV_CONFIG_AUTOLOADER_ENABLED = "FALSE")

conda_lib <- R.home("library")
.libPaths(conda_lib)

options(
  repos   = c(CRAN = "https://cloud.r-project.org"),
  pkgType = "source",
  Ncpus   = max(1L, parallel::detectCores() - 1L)
)

install_missing <- function(pkgs) {
  installed <- rownames(installed.packages(lib.loc = conda_lib))
  needed    <- pkgs[!pkgs %in% installed]
  if (length(needed) == 0) {
    message(sprintf("  All %d packages already installed.", length(pkgs)))
    return(invisible(NULL))
  }
  message(sprintf("  Installing (%d): %s", length(needed), paste(needed, collapse = ", ")))
  utils::install.packages(needed, lib = conda_lib,
                          repos = c(CRAN = "https://cloud.r-project.org"),
                          type = "source")
}

# Packages used by TrAP R notebooks (notebooks/5.01-*).
cran_packages <- c(
  "data.table",
  "ggplot2",
  "jsonlite",
  "reshape2"
)

cat("\n== Installing CRAN packages ==\n")
install_missing(cran_packages)

# Verify
cat("\n== Import verification ==\n")
installed_now <- rownames(installed.packages(lib.loc = conda_lib))
missing_pkgs  <- cran_packages[!cran_packages %in% installed_now]
if (length(missing_pkgs) > 0) {
  warning("Some packages failed to install: ", paste(missing_pkgs, collapse = ", "))
  quit(status = 1)
}
message("All required packages installed successfully!")

# Register the IRkernel (idempotent) so the base-env Jupyter sees this env.
# IRkernel is installed via conda (r-irkernel); this just writes the kernel spec.
if ("IRkernel" %in% rownames(installed.packages(lib.loc = conda_lib))) {
  tryCatch(
    IRkernel::installspec(
      name        = "ir-trap",
      displayname = "R (trap)",
      user        = TRUE
    ),
    error = function(e) message("IRkernel::installspec skipped: ", conditionMessage(e))
  )
} else {
  message("IRkernel not found; skipping kernel registration.")
}
