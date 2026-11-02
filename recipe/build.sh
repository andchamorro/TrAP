#!/usr/bin/env bash

set -euo pipefail

# TrAP conda package build script
# This script is called during the conda build process

# Install the package using pip
# --no-deps: Don't install dependencies (conda handles this)
# --no-build-isolation: Use the conda build environment
# -vvv: Verbose output for debugging
"${PYTHON}" -m pip install . \
    --no-deps \
    --no-build-isolation \
    --ignore-installed \
    --no-cache-dir \
    -vvv

# Imports are verified by the recipe's test section, which runs with the run
# dependencies installed; the build (host) environment only has pip and flit-core.
