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

# Verify installation
echo "Verifying trap installation..."
"${PYTHON}" -c "import trap; print(f'TrAP version: {trap.__version__ if hasattr(trap, \"__version__\") else \"unknown\"}')"
"${PYTHON}" -c "import trap.config.config; print(f'Project root: {trap.config.config.PROJ_ROOT}')"

echo "Build completed successfully!"
