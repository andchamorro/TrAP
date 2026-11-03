"""Run the R smoke test for the shared notebook plotting helpers.

Skipped when R or its plotting packages are not installed.  Set ``TRAP_REQUIRE_R=1`` (CI
does) to turn that skip into a failure, so a broken R toolchain cannot pass silently.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "R" / "test_plot_helpers.R"
PACKAGES = ("ggplot2", "data.table")


def _r_ready() -> str:
    """Return an empty string when R and the packages are usable, else the reason not."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return "Rscript not on PATH"
    probe = subprocess.run(
        [rscript, "--vanilla", "-e", "; ".join(f"library({p})" for p in PACKAGES)],
        capture_output=True,
        text=True,
    )
    return "" if probe.returncode == 0 else f"missing R packages: {probe.stderr.strip()[-200:]}"


@pytest.mark.integration
def test_notebook_plot_helpers():
    reason = _r_ready()
    if reason:
        if os.environ.get("TRAP_REQUIRE_R"):
            pytest.fail(reason)
        pytest.skip(reason)
    run = subprocess.run(
        ["Rscript", "--vanilla", str(SCRIPT)], capture_output=True, text=True, cwd=ROOT
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all helper checks passed" in run.stdout
