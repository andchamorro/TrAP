"""Tracked notebooks carry source only.

Stored outputs bloat diffs and go stale as the plotting stack changes, so figures are
regenerated from the notebooks rather than read back out of them.  Clear outputs with
``jupyter nbconvert --clear-output --inplace notebooks/<name>.ipynb`` before committing.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_stored_outputs(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    dirty = [
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count") is not None)
    ]
    assert not dirty, f"{path.name}: cells with outputs or execution counts: {dirty}"
