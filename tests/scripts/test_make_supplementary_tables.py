"""Golden-file test for ``scripts/make_supplementary_tables.py``.

The script is imperative: importing it reads ``results/`` and writes the tables.  It is
run here against a fixture ``results/`` tree whose numbers are fabricated (see
``tests/fixtures/supplementary/README.md``), so the test pins the formatting, the
caption numbering and the cross-table joins without touching real results.
"""

import importlib
import os
from pathlib import Path
import re
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_supplementary_tables.py"
FIXTURES = ROOT / "tests" / "fixtures" / "supplementary"
GOLDEN = FIXTURES / "expected_supplementary.md"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("supp") / "supplementary.md"
    # ``import trap.config.config as cfg`` resolves to an attribute that shadows the module.
    cfg = importlib.import_module("trap.config.config")

    saved_root, saved_argv = cfg.PROJ_ROOT, sys.argv
    cfg.PROJ_ROOT = FIXTURES
    sys.argv = [str(SCRIPT), str(out)]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        cfg.PROJ_ROOT, sys.argv = saved_root, saved_argv
    return out.read_text(encoding="utf-8")


@pytest.mark.integration
class TestSupplementaryTables:
    def test_matches_golden(self, generated):
        if os.environ.get("UPDATE_GOLDEN"):
            GOLDEN.write_text(generated, encoding="utf-8")
        assert generated == GOLDEN.read_text(encoding="utf-8")

    def test_eleven_tables_numbered_in_order(self, generated):
        numbers = [int(n) for n in re.findall(r"^: \*\*Table S(\d+)\.\*\*", generated, re.M)]
        assert numbers == list(range(1, 12))

    def test_no_missing_values_leak_into_cells(self, generated):
        assert not re.search(r"\bnan\b|\bNone\b", generated)

    def test_every_table_row_has_a_consistent_column_count(self, generated):
        block = []
        for line in generated.splitlines() + [""]:
            if line.startswith("|"):
                block.append(line.count("|") - line.count("\\|"))
            elif block:
                assert len(set(block)) == 1, f"ragged table: {block}"
                block = []

    def test_small_p_values_use_scientific_notation(self, generated):
        assert re.search(r"\$\d\.\d\\times10\^\{-\d+\}\$", generated)

    def test_front_matter_is_valid_pandoc_yaml(self, generated):
        assert generated.startswith("---\ntitle:")
        assert generated.split("---")[1].count("name:") == 2
