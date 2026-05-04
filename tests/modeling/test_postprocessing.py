"""Tests for trap.modeling.postprocessing — sanitize_paths + filter_ids."""

import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from trap.modeling.postprocessing import app, sanitize_paths


class TestSanitizePaths:
    """Tests for the sanitize_paths helper."""

    @pytest.mark.unit
    def test_string_converted_to_path(self):
        paths = sanitize_paths("/tmp/test")
        assert paths == [Path("/tmp/test")]

    @pytest.mark.unit
    def test_path_object_unchanged(self):
        p = Path("/tmp/test")
        paths = sanitize_paths(p)
        assert paths == [p]

    @pytest.mark.unit
    def test_none_preserved(self):
        paths = sanitize_paths(None)
        assert paths == [None]

    @pytest.mark.unit
    def test_multiple_arguments(self):
        paths = sanitize_paths("/a", Path("/b"), None, "/c")
        assert len(paths) == 4
        assert paths[0] == Path("/a")
        assert paths[1] == Path("/b")
        assert paths[2] is None
        assert paths[3] == Path("/c")

    @pytest.mark.unit
    def test_empty_call(self):
        paths = sanitize_paths()
        assert paths == []


# ---------------------------------------------------------------------------
# TestFilterIds
# ---------------------------------------------------------------------------

_runner = CliRunner()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    """Write a minimal Parquet score file to *path*."""
    table = pa.table(
        {
            "id": pa.array([r["id"] for r in rows], type=pa.string()),
            "NEGATIVE": pa.array([r["NEGATIVE"] for r in rows], type=pa.float32()),
            "L1HS": pa.array([r["L1HS"] for r in rows], type=pa.float32()),
            "L1PA": pa.array([r["L1PA"] for r in rows], type=pa.float32()),
        }
    )
    pq.write_table(table, str(path))


class TestFilterIds:
    """Regression tests for the filter_ids command (Bug 1 guard)."""

    # --- Parquet path --------------------------------------------------------

    @pytest.mark.integration
    def test_parquet_returns_low_negative_ids(self, tmp_path):
        rows = [
            {"id": "read1", "NEGATIVE": 0.2, "L1HS": 0.5, "L1PA": 0.3},
            {"id": "read2", "NEGATIVE": 0.8, "L1HS": 0.1, "L1PA": 0.1},
            {"id": "read3", "NEGATIVE": 0.4, "L1HS": 0.3, "L1PA": 0.3},
        ]
        _write_parquet(tmp_path / "scores_0_0.parquet", rows)
        id_list = tmp_path / "ids.txt"
        id_list.write_text("read1\nread2\nread3\n")
        out_file = tmp_path / "filtered.txt"
        result = _runner.invoke(
            app,
            [
                "--output-path", str(tmp_path),
                "--id-list", str(id_list),
                "--threshold", "0.5",
                "--output-filtered-ids", str(out_file),
            ],
        )
        assert result.exit_code == 0, result.output
        kept = out_file.read_text().splitlines()
        assert "read1" in kept
        assert "read3" in kept
        assert "read2" not in kept

    @pytest.mark.integration
    def test_parquet_threshold_boundary(self, tmp_path):
        """Score exactly equal to threshold is NOT kept (< not <=)."""
        rows = [
            {"id": "exact", "NEGATIVE": 0.5, "L1HS": 0.25, "L1PA": 0.25},
            {"id": "below", "NEGATIVE": 0.4999, "L1HS": 0.25, "L1PA": 0.25},
        ]
        _write_parquet(tmp_path / "scores_0_0.parquet", rows)
        id_list = tmp_path / "ids.txt"
        id_list.write_text("exact\nbelow\n")
        out_file = tmp_path / "filtered.txt"
        result = _runner.invoke(
            app,
            [
                "--output-path", str(tmp_path),
                "--id-list", str(id_list),
                "--output-filtered-ids", str(out_file),
            ],
        )
        assert result.exit_code == 0, result.output
        kept = out_file.read_text().splitlines()
        assert "below" in kept
        assert "exact" not in kept

    # --- Pickle (legacy) path ------------------------------------------------

    @pytest.mark.integration
    def test_pickle_returns_low_negative_ids(self, tmp_path):
        scores = [
            [{"label": "NEGATIVE", "score": 0.2}, {"label": "L1HS", "score": 0.8}],
            [{"label": "NEGATIVE", "score": 0.9}, {"label": "L1HS", "score": 0.1}],
            [{"label": "NEGATIVE", "score": 0.3}, {"label": "L1HS", "score": 0.7}],
        ]
        with open(tmp_path / "class_scores.pkl", "wb") as f:
            pickle.dump(scores, f)
        id_list = tmp_path / "ids.txt"
        id_list.write_text("readA\nreadB\nreadC\n")
        out_file = tmp_path / "filtered.txt"
        result = _runner.invoke(
            app,
            [
                "--output-path", str(tmp_path),
                "--id-list", str(id_list),
                "--threshold", "0.5",
                "--output-filtered-ids", str(out_file),
            ],
        )
        assert result.exit_code == 0, result.output
        kept = out_file.read_text().splitlines()
        assert "readA" in kept
        assert "readC" in kept
        assert "readB" not in kept

    @pytest.mark.integration
    def test_pickle_mismatched_lengths_raise(self, tmp_path):
        """ID list and score file length mismatch must raise ValueError."""
        scores = [
            [{"label": "NEGATIVE", "score": 0.2}],
            [{"label": "NEGATIVE", "score": 0.9}],
        ]
        with open(tmp_path / "class_scores.pkl", "wb") as f:
            pickle.dump(scores, f)
        id_list = tmp_path / "ids.txt"
        # 3 IDs but only 2 score rows
        id_list.write_text("r1\nr2\nr3\n")
        result = _runner.invoke(
            app,
            [
                "--output-path", str(tmp_path),
                "--id-list", str(id_list),
                "--output-filtered-ids", str(tmp_path / "out.txt"),
            ],
        )
        assert result.exit_code != 0 or isinstance(result.exception, ValueError)

    # --- Guard: no fastq or id_list -----------------------------------------

    @pytest.mark.unit
    def test_no_input_exits_nonzero(self, tmp_path):
        result = _runner.invoke(app, ["--output-path", str(tmp_path)])
        assert result.exit_code != 0
