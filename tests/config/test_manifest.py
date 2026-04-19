"""Tests for trap.config.manifest (DR-7)."""
import json
from pathlib import Path

import pytest

from trap.config.manifest import sha256_file, write


@pytest.mark.unit
class TestSha256File:
    def test_none_returns_none(self):
        assert sha256_file(None) is None

    def test_missing_path_returns_none(self, tmp_path):
        assert sha256_file(tmp_path / "nonexistent.txt") is None

    def test_known_content(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello")
        digest = sha256_file(f)
        import hashlib
        assert digest == hashlib.sha256(b"hello").hexdigest()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        digest = sha256_file(f)
        import hashlib
        assert digest == hashlib.sha256(b"").hexdigest()


@pytest.mark.unit
class TestWrite:
    def test_creates_manifest_json(self, tmp_path):
        path = write(tmp_path, seed=42)
        assert path == tmp_path / "manifest.json"
        assert path.exists()

    def test_standard_header_fields(self, tmp_path):
        write(tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert "trap_version" in data
        assert "git_commit" in data
        assert "timestamp_utc" in data

    def test_kwargs_merged(self, tmp_path):
        write(tmp_path, seed=3469, k=17, model={"path": "foo"})
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["seed"] == 3469
        assert data["k"] == 17
        assert data["model"] == {"path": "foo"}

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        write(nested)
        assert (nested / "manifest.json").exists()

    def test_idempotent_overwrite(self, tmp_path):
        write(tmp_path, x=1)
        write(tmp_path, x=2)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["x"] == 2
