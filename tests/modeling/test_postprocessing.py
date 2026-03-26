"""Tests for trap.modeling.postprocessing — sanitize_paths utility."""

from pathlib import Path

import pytest

from trap.modeling.postprocessing import sanitize_paths


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
