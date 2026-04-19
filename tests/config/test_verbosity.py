"""Tests for trap.config.verbosity."""

import pytest

from trap.config.verbosity import (
    VerbosityLevel,
    get_verbosity,
    is_at_least,
    set_verbosity,
    verbosity_filter,
)


@pytest.fixture(autouse=True)
def reset_verbosity():
    """Restore verbosity to OFF after each test."""
    yield
    set_verbosity(VerbosityLevel.OFF)


# ---------------------------------------------------------------------------
# VerbosityLevel enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerbosityLevelEnum:
    def test_values(self):
        assert VerbosityLevel.OFF.value == "off"
        assert VerbosityLevel.NORMAL.value == "normal"
        assert VerbosityLevel.DETAILED.value == "detailed"

    def test_str_coercion(self):
        assert VerbosityLevel("off") is VerbosityLevel.OFF
        assert VerbosityLevel("normal") is VerbosityLevel.NORMAL
        assert VerbosityLevel("detailed") is VerbosityLevel.DETAILED

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            VerbosityLevel("debug")


# ---------------------------------------------------------------------------
# set_verbosity / get_verbosity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetGetVerbosity:
    def test_set_enum(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert get_verbosity() is VerbosityLevel.NORMAL

    def test_set_string(self):
        set_verbosity("detailed")
        assert get_verbosity() is VerbosityLevel.DETAILED

    def test_set_string_case_insensitive(self):
        set_verbosity("NORMAL")
        assert get_verbosity() is VerbosityLevel.NORMAL

    def test_set_off(self):
        set_verbosity("normal")
        set_verbosity("off")
        assert get_verbosity() is VerbosityLevel.OFF

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            set_verbosity("verbose")

    def test_default_is_off(self):
        assert get_verbosity() is VerbosityLevel.OFF


# ---------------------------------------------------------------------------
# is_at_least
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsAtLeast:
    def test_off_is_only_at_least_off(self):
        set_verbosity(VerbosityLevel.OFF)
        assert is_at_least(VerbosityLevel.OFF)
        assert not is_at_least(VerbosityLevel.NORMAL)
        assert not is_at_least(VerbosityLevel.DETAILED)

    def test_normal_at_least_off_and_normal(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert is_at_least(VerbosityLevel.OFF)
        assert is_at_least(VerbosityLevel.NORMAL)
        assert not is_at_least(VerbosityLevel.DETAILED)

    def test_detailed_at_least_all(self):
        set_verbosity(VerbosityLevel.DETAILED)
        assert is_at_least(VerbosityLevel.OFF)
        assert is_at_least(VerbosityLevel.NORMAL)
        assert is_at_least(VerbosityLevel.DETAILED)


# ---------------------------------------------------------------------------
# verbosity_filter — loguru record duck-type mocks
# ---------------------------------------------------------------------------


class _Level:
    """Minimal stand-in for loguru's Level object (has .no attribute)."""

    def __init__(self, no: int):
        self.no = no


def _record(level_no: int) -> dict:
    return {"level": _Level(level_no)}


# Numeric loguru levels used in the filter:
#   INFO    = 20
#   STAGE   = 23  (custom)
#   SUCCESS = 25
#   WARNING = 30
#   ERROR   = 40


@pytest.mark.unit
class TestVerbosityFilter:
    def test_off_blocks_info(self):
        set_verbosity(VerbosityLevel.OFF)
        assert not verbosity_filter(_record(20))  # INFO

    def test_off_blocks_stage(self):
        set_verbosity(VerbosityLevel.OFF)
        assert not verbosity_filter(_record(23))  # STAGE

    def test_off_passes_success(self):
        set_verbosity(VerbosityLevel.OFF)
        assert verbosity_filter(_record(25))  # SUCCESS

    def test_off_passes_warning(self):
        set_verbosity(VerbosityLevel.OFF)
        assert verbosity_filter(_record(30))  # WARNING

    def test_off_passes_error(self):
        set_verbosity(VerbosityLevel.OFF)
        assert verbosity_filter(_record(40))  # ERROR

    def test_normal_blocks_info(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert not verbosity_filter(_record(20))  # INFO below STAGE floor

    def test_normal_passes_stage(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert verbosity_filter(_record(23))  # STAGE

    def test_normal_passes_success(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert verbosity_filter(_record(25))  # SUCCESS

    def test_normal_passes_warning(self):
        set_verbosity(VerbosityLevel.NORMAL)
        assert verbosity_filter(_record(30))  # WARNING

    def test_detailed_passes_info(self):
        set_verbosity(VerbosityLevel.DETAILED)
        assert verbosity_filter(_record(20))  # INFO

    def test_detailed_blocks_debug(self):
        set_verbosity(VerbosityLevel.DETAILED)
        assert not verbosity_filter(_record(10))  # DEBUG

    def test_detailed_passes_stage(self):
        set_verbosity(VerbosityLevel.DETAILED)
        assert verbosity_filter(_record(23))  # STAGE

    def test_detailed_passes_success_and_above(self):
        set_verbosity(VerbosityLevel.DETAILED)
        assert verbosity_filter(_record(25))  # SUCCESS
        assert verbosity_filter(_record(30))  # WARNING
        assert verbosity_filter(_record(40))  # ERROR
