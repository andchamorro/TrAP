"""Pipeline-wide verbosity control for TrAP.

Three levels — configure once at CLI entry, propagates through loguru filters:

    off      Only SUCCESS, WARNING, ERROR.  Default — minimal noise on HPC.
    normal   + STAGE (custom level 23): stage start/end, key metrics, elapsed.
    detailed + INFO (level 20): full operational detail without debug noise.

Usage::

    from trap.config.verbosity import set_verbosity, VerbosityLevel
    set_verbosity("normal")          # or VerbosityLevel.NORMAL
    set_verbosity(VerbosityLevel.DETAILED)

    # environment variable (applied at import time of trap.config.config)
    TRAP_VERBOSITY=normal python -m trap.loaders.tokenizer train ...

Expected output examples
------------------------
off::

    2026-05-30 12:00:01 | SUCCESS | Tokenizer training complete.

normal::

    2026-05-30 12:00:00 | STAGE | [tokenizer:train] algorithm=unigram k=17 vocab_size=32000
    2026-05-30 12:00:00 | STAGE | [tokenizer:train] corpus loaded — 87,324 sequences
    2026-05-30 12:00:42 | STAGE | [tokenizer:train] done — elapsed=42.3 s
    2026-05-30 12:00:42 | SUCCESS | Tokenizer training complete.

detailed::

    (all of normal, plus)
    2026-05-30 12:00:00 | INFO | [fix-4] Using all 87,324 seqs (~420M post-k-mer chars …)
    2026-05-30 12:00:00 | INFO | [fix-1] Training on 87,324 seqs via chain.from_iterable …
    2026-05-30 12:00:00 | INFO | Training tokenizer...
"""

from __future__ import annotations

import enum
from typing import Union


class VerbosityLevel(str, enum.Enum):
    OFF = "off"
    NORMAL = "normal"
    DETAILED = "detailed"


_ORDER = [VerbosityLevel.OFF, VerbosityLevel.NORMAL, VerbosityLevel.DETAILED]

_current: VerbosityLevel = VerbosityLevel.OFF

# Numeric loguru level for each verbosity tier.
# STAGE is registered as 23 in trap.config.config; INFO is 20; SUCCESS is 25.
_MIN_LEVEL: dict[VerbosityLevel, int] = {
    VerbosityLevel.OFF: 25,       # SUCCESS and above
    VerbosityLevel.NORMAL: 23,    # STAGE and above
    VerbosityLevel.DETAILED: 20,  # INFO and above (no DEBUG at 10)
}


def set_verbosity(level: Union[VerbosityLevel, str]) -> None:
    """Set the global pipeline verbosity level.

    Args:
        level: One of ``"off"``, ``"normal"``, ``"detailed"``
            or a :class:`VerbosityLevel` enum member.

    Raises:
        ValueError: If *level* is a string that does not match any known level.
    """
    global _current
    if isinstance(level, str):
        level = VerbosityLevel(level.lower())
    _current = level


def get_verbosity() -> VerbosityLevel:
    """Return the current global verbosity level."""
    return _current


def is_at_least(level: VerbosityLevel) -> bool:
    """Return True if the current verbosity is at least *level*.

    Args:
        level: Minimum :class:`VerbosityLevel` to test against.

    Returns:
        True when the current level is equal to or more verbose than *level*.
    """
    return _ORDER.index(_current) >= _ORDER.index(level)


def verbosity_filter(record: dict) -> bool:
    """Loguru ``filter`` callback — returns True when the record should pass.

    Attach to a loguru sink::

        logger.add(sink, filter=verbosity_filter)

    Args:
        record: Loguru log-record dict (contains ``"level"`` with a
            ``.no`` attribute).

    Returns:
        True if the record's numeric level meets the current verbosity floor.
    """
    return record["level"].no >= _MIN_LEVEL[_current]
