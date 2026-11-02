from importlib.metadata import PackageNotFoundError, version

from trap.config import config  # noqa: F401

try:
    __version__ = version("trap")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "unknown"
