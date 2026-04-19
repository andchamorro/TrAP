import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from trap.config.verbosity import VerbosityLevel, set_verbosity, verbosity_filter

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
CONFIG_DIR = PROJ_ROOT / "config"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESULTS_DIR = PROJ_ROOT / "results"

# Register the custom STAGE level (between INFO=20 and SUCCESS=25).
# STAGE messages are visible at --verbosity normal and above.
try:
    logger.level("STAGE", no=23, color="<cyan>", icon="◎")
except ValueError:
    pass  # already registered (e.g. in multi-import or test scenarios)

# Initialise verbosity from environment before wiring the filtered sink so
# that STAGE logs emitted by CLI startup code respect the env setting.
_env_verbosity = os.environ.get("TRAP_VERBOSITY", "off")
try:
    set_verbosity(_env_verbosity)
except ValueError:
    pass  # unknown value — keep the default (off)

# If tqdm is installed, replace the default sink with a tqdm-compatible one
# that also applies the verbosity filter.
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        filter=verbosity_filter,
    )
except ModuleNotFoundError:
    # tqdm not available — patch the default sink with just the filter.
    try:
        logger.remove(0)
        logger.add(lambda msg: print(msg, end=""), colorize=True, filter=verbosity_filter)
    except Exception:
        pass

# Log the project root at STAGE level so it only appears at --verbosity normal+.
logger.log("STAGE", f"PROJ_ROOT: {PROJ_ROOT}")
