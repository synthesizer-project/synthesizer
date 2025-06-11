"""The main module for the synthesizer package."""

# Get the initialisation and data directory stuff we need before importing
# the rest of the package.
from synthesizer.data.initialise import (
    get_base_dir,
    get_data_dir,
    get_database_dir,
    get_grids_dir,
    get_instrument_dir,
    get_test_data_dir,
    synth_initialise,
)

# Initialize Synthesizer. This will only be run if the data directory and
# subdirectories do not exist, i.e. when the package is first imported or after
# any of the envionment variables have been changed.
synth_initialise()

# Define all the directory paths we need throughout the package.
BASE_DIR = get_base_dir()
DATA_DIR = get_data_dir()
DATABASE_FILE = get_database_dir() / "downloader_database.yml"
GRID_DIR = get_grids_dir()
TEST_DATA_DIR = get_test_data_dir()
INSTRUMENT_CACHE_DIR = get_instrument_dir()


# Make a version available at the top level
from synthesizer._version import __version__

# Import things we want at the top level
from synthesizer.emissions.line import LineCollection
from synthesizer.emissions.sed import Sed
from synthesizer.extensions.openmp_check import check_openmp
from synthesizer.galaxy import galaxy
from synthesizer.galaxy import galaxy as Galaxy  # add a convenient alias
from synthesizer.grid import Grid
from synthesizer.instruments import filters
from synthesizer.utils import art, integrate, plt, stats, util_funcs

# Define the __all__ variable to control what is imported with
# 'from synthesizer import *'
__all__ = [
    "art",
    "integrate",
    "plt",
    "stats",
    "util_funcs",
    "Grid",
    "Sed",
    "galaxy",
    "Galaxy",
    "check_openmp",
    "filters",
    "__version__",
    "BASE_DIR",
    "DATA_DIR",
    "DATABASE_FILE",
    "GRID_DIR",
    "TEST_DATA_DIR",
    "INSTRUMENT_CACHE_DIR",
]
