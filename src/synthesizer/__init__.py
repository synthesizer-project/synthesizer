import os
from pathlib import Path

import yaml
from platformdirs import user_data_dir

from synthesizer.data.initialise import (
    get_base_dir,
    get_data_dir,
    get_database_dir,
    get_grids_dir,
    get_instrument_dir,
    get_test_data_dir,
    synth_initialise,
)

# Initialize Synthesizer, this will only be run if the data directory and
# subdirectories do not exist, i.e. when the package is first imported or after
# any of the envionment variables have been changed.
synth_initialise()

# Define all the directory paths
BASE_DIR = get_base_dir()
DATA_DIR = get_data_dir()
DATABASE_FILE = get_database_dir() / "downloader_database.yml"
GRID_DIR = get_grids_dir()
TEST_DATA_DIR = get_test_data_dir()
INSTRUMENT_CACHE_DIR = get_instrument_dir()


# Make a version available at the top level
from synthesizer._version import __version__

# Make the openmp check available at the top level
from synthesizer.extensions.openmp_check import check_openmp

# Import an alias for the galaxy factory function
from synthesizer.galaxy import galaxy
from synthesizer.galaxy import galaxy as Galaxy

# Import the main classes to make them available at the top level
from synthesizer.grid import Grid

# Import the filters module to the top level to maintain old API
# before the filters module was moved to the instruments module
from synthesizer.instruments import filters

# Import the various utils submodules to make them available
# at the top level
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
    "INSTRUMENT_DIR",
]
