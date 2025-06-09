from importlib import resources
from pathlib import Path

import yaml
from platformdirs import user_data_dir

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

# Define the data directory where we expect to find the data files
DATA_DIR = Path(user_data_dir("synthesizer")) / "data"

# If the data directory does not exist, create it and copy some files
if not DATA_DIR.exists():
    # Create the subdirectories we expect to exist
    for sub in ("instrument_cache", "test_data", "grids", "database"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Copy the default units file to the data directory so that it is readily
    # editable by the user
    with resources.open_binary(__name__, "default_units.yml") as src, open(
        DATA_DIR / "default_units.yml", "wb"
    ) as dst:
        dst.write(src.read())

    # Copy the downloaders ids database yaml file to the data directory
    with resources.open_binary(
        f"{__name__}.downloader", "_data_ids.yml"
    ) as src, open(DATA_DIR / "downloader_database.yml", "wb") as dst:
        dst.write(src.read())


__all__ = [
    art,
    integrate,
    plt,
    stats,
    util_funcs,
    Grid,
    galaxy,
    Galaxy,
    check_openmp,
    filters,
    DATA_DIR,
]
