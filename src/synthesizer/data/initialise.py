"""A module containing classes and functions for initializing Synthesizer.

This module runs at first import or on-demand to:
  - Create the user data directory and subdirectories
  - Copy default resource files (unit config, ID databases)
  - Set environment variables for easy access
  - Report status with colored symbols and ASCII art

NOTE: This module only uses standard library and importlib.resources; it must
not import other Synthesizer modules to avoid circular dependencies.
"""

import os
from importlib import resources
from pathlib import Path

from platformdirs import user_data_dir

# ASCII art mirror of the galaxy logo (cannot import directly)
galaxy = (
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠒⠒⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⢀⣤⣶⡾⠿⠿⠿⠿⣿⣿⣶⣦⣄⠙⠷⣤⡀⠀⠀⠀⠀\n"
    "⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣷⣄⠘⢿⡄⠀⠀⠀\n"
    "⠀⢀⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠐⠂⠠⢄⡀⠈⢿⣿⣧⠈⢿⡄⠀⠀\n"
    "⢀⠏⠀⠀⠀⢀⠄⣀⣴⣾⠿⠛⠛⠛⠷⣦⡙⢦⠀⢻⣿⡆⠘⡇⠀⠀\n"
    "⠀⠀⠀+-+-+-+-+-+-+-+-+-+-+-+⡇⠀⠀\n"
    "⠀⠀⠀|S|Y|N|T|H|E|S|I|Z|E|R|⠃⠀⠀\n"
    "⠀⠀⢰+-+-+-+-+-+-+-+-+-+-+-+⠀⠀⠀\n"
    "⠀⠀⢸⡇⠸⣿⣷⠀⢳⡈⢿⣦⣀⣀⣀⣠⣴⣾⠟⠁⠀⠀⠀⠀⢀⡎\n"
    "⠀⠀⠘⣷⠀⢻⣿⣧⠀⠙⠢⠌⢉⣛⠛⠋⠉⠀⠀⠀⠀⠀⠀⣠⠎⠀\n"
    "⠀⠀⠀⠹⣧⡀⠻⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠃⠀⠀\n"
    "⠀⠀⠀⠀⠈⠻⣤⡈⠻⢿⣿⣷⣦⣤⣤⣤⣤⣤⣴⡾⠛⠉⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⠀⠈⠙⠶⢤⣈⣉⠛⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
)


def get_base_dir() -> Path:
    """Get the Synthesizer base directory path.

    This function returns the path to the Synthesizer base directory,
    which is determined using platformdirs.user_data_dir.
    """
    # First check if we have an environment variable set
    if "SYNTHESIZER_DIR" in os.environ:
        return Path(os.environ["SYNTHESIZER_DIR"])

    # Otherwise, use the platformdirs to define the default location
    return Path(user_data_dir("Synthesizer"))


def base_dir_exists() -> bool:
    """Check if the Synthesizer base directory exists.

    This function checks if the Synthesizer base directory, as defined by
    get_base_dir(), exists on the filesystem.
    """
    return get_base_dir().exists()


def get_data_dir() -> Path:
    """Get the Synthesizer data directory path.

    This function returns the path to the Synthesizer data directory,
    which is determined using platformdirs.user_data_dir with an added
    'data' subdirectory.
    """
    # First check if we have an environment variable set
    if "SYNTHESIZER_DATA_DIR" in os.environ:
        return Path(os.environ["SYNTHESIZER_DATA_DIR"])

    # Otherwise, use the platformdirs to define the default location
    data_dir = get_base_dir() / "data"

    return data_dir


def data_dir_exists() -> bool:
    """Check if the Synthesizer data directory exists.

    This function checks if the Synthesizer data directory, as defined by
    get_data_dir(), exists on the filesystem.
    """
    return get_data_dir().exists()


def get_grids_dir() -> Path:
    """Get the Synthesizer grids directory path.

    This function returns the path to the Synthesizer grids directory,
    which is a subdirectory of the Synthesizer data directory.
    """
    # First check if we have an environment variable set
    if "SYNTHESIZER_GRID_DIR" in os.environ:
        return Path(os.environ["SYNTHESIZER_GRID_DIR"])

    # Otherwise, use the platformdirs to define the default location
    return get_base_dir() / "grids"


def grids_dir_exists() -> bool:
    """Check if the Synthesizer grids directory exists.

    This function checks if the Synthesizer grids directory, as defined by
    get_grids_dir(), exists on the filesystem.
    """
    return get_grids_dir().exists()


def get_test_data_dir() -> Path:
    """Get the Synthesizer test data directory path.

    This function returns the path to the Synthesizer test data directory,
    which is a subdirectory of the Synthesizer data directory.
    """
    # First check if we have an environment variable set
    if "SYNTHESIZER_TEST_DATA_DIR" in os.environ:
        return Path(os.environ["SYNTHESIZER_TEST_DATA_DIR"])

    # Otherwise, use the platformdirs to define the default location
    return get_data_dir() / "test_data"


def test_data_dir_exists() -> bool:
    """Check if the Synthesizer test data directory exists.

    This function checks if the Synthesizer test data directory, as defined by
    get_test_data_dir(), exists on the filesystem.
    """
    return get_test_data_dir().exists()


def get_instrument_dir() -> Path:
    """Get the Synthesizer instrument cache directory path.

    This function returns the path to the Synthesizer instrument cache
    directory, which is a subdirectory of the Synthesizer data directory.
    """
    # First check if we have an environment variable set
    if "SYNTHESIZER_INSTRUMENT_CACHE" in os.environ:
        return Path(os.environ["SYNTHESIZER_INSTRUMENT_CACHE"])

    # Otherwise, use the platformdirs to define the default location
    return get_base_dir() / "instrument_cache"


def instrument_cache_exists() -> bool:
    """Check if the Synthesizer instrument cache directory exists.

    This function checks if the Synthesizer instrument cache directory,
    as defined by get_instrument_dir(), exists on the filesystem.
    """
    return get_instrument_dir().exists()


class SynthesizerInitializer:
    """Encapsulates the initialisation of the Sythesizer data directory.

    This class handles the creation of the Synthesizer data directory and its
    subdirectories, and copies default resource files into the data directory.

    This should not be instantiated directly (though there is no nasty side
    effect of doing so). It is intended to be used via the synth_initialise()
    function, which will create an instance and run the initialization
    process. This function is both an entry point that be specifically invoked
    and automatically called on import if the data directory does not exist.
    """

    # Mapping of operation results to symbols
    _SYMBOLS = {
        "created": "✔",  # successfully created or copied
        "exists": "○",  # already existed
        "failed": "✖",  # failed to create or copy
    }

    def __init__(self) -> None:
        """Initialise the initializer (intialiseception?)."""
        # Attach the various directories to the instance
        self.base_dir = get_base_dir()
        self.data_dir = get_data_dir()
        self.grids_dir = get_grids_dir()
        self.test_data_dir = get_test_data_dir()
        self.instrument_cache_dir = get_instrument_dir()
        self.database_dir = self.data_dir / "database"

        # Initialize status dictionary for each step
        keys = [
            "base_dir",
            "data_dir",
            "grids",
            "instrument_cache",
            "test_data",
            "database",
            "units_file",
            "ids_file",
        ]
        self.status = {key: None for key in keys}

    def _make_dir(self, path: Path, key: str) -> None:
        """Create a directory if missing.

        This will also record status of the creation:
            - 'created' if newly made,
            - 'exists' if already present,
            - 'failed' on exception.

        Args:
            path (Path): The directory path to create.
            key (str): The key to use in the status dictionary.
        """
        try:
            if path.exists():
                self.status[key] = "exists"
            else:
                path.mkdir(parents=True, exist_ok=False)
                self.status[key] = "created"
        except Exception:
            self.status[key] = "failed"

    def _copy_resource(
        self,
        package: str,
        resource_name: str,
        dest: Path,
        key: str,
    ) -> None:
        """Copy a bundled resource (binary) from the installed package.

        This will move a bundled resource (i.e. the package data defined in
        the setup call in setup.py) into the user's data directory, tracking
        the outcome.

        Args:
            package (str): The package name containing the resource.
            resource_name (str): The name of the resource to copy.
            dest (Path): The destination path where the resource should
                be copied to.
            key (str): The key to use in the status dictionary for this
                operation.
        """
        try:
            if dest.exists():
                self.status[key] = "exists"
            else:
                with resources.open_binary(
                    package, resource_name
                ) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                self.status[key] = "created"
        except Exception:
            self.status[key] = "failed"

    def _remove_dir(self, path: Path) -> None:
        """Recursively remove a directory and its contents.

        Args:
            path (Path): The directory path to remove.
        """
        if path.exists():
            # Recursively remove all files and subdirectories
            for item in path.iterdir():
                if item.is_dir():
                    self._remove_dir(item)
                else:
                    item.unlink()
                    print(f"  🗑 Removed file: {item}")

            # Remove the directory itself
            path.rmdir()
            print(f"  🗑 Removed directory: {path}")

    def initialize(self) -> None:
        """Run the full initialization sequence.

        This method performs the following steps:
            1) Create base and subdirectories
            2) Copy default_units.yml and ID database
            3) Check the environment variable state.
        """
        # Create the data directory and all subdirectories, this is safe
        # if any of the directories already exist
        self._make_dir(self.base_dir, "base_dir")
        self._make_dir(self.data_dir, "data_dir")
        self._make_dir(self.grids_dir, "grids")
        self._make_dir(self.instrument_cache_dir, "instrument_cache")
        self._make_dir(self.test_data_dir, "test_data")
        self._make_dir(self.database_dir, "database")

        # Copy the default units and database IDs to their user
        # facing locations
        self._copy_resource(
            "synthesizer",
            "default_units.yml",
            self.data_dir / "default_units.yml",
            "units_file",
        )
        self._copy_resource(
            "synthesizer.downloader",
            "_data_ids.yml",
            self.database_dir / "downloader_database.yml",
            "ids_file",
        )

    def report(self) -> None:
        """Print a report of the initialization status."""
        # ANSI escape codes for styling
        yellow = "\033[93m"
        green = "\033[92m"
        cyan = "\033[96m"
        magenta = "\033[95m"
        reset = "\033[0m"

        def sym(key: str) -> str:
            """Return the symbol representing status[key]."""
            return self._SYMBOLS.get(self.status.get(key, "failed"), "✖")

        # Header + ASCI art display
        print(
            f"{yellow}Synthesizer initialising...{reset}\n\n"
            f"{cyan}{galaxy}{reset}\n"
        )

        print("  Initialised Synthesizer directories:")

        # Base directory outcome
        print(
            f"  {sym('base_dir')} {yellow}Base directory:{reset} "
            f"{cyan}{self.base_dir}{reset}"
        )

        # Data directory outcome
        print(
            f"  {sym('data_dir')} {yellow}Data directory:{reset} "
            f"{cyan}{self.data_dir}{reset}"
        )

        # Grids directory outcome
        print(
            f"  {sym('grids')} {yellow}Grids directory:{reset} "
            f"{cyan}{self.grids_dir}{reset}"
        )

        # Instrument cache directory outcome
        print(
            f"  {sym('instrument_cache')} "
            f"{yellow}Instrument cache directory:{reset} "
            f"{cyan}{self.instrument_cache_dir}{reset}"
        )

        # Test data directory outcome
        print(
            f"  {sym('test_data')} {yellow}Test data directory:{reset} "
            f"{cyan}{self.test_data_dir}{reset}"
        )

        # Database directory outcome
        print(
            f"  {sym('database')} "
            f"{yellow}Downloader database directory:{reset} "
            f"{cyan}{self.database_dir}{reset}"
        )

        print()

        # Files outcomes
        print(
            f"  {sym('units_file')} {yellow}Default units file:{reset} "
            f"{cyan}{self.data_dir / 'default_units.yml'}{reset}"
        )
        print(
            f"  {sym('ids_file')} {yellow}Downloaders IDs DB:{reset} "
            f"{cyan}{self.data_dir / 'downloader_database.yml'}{reset}"
        )

        print()

        # Also report the environment variables that are set, or if not set
        # give the user a copy and pasteable command to set them
        print(
            f"  {yellow}🔧 Environment variables (setting these will "
            f"overide all defaults):{reset}"
        )
        if "SYNTHESIZER_DIR" in os.environ:
            print(
                f"  {yellow}Found SYNTHESIZER_DIR ={reset} "
                f"{cyan}{os.environ['SYNTHESIZER_DIR']}{reset}"
            )
        else:
            print(
                f"  {yellow}To set SYNTHESIZER_DIR add this (or a custom"
                f" path) to your shell config:{reset}\n"
                f"    {magenta}export SYNTHESIZER_DIR="
                f"'{self.base_dir}'{reset} "
            )
        if "SYNTHESIZER_DATA_DIR" in os.environ:
            print(
                f"  {yellow}Found SYNTHESIZER_DATA_DIR ={reset} "
                f"{cyan}{os.environ['SYNTHESIZER_DATA_DIR']}{reset}"
            )
        else:
            print(
                f"  {yellow}To set SYNTHESIZER_DATA_DIR add this (or a custom"
                f" path) to your shell config:{reset}\n"
                f"    {magenta}export SYNTHESIZER_DATA_DIR="
                f"'{self.data_dir}'{reset} "
            )
        if "SYNTHESIZER_GRID_DIR" in os.environ:
            print(
                f"  {yellow}Found SYNTHESIZER_GRID_DIR ={reset} "
                f"{cyan}{os.environ['SYNTHESIZER_GRID_DIR']}{reset}"
            )
        else:
            print(
                f"  {yellow}To set SYNTHESIZER_GRID_DIR add this (or a custom"
                f" path) to your shell config:{reset}\n"
                f"    {magenta}export SYNTHESIZER_GRID_DIR="
                f"'{self.grids_dir}'{reset} "
            )
        if "SYNTHESIZER_INSTRUMENT_CACHE" in os.environ:
            print(
                f"  {yellow}Found SYNTHESIZER_INSTRUMENT_CACHE ={reset} "
                f"{cyan}{os.environ['SYNTHESIZER_INSTRUMENT_CACHE']}{reset}"
            )
        else:
            print(
                f"  {yellow}To set SYNTHESIZER_INSTRUMENT_CACHE add "
                f"this (or a custom path) to your shell config:{reset}\n"
                f"    {magenta}export SYNTHESIZER_INSTRUMENT_CACHE="
                f"'{self.instrument_cache_dir}'{reset} "
            )
        if "SYNTHESIZER_TEST_DATA_DIR" in os.environ:
            print(
                f"  {yellow}Found SYNTHESIZER_TEST_DATA_DIR ={reset} "
                f"{cyan}{os.environ['SYNTHESIZER_TEST_DATA_DIR']}{reset}"
            )
        else:
            print(
                f"  {yellow}To set SYNTHESIZER_TEST_DATA_DIR add "
                f" this (or a custom path) to your shell config:{reset}\n"
                f"    {magenta}export SYNTHESIZER_TEST_DATA_DIR="
                f"'{self.test_data_dir}'{reset} "
            )
        print()
        print(f"{green}Synthesizer initialisation complete!{reset}\n")


def synth_initialise() -> None:
    """Run the Synthesizer initialization process.

    This function runs the initialisation process. It creates the necessary
    directories, copies default files, sets environment variables,
    and prints a report.
    """
    # Do all the directories already exist?
    all_exist = (
        data_dir_exists()
        and grids_dir_exists()
        and test_data_dir_exists()
        and instrument_cache_exists()
    )

    # Have the files already been copied?
    if all_exist:
        default_units_file = get_data_dir() / "default_units.yml"
        ids_file = get_data_dir() / "database" / "downloader_database.yml"
        all_exist = default_units_file.exists() and ids_file.exists()

    # Just exit if the data directory already exists
    if all_exist:
        return

    # Otherwise, create the initializer and run it, this will only make or copy
    # what is necessary
    initializer = SynthesizerInitializer()
    initializer.initialize()
    initializer.report()


def synth_clear_data() -> None:
    """Clear the Synthesizer data directory.

    This function removes the entire Synthesizer data directory and its
    contents. It is useful for resetting the environment or clearing
    cached data.
    """
    initializer = SynthesizerInitializer()
    try:
        initializer._remove_dir(initializer.base_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer base directory: {e}")
    try:
        initializer._remove_dir(initializer.data_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer data directory: {e}")
    try:
        initializer._remove_dir(initializer.grids_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer grids directory: {e}")
    try:
        initializer._remove_dir(initializer.test_data_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer test data directory: {e}")
    try:
        initializer._remove_dir(initializer.instrument_cache_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer instrument cache directory: {e}")
    try:
        initializer._remove_dir(initializer.database_dir)
    except Exception as e:
        print(f"Failed to clear Synthesizer database directory: {e}")
