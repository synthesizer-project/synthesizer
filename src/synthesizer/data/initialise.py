"""A module containing classes and functions for initializing Synthesizer.

This module runs at first import or on-demand to:
  - Create the user data directory and subdirectories
  - Copy default resource files (unit config, etc.)
  - Set environment variables for easy access
  - Report status with colored symbols and ASCII art

NOTE: This module only uses standard library and importlib.resources; it must
not import other Synthesizer modules to avoid circular dependencies.
"""

import os
from importlib import resources
from pathlib import Path

import yaml
from platformdirs import user_data_dir

from synthesizer import exceptions

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


def testdata_dir_exists() -> bool:
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


def default_units_exists() -> bool:
    """Check if the default units file exists.

    This function checks if the default_units.yml file exists in the
    Synthesizer base directory.
    """
    user_units_file = get_base_dir() / "default_units.yml"
    return user_units_file.exists()


def default_units_needs_update() -> bool:
    """Check if the default units file is missing entries or invalid.

    This function will compare the default_units.yml file in the source
    code with the one in the user's base directory, and determine if the
    user's file needs updating with new entries, is corrupted, or is missing.

    We only add new entries, so we don't overwrite the user's preferences.

    Returns:
        True if the file needs updating (missing, invalid, or incomplete),
        False if all default categories are present and file is valid.
    """
    # We need to update if the file doesn't exist
    if not default_units_exists():
        return True

    # Otherwise, see if we need to update the existing file
    try:
        # Load the default units from the package
        with resources.open_text("synthesizer", "default_units.yml") as f:
            default_units = yaml.safe_load(f)

        # Load the user's default units
        user_units_file = get_base_dir() / "default_units.yml"
        with open(user_units_file, "r") as f:
            user_units = yaml.safe_load(f)

        # Handle case where file exists but is empty or invalid
        if user_units is None or not isinstance(user_units, dict):
            return True  # Invalid file, needs update
        if "UnitCategories" not in user_units:
            return True  # Missing UnitCategories, needs update

        # Check for missing keys in the user's units
        for key in default_units["UnitCategories"].keys():
            if key not in user_units["UnitCategories"]:
                return True  # Missing key found, needs update

        return False  # All keys present, no update needed

    except yaml.YAMLError:
        # Invalid YAML syntax - needs update
        return True
    except (OSError, IOError):
        # File access issues - needs update
        return True
    except Exception:
        # Unexpected error - needs update
        return True


class SynthesizerInitializer:
    """Encapsulates the initialisation of the Synthesizer data directory.

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

        # Initialize status dictionary for each step
        keys = [
            "base_dir",
            "data_dir",
            "grids",
            "instrument_cache",
            "test_data",
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
                with (
                    resources.open_binary(package, resource_name) as src,
                    open(dest, "wb") as dst,
                ):
                    dst.write(src.read())
                self.status[key] = "created"
        except Exception:
            self.status[key] = "failed"

    def _copy_units(self) -> None:
        """Copy the units over respecting existing user file.

        Instead of copying over the file blindly this will read both and update
        the default units with the user preference before writing it back out.
        This process will leave the users preferences intact while adding any
        new default units that may have been added since their last update.

        Raises:
            MissingUnits: If the default units file cannot be loaded or
                the user's units file cannot be written.
        """
        try:
            # Load the default units from the package
            with resources.open_text("synthesizer", "default_units.yml") as f:
                default_units = yaml.safe_load(f)

            # Load the user's default units
            user_units_file = self.base_dir / "default_units.yml"
            if user_units_file.exists():
                with open(user_units_file, "r") as f:
                    user_units = yaml.safe_load(f)

                # Handle case where file exists but is empty or invalid
                if user_units is None or not isinstance(user_units, dict):
                    user_units = {"UnitCategories": {}}
                elif "UnitCategories" not in user_units:
                    user_units = {"UnitCategories": {}}
            else:
                user_units = {"UnitCategories": {}}

            # Update the default units with the users to overwrite any
            # old preferences
            default_units["UnitCategories"].update(
                user_units["UnitCategories"]
            )

            # Write the updated units back to the user's file
            with open(user_units_file, "w") as f:
                yaml.dump(default_units, f)

            self.status["units_file"] = "created"

        except yaml.YAMLError as e:
            raise exceptions.MissingUnits(
                "Failed to parse YAML in default units file."
            ) from e
        except (OSError, IOError) as e:
            raise exceptions.MissingUnits(
                "Failed to read or write default units file."
            ) from e
        except Exception as e:
            raise exceptions.MissingUnits(
                "Failed to update default units file."
            ) from e

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
            2) Copy default_units.yml
            3) Check the environment variable state.
        """
        # Create the data directory and all subdirectories, this is safe
        # if any of the directories already exist
        self._make_dir(self.base_dir, "base_dir")
        self._make_dir(self.data_dir, "data_dir")
        self._make_dir(self.grids_dir, "grids")
        self._make_dir(self.instrument_cache_dir, "instrument_cache")
        self._make_dir(self.test_data_dir, "test_data")

        # Copy the default units to their user facing location
        if not default_units_exists():
            self._copy_resource(
                "synthesizer",
                "default_units.yml",
                self.base_dir / "default_units.yml",
                "units_file",
            )

        # Otherwise, we may need to update it
        elif default_units_needs_update():
            self._copy_units()

    def report(self) -> None:
        """Print a report of the initialisation."""
        # ANSI escape codes for styling
        yellow = "\033[93m"
        green = "\033[92m"
        cyan = "\033[96m"
        magenta = "\033[95m"
        reset = "\033[0m"

        def sym(key: str) -> str:
            return self._SYMBOLS.get(self.status.get(key, "failed"), "✖")

        # Prepare the directory output
        sections = [
            ("base_dir", "Base directory:", self.base_dir),
            ("data_dir", "Data directory:", self.data_dir),
            ("grids", "Grids directory:", self.grids_dir),
            (
                "instrument_cache",
                "Instrument cache directory:",
                self.instrument_cache_dir,
            ),
            ("test_data", "Test data directory:", self.test_data_dir),
        ]

        # Prepare the file output
        files = [
            (
                "units_file",
                "Default units file:",
                self.base_dir / "default_units.yml",
            ),
        ]

        # Prepare the environment variables output
        env_vars = [
            ("SYNTHESIZER_DIR", self.base_dir),
            ("SYNTHESIZER_DATA_DIR", self.data_dir),
            ("SYNTHESIZER_GRID_DIR", self.grids_dir),
            ("SYNTHESIZER_INSTRUMENT_CACHE", self.instrument_cache_dir),
            ("SYNTHESIZER_TEST_DATA_DIR", self.test_data_dir),
        ]

        # Figure out the longest label so we can pad
        all_labels = [label for _, label, _ in sections + files]
        max_label_len = max(len(label) for label in all_labels)

        # Center the galaxy art in an arbitrary width
        galaxy_lines = galaxy.splitlines()
        centered = "\n".join(line.center(100) for line in galaxy_lines)

        # Print the initialisation header with centered galaxy art
        print(f"{yellow}Synthesizer initialising...{reset}\n\n{centered}\n")

        # Print the status of directories and files
        print("  Initialised Synthesizer directories:")
        for key, label, val in sections:
            padded = label.ljust(max_label_len)
            print(f"  {sym(key)} {yellow}{padded}{reset}  {cyan}{val}{reset}")
        print()

        print("  Initialised Synthesizer files:")
        for key, label, val in files:
            padded = label.ljust(max_label_len)
            print(f"  {sym(key)} {yellow}{padded}{reset}  {cyan}{val}{reset}")
        print()

        # Environment variables
        print("  🔧 Environment variables (override all defaults):")
        for var, default in env_vars:
            if var in os.environ:
                padded = f"Found {var} =".ljust(
                    max_label_len + 5
                )  # +5 to line up with other lines
                print(
                    f"  {yellow}{padded}{reset} {cyan}{os.environ[var]}{reset}"
                )
            else:
                padded = f"To set {var}, add this to your shell config:".ljust(
                    max_label_len + 5
                )
                print(f"  {yellow}{padded}{reset}")
                print(f"    {magenta}export {var}='{default}'{reset}")
        print()

        print(f"{green}Synthesizer initialisation complete!{reset}\n")


def synth_initialise(verbose=True) -> None:
    """Run the Synthesizer initialization process.

    This function runs the initialisation process. It creates the necessary
    directories, copies default files, sets environment variables,
    and prints a report.
    """
    # Do all the directories already exist?
    all_exist = (
        data_dir_exists()
        and grids_dir_exists()
        and testdata_dir_exists()
        and instrument_cache_exists()
        and default_units_exists()
    )

    # Just exit if everything exists
    if all_exist:
        # But hang on, do we need to update the default units file?
        if default_units_needs_update():
            initializer = SynthesizerInitializer()
            initializer._copy_units()
            if verbose:
                print("  🟢 Default units file updated with new entries.")
            return

        # OK, everything exists, nothing to do
        if verbose:
            print(
                "  🟢 Synthesizer data directory already exists, "
                "no need to re-initialize."
            )
        return

    # Otherwise, create the initializer and run it, this will only make or copy
    # what is necessary
    initializer = SynthesizerInitializer()
    initializer.initialize()
    initializer.report()


def synth_report_config() -> None:
    """Report the Synthesizer configuration.

    This function prints the current Synthesizer configuration, including
    the base directory, data directory, grids directory, instrument cache,
    and test data directory.
    """
    initializer = SynthesizerInitializer()
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
