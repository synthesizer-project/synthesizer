"""A module for downloading different datasets required by synthesizer.

This module contains functions for downloading different datasets required by
synthesizer. This includes the test grid data, dust emission grid from Draine
and Li (2007) and CAMELS simulation data.

An entry point (synthesizer-download) is provided for each dataset, allowing
the user to download the data from the command line. Although the script can
be called directly.

Example Usage:
    synthesizer-download --test-grids --destination /path/to/destination
    synthesizer-download --dust-grid --destination /path/to/destination
    synthesizer-download --camels-data --destination /path/to/destination

"""

import argparse
import os

import requests
import yaml
from tqdm import tqdm

from synthesizer import DATA_DIR, exceptions
from synthesizer.instruments import AVAILABLE_INSTRUMENTS
from synthesizer.synth_warnings import warn

# Define the location of this file
THIS_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])


def load_database_yaml():
    """Load the database yaml file."""
    with open(f"{DATA_DIR}/downloader_database.yml", "r") as f:
        data = yaml.safe_load(f)

    return data


def load_test_data_links():
    """Load the test data links from the yaml file."""
    return load_database_yaml()["TestData"]


def load_dust_data_links():
    """Load the dust data links from the yaml file."""
    return load_database_yaml()["DustData"]


def load_instrument_data_links():
    """Load the instrument data links from the yaml file."""
    return load_database_yaml()["InstrumentData"]


# Get the dicts contain the locations of the test and dust data
TEST_FILES = load_test_data_links()
DUST_FILES = load_dust_data_links()
INSTRUMENT_FILES = load_instrument_data_links()

# Combine everything into a nice single dict
AVAILABLE_FILES = {**TEST_FILES, **DUST_FILES, **INSTRUMENT_FILES}

# Define the instruments destination (this is always fixed to the cache dir)
INSTRUMENT_CACHE_DIR = os.path.join(
    DATA_DIR,
    "instrument_cache",
)

# Define the path to the test data
TEST_DATA_DIR = os.path.join(
    DATA_DIR,
    "test_data",
)
TEST_GRID_DIR = os.path.join(
    DATA_DIR,
    "grids",
)

# Define a translation between instrument file names and their class names
# (remove the .hdf5 extension and remove underscores)
INSTRUMENT_TRANSLATION = {
    key.replace(".hdf5", "").replace("_", ""): key for key in INSTRUMENT_FILES
}

# Define a translation for the test data files that converts the true
# filename into the "test data alias"
TEST_DATA_TRANSLATION = {
    "test_grid.hdf5": (
        "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"
    ),
    "test_grid_agn-blr.hdf5": "qsosed-test_cloudy-c23.01-agn-test.hdf5",
    "test_grid_agn-nlr.hdf5": "qsosed-test_cloudy-c23.01-agn-test.hdf5",
}


def _download(
    filename,
    save_dir,
):
    """Download the file from the data server.

    We extract the link for the file and its name on the server from the
    AVAILABLE_FILES dictionary.

    We are now using Box

    Args:
        filename (str):
            The name of the file to download.
        save_dir (str):
            The directory in which to save the file.
    """
    # Define the filename we will save under (this will ignore any aliases)
    savename = filename

    # Do we have an file with an alias?
    if filename in TEST_DATA_TRANSLATION:
        # If the filename is in the translation dict, use the alias
        filename = TEST_DATA_TRANSLATION[filename]

    # Unpack the file details for extraction
    file_details = AVAILABLE_FILES[filename]

    # Unpack the url
    url = file_details["direct_link"]

    # Define the save path
    save_path = f"{save_dir}/{savename}"

    # Download the file
    response = requests.get(url, stream=True)

    # Ensure the request was successful
    if response.status_code != 200:
        raise exceptions.DownloadError(
            f"Failed to download {url}. Status code: {response.status_code}"
        )

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    # Stream the file to disk with a nice progress bar.
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)


def download_test_grids(destination):
    """Download the test grids for synthesizer.

    Args:
        destination (str):
            The path to the destination directory.
    """
    # Download each file
    for f in [
        "test_grid.hdf5",
        "test_grid_agn-blr.hdf5",
        "test_grid_agn-nlr.hdf5",
    ]:
        _download(f, destination)


def download_stellar_test_grids(destination):
    """Download the SPS test grids for synthesizer.

    Args:
        destination (str):
            The path to the destination directory.
    """
    # Download the stellar grid
    _download("test_grid.hdf5", destination)


def download_agn_test_grids(destination):
    """Download the AGN test grids for synthesizer.

    Args:
        destination (str):
            The path to the destination directory.
    """
    # Download each file
    for f in ["test_grid_agn-blr.hdf5", "test_grid_agn-nlr.hdf5"]:
        _download(f, destination)


def download_dust_grid(destination):
    """Download the Drain and Li (2007) dust emission grid for synthesizer.

    Args:
        destination (str):
            The path to the destination directory.
    """
    # Download the dust grid
    _download("draine_li_dust_emission_grid_MW_3p1.hdf5", destination)


def download_camels_data(destination):
    """Download the CAMELS data.

    Args:
        destination (str):
            The path to the destination directory.
    """
    _download("camels_snap.hdf5", destination)
    _download("camels_subhalo.hdf5", destination)


def download_sc_sam_test_data(destination):
    """Download the SC-SAM test data.

    Args:
        destination (str):
            The path to the destination directory.
    """
    _download("sc-sam_sfhist.dat", destination)


def download_instruments(destination, instruments):
    """Download the instruments.

    Args:
        destination (str):
            The path to the destination directory.
        instruments (list):
            The list of instruments to download.
    """
    # Download each instrument
    for instrument in instruments:
        _download(INSTRUMENT_TRANSLATION[instrument], destination)


def download():
    """Download different datasets based on command line args."""
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Download datasets for synthesizer"
    )

    # Add a flag to handle the test data
    parser.add_argument(
        "--test-grids",
        "-t",
        action="store_true",
        help="Download the test data for synthesizer",
    )
    parser.add_argument(
        "--stellar-test-grids",
        "-s",
        action="store_true",
        help="Download only the stellar test data for synthesizer",
    )
    parser.add_argument(
        "--agn-test-grids",
        "-a",
        action="store_true",
        help="Download only the AGN test data for synthesizer",
    )

    # Add the flag for dust data
    parser.add_argument(
        "--dust-grid",
        "-D",
        action="store_true",
        help="Download the dust grid for the Drain & Li (2007) model",
    )

    # Add the flag for processed camels data
    parser.add_argument(
        "--camels-data",
        "-c",
        action="store_true",
        help="Download the CAMELS TNG dataset for testing",
    )

    # Add the flag for processed sc_sam data
    parser.add_argument(
        "--scsam-data",
        "-S",
        action="store_true",
        help="Download the SC-SAM data for testing",
    )

    # Add a flag to grab all simulation data
    parser.add_argument(
        "--all-sim-data",
        action="store_true",
        help="Download all available simulation data",
    )

    # Add a flag to download instruments
    parser.add_argument(
        "--instruments",
        nargs="+",
        type=str,
        help="Download the specified instruments",
        choices=AVAILABLE_INSTRUMENTS,
        default=[],
    )

    # Add a flag to download all instruments
    parser.add_argument(
        "--all-instruments",
        action="store_true",
        help="Download all available instruments",
    )

    # Add a flag to go ham and download everything
    parser.add_argument(
        "--all",
        "-A",
        action="store_true",
        help="Download all available data including test data, "
        "camels simulation data, and the dust grid",
    )

    # Add the destination argument (will not effect test data)
    parser.add_argument(
        "--destination",
        "-d",
        type=str,
        help="The path to the destination directory",
        default=None,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract flags
    test = args.test_grids
    stellar = args.stellar_test_grids
    agn = args.agn_test_grids
    dust = args.dust_grid
    camels = args.camels_data
    everything = args.all
    dest = args.destination
    scsam = args.scsam_data
    all_sim_data = args.all_sim_data
    instruments = args.instruments
    all_instruments = args.all_instruments

    # Check if the destination directory exists
    if dest is not None and not os.path.exists(dest):
        raise exceptions.DownloadError(
            f"Destination directory {dest} does not exist. "
            "Please create it before running this script."
        )

    # Warn that dest is ignored for instruments
    if (len(instruments) > 0 or all_instruments) and dest is not None:
        warn(
            "The destination argument is ignored for instruments. "
            "Instruments are always downloaded to the cache directory "
            f"({INSTRUMENT_CACHE_DIR}).",
        )

    # Are we just getting everything?
    if everything:
        download_test_grids(dest if dest is not None else TEST_GRID_DIR)
        download_dust_grid(dest if dest is not None else TEST_GRID_DIR)
        download_camels_data(dest if dest is not None else TEST_DATA_DIR)
        download_sc_sam_test_data(dest if dest is not None else TEST_DATA_DIR)
        download_instruments(INSTRUMENT_CACHE_DIR, AVAILABLE_INSTRUMENTS)
        return

    # Test data?
    if test:
        download_test_grids(dest if dest is not None else TEST_GRID_DIR)
    else:
        if stellar:
            download_stellar_test_grids(
                dest if dest is not None else TEST_GRID_DIR
            )
        if agn:
            download_agn_test_grids(
                dest if dest is not None else TEST_GRID_DIR
            )

    # Instruments?
    if len(instruments) > 0:
        download_instruments(INSTRUMENT_CACHE_DIR, instruments)

    # All instruments?
    if all_instruments:
        download_instruments(INSTRUMENT_CACHE_DIR, AVAILABLE_INSTRUMENTS)

    # Dust data?
    if dust:
        download_dust_grid(dest if dest is not None else TEST_GRID_DIR)

    # Camels data?
    if camels:
        download_camels_data(dest if dest is not None else TEST_DATA_DIR)

    # SC-SAM data?
    if scsam:
        download_sc_sam_test_data(dest if dest is not None else TEST_DATA_DIR)

    # All simulation data?
    if all_sim_data:
        download_camels_data(dest if dest is not None else TEST_DATA_DIR)
        download_sc_sam_test_data(dest if dest is not None else TEST_DATA_DIR)


if __name__ == "__main__":
    download()
