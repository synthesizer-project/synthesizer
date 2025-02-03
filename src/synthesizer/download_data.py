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

import requests
from tqdm import tqdm

from synthesizer import exceptions

# Define all the available files and their information
AVAILABLE_FILES = {
    "test_grid.hdf5": {
        "file": "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5",
        "id": "ywu3dy73cdezohvytyb9k",
        "rlkey": "05agbbdrmxytsc2x1x2jgh3xk",
    },
    "test_grid_agn-blr.hdf5": {
        "file": "agnsed-limited_cloudy-c23.01-blr.hdf5",
        "id": "r7pbdvbvujypgx8ady6bl",
        "rlkey": "4tdscxnoaepvog8skil15ehgk",
    },
    "test_grid_agn-nlr.hdf5": {
        "file": "agnsed-limited_cloudy-c23.01-nlr.hdf5",
        "id": "7h971875rkkmkxvmgdqnn",
        "rlkey": "e6oyr8l9gyqlz3i2nlko7pne6",
    },
    "MW3.1.hdf5": {
        "file": "MW3.1.hdf5",
        "id": "jidw4cgtf95x3gjvw4hj6",
        "rlkey": "z7sbb7z5253dt90ootr5hm5jv",
    },
    "camels_snap.hdf5": {
        "file": "camels_snap.hdf5",
        "id": "c44wvkjm5pqsxpsl54oq0",
        "rlkey": "j14smjen4osffhlyif1kz00bu",
    },
    "camels_subhalo.hdf5": {
        "file": "camels_subhalo.hdf5",
        "id": "srjaltgac4e2tsrxmxrdb",
        "rlkey": "ov0icvv7znw9ybfr31h133jiq",
    },
}


def _download_from_xcs_host(filename, save_dir):
    """
    Download the file from the XCS server.

    Args:
        filename (str)
            The name of the file to download.
        save_dir (str)
            The directory in which to save the file.
    """
    # Define the base URL
    xcs_url = (
        "https://xcs-host.phys.sussex.ac.uk/html/sym_links/synthesizer_data/"
    )

    # Define the full URL
    url = xcs_url + AVAILABLE_FILES[filename]["file"]

    # Define the save path
    save_path = f"{save_dir}/{filename}"

    # Download the file
    response = requests.get(url, stream=True, timeout=10)

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


def _download_from_dropbox(filename, save_dir):
    """
    Download the file from the Dropbox server.

    Args:
        filename (str)
            The name of the file to download.
        save_dir (str)
            The directory in which to save the file.
    """
    # Define the base URL
    dropbox_url = "https://www.dropbox.com/scl/fi/"

    # Unpack the file details for extraction
    file_details = AVAILABLE_FILES[filename]

    # Define the full URL
    url = (
        f"{dropbox_url}/{file_details['id']}/{file_details['file']}"
        f"?rlkey={file_details['rlkey']}&dl=1"
    )

    # Define the save path
    save_path = f"{save_dir}/{filename}"

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


def _download(
    filename,
    save_dir,
):
    """
    Download the file at the given URL to the given path.

    Args:
        filename (str)
            The name of the file to download.
        save_dir (str)
            The directory in which to save the file.
    """
    # Download from the dropbox
    _download_from_dropbox(filename, save_dir)


def download_test_grids(destination):
    """
    Download the test grids for synthesizer.

    Args:
        destination (str)
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
    """
    Download the SPS test grids for synthesizer.

    Args:
        destination (str)
            The path to the destination directory.
    """
    # Download the stellar grid
    _download("test_grid.hdf5", destination)


def download_agn_test_grids(destination):
    """
    Download the AGN test grids for synthesizer.

    Args:
        destination (str)
            The path to the destination directory.
    """
    # Download each file
    for f in ["test_grid_agn-blr.hdf5", "test_grid_agn-nlr.hdf5"]:
        _download(f, destination)


def download_dust_grid(destination):
    """
    Download the Drain and Li (2007) dust emission grid for synthesizer.

    Args:
        destination (str)
            The path to the destination directory.
    """
    # Download the dust grid
    _download("MW3.1.hdf5", destination)


def download_camels_data(destination):
    """
    Download the CAMELS data.

    Args:
        destination (str)
            The path to the destination directory.
    """
    _download("camels_snap.hdf5", destination)
    _download("camels_subhalo.hdf5", destination)


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
        required=True,
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

    # Are we just getting everything?
    if everything:
        download_test_grids(dest)
        download_dust_grid(dest)
        download_camels_data(dest)
        return

    # Test data?
    if test:
        download_test_grids(dest)
    else:
        if stellar:
            download_stellar_test_grids(dest)
        if agn:
            download_agn_test_grids(dest)

    # Dust data?
    if dust:
        download_dust_grid(dest)

    # Camels data?
    if camels:
        download_camels_data(dest)


if __name__ == "__main__":
    download()
