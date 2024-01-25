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


def _download(url, save_dir):
    """
    Download the file at the given URL to the given path.

    Args:
        url (str)
            The url to download from.
        save_dir (str)
            The directory in which to save the file.
    """
    # Get the file name
    filename = url.split("/")[-1]

    # Define the save path
    save_path = f"{save_dir}/{filename}"

    # Download the file
    response = requests.get(url, stream=True)

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
    """
    Download the test grids for synthesizer.

    Args:
        destination (str)
            The path to the destination directory.
    """
    base_url = (
        "https://xcs-host.phys.sussex.ac.uk/html/sym_links/synthesizer_data/"
    )

    # Define the files to get
    files = [
        "test_grid.hdf5",
        "test_grid_agn-blr.hdf5",
        "test_grid_agn-nlr.hdf5",
    ]

    # Download each file
    for f in files:
        _download(base_url + f, destination)


def download_dust_grid(destination):
    """
    Download the Drain and Li (2007) dust emission grid for synthesizer.

    Args:
        destination (str)
            The path to the destination directory.
    """
    base_url = (
        "https://xcs-host.phys.sussex.ac.uk/html/sym_links/synthesizer_data/"
    )

    raise exceptions.UnimplementedFunctionality(
        "DLi+07 grid not yet available through https."
    )

    # Download each file
    _download(base_url + "MW3.1.hdf5", destination)


def download_camels_data(snap, lh, destination):
    """
    Download a CAMELs dataset.

    Args:
        snap (str)
            The snapshot tag to download.
        lh (str)
            The LH variant tag of the sim to download.
        destination (str)
            The path to the destination directory.
    """
    # Convert lh
    lh = str(lh)
    raise exceptions.UnimplementedFunctionality(
        "CAMELS data is not yet available for download."
    )


def download():
    """Download different datasets based on command line args."""
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Download the datasets for synthesizer."
    )

    # Add a flag to handle the test data
    parser.add_argument(
        "--test-grids",
        "-t",
        action="store_true",
        help="Download the test data for synthesizer.",
    )

    # Add the flag for dust data
    parser.add_argument(
        "--dust-grid",
        "-D",
        action="store_true",
        help="Download the dust grid for the Drain & Li (2007) model",
    )

    # Add the flag for camels data (this requires related arguments to define
    # exactly which dataset to download)
    parser.add_argument(
        "--camels-data",
        "-c",
        action="store_true",
        help="Download the CAMELS dataset.",
    )

    # Add the CAMELs arguments
    parser.add_argument(
        "--camels-snap",
        type=str,
        help="Which snapshot should be downloaded? (Default: 031)",
        default="031",
    )
    parser.add_argument(
        "--camels-lh",
        type=int,
        help="Which LH variant should be downloaded? (Default: 1)",
        default=1,
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
    dust = args.dust_grid
    camels = args.camels_data
    everything = args.all
    dest = args.destination

    # Are we just getting everything?
    if everything:
        download_test_grids(dest)
        download_dust_grid(dest)
        download_camels_data(args.camels_snap, args.camels_lh, dest)
        return

    # Test data?
    if test:
        download_test_grids(dest)

    # Dust data?
    if dust:
        download_dust_grid(dest)

    # Camels data?
    if camels:
        download_camels_data(args.camels_snap, args.camels_lh, dest)


if __name__ == "__main__":
    download()
