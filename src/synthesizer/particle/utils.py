"""A submodule containing utility functions for particle distributions."""

import numpy as np
from scipy.spatial import cKDTree
from unyt import Mpc, rad, unyt_array

from synthesizer.units import accepts


@accepts(phi=rad, theta=rad)
def rotate(
    coordinates,
    phi=0 * rad,
    theta=0 * rad,
    rot_matrix=None,
):
    """Rotate 3D array of coordinates.

    Args:
        coordinates (np.ndarray of float):
            1D or 2D array of coordinates
        phi (unyt_quantity):
            The angle in radians to rotate around the z-axis. If rot_matrix
            is defined this will be ignored.
        theta (unyt_quantity):
            The angle in radians to rotate around the y-axis. If rot_matrix
            is defined this will be ignored.
        rot_matrix (array-like, float):
            A 3x3 rotation matrix to apply to the coordinates
            instead of phi and theta.

    Returns:
        coordinates (np.array)
            transformed coordinate array
    """
    # if len(coordinates.shape) == 1:
    #     coordinates = np.array([coordinates])

    # Are we using angles?
    if rot_matrix is None:
        # Rotation matrix around z-axis (phi)
        rot_matrix_z = np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )

        # Rotation matrix around y-axis (theta)
        rot_matrix_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

        # Combined rotation matrix
        rot_matrix = np.dot(rot_matrix_y, rot_matrix_z)

    return np.dot(coordinates, rot_matrix.T)


@accepts(coordinates=Mpc, source_coords=Mpc, boxsize=Mpc)
def calculate_smoothing_lengths(
    coordinates: unyt_array,
    source_coords: unyt_array = None,
    kernel_gamma: np.float32 = 1.4,
    num_neighbours: int = 32,
    speedup_fac: int = 2,
    dimension: int = 3,
    boxsize: unyt_array = None,
    workers: int = 1,
) -> unyt_array[Mpc]:
    """Calculate smoothing lengths based on the kth nearest neighbour distance.

    This is approximately what is done in SPH codes to calculate the smoothing
    length for each particle.

    Adapted from the Swiftsimio implementation
    (https://github.com/SWIFTSIM/swiftsimio)

    Args:
        coordinates (unyt_array):
            The coordinates to calculate the smoothing lengths for.
        source_coords (unyt_array, optional):
            The source coordinates to query against. If None, smoothing
            lengths are calculated from `coordinates` themselves.
        kernel_gamma (float, optional):
            The kernel gamma of the kernel being used. (default: 1.4)
        num_neighbours (int, optional):
            The number of neighbours to encompass.
        speedup_fac (int, optional):
            A parameter that neighbours is divided by to provide a speed-up
            by only searching for a lower number of neighbours. For example,
            if neighbours is 32, and speedup_fac is 2, we only search for 16
            (32 / 2) neighbours, and extend the smoothing length out to
            (speedup)**(1/dimension) such that we encompass an approximately
            higher number of neighbours. A factor of 2 gives smoothing lengths
            the same as the full search within 10%, good enough for
            visualisation.
        dimension (int, optional):
            The dimensionality of the problem (used for speedup_fac
            calculation).
        boxsize (unyt_array, optional):
            The boxsize to use for the periodic boundary conditions. If None,
            no periodic boundary conditions are used
        workers (int, optional):
            The number of workers to use in the neighbour search. The default
            is 1.

    Returns:
        unyt_array: An unyt array of smoothing lengths.
    """
    nparts: int = coordinates.shape[0]
    source_coords = coordinates if source_coords is None else source_coords
    query_coords = coordinates.value
    source_values = source_coords.to(coordinates.units).value

    if num_neighbours < 1:
        raise ValueError("num_neighbours must be at least 1.")

    if speedup_fac < 1:
        raise ValueError("speedup_fac must be at least 1.")

    # Build the tree (with or without periodic boundary conditions)
    tree: cKDTree
    if boxsize is None:
        tree = cKDTree(source_values)
    else:
        tree = cKDTree(
            source_values, boxsize=boxsize.to(coordinates.units).value
        )

    smoothing_lengths: np.ndarray = np.empty(nparts, dtype=np.float32)

    # Include speedup_fac stuff here:
    neighbours_search: int = max(1, num_neighbours // speedup_fac)
    hsml_correction_fac_speedup: float = (speedup_fac) ** (1 / dimension)

    # We create a lot of data doing this, so we want to do it in small chunks
    # such that we keep the memory from filling up - "this seems to be a
    # reasonable chunk size based on previous performance
    # testing." - SWIFTsimio (probably Josh)
    block_size: int = 65536

    for starting_index in range(0, nparts, block_size):
        ending_index = min(starting_index + block_size, nparts)

        # Query the tree for this chunk
        d, _ = tree.query(
            query_coords[starting_index:ending_index],
            k=neighbours_search,
            workers=workers,
        )

        # Store the smoothing lengths
        if np.ndim(d) == 1:
            smoothing_lengths[starting_index:ending_index] = d
        else:
            smoothing_lengths[starting_index:ending_index] = d[:, -1]

    return unyt_array(
        smoothing_lengths * (hsml_correction_fac_speedup / kernel_gamma),
        units=coordinates.units,
    )
