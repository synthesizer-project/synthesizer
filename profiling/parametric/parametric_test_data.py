"""Shared setup for the parametric profiling scripts.

A "population" here is one parametric stellar component with its own SFZH
and morphology, e.g. a bulge, a disk or one annulus of a resolved galaxy.
Until parametric Stars can hold multiple populations, each population is
an independent Stars object and every operation loops over them.
"""

import sys

import numpy as np
from unyt import Gyr, Msun, kpc, unyt_array

from synthesizer.parametric import SFH, Sersic2D, Stars, ZDist

# The population counts profiled, spanning a single component, bulge+disk
# and finely resolved annuli
NPOPS = np.array([1, 2, 5, 10, 20, 50, 100])


def make_populations(grid, npops, seed=42):
    """Build npops parametric populations on the grid axes.

    Each population has a delayed exponential SFH, a normal metallicity
    distribution and a Sersic morphology with randomly drawn parameters.

    Args:
        grid (Grid):
            The grid whose axes define the SFZH.
        npops (int):
            The number of populations to make.
        seed (int):
            The random seed.

    Returns:
        list of Stars:
            The populations.
    """
    rng = np.random.default_rng(seed)
    pops = []
    for _ in range(npops):
        pops.append(
            Stars(
                grid.log10ages,
                grid.metallicities,
                sf_hist=SFH.DelayedExponential(
                    tau=rng.uniform(0.1, 3) * Gyr,
                    max_age=rng.uniform(0.5, 10) * Gyr,
                ),
                metal_dist=ZDist.Normal(
                    mean=rng.uniform(0.002, 0.02),
                    sigma=rng.uniform(0.001, 0.005),
                ),
                initial_mass=10 ** rng.uniform(8, 10) * Msun,
                morphology=Sersic2D(
                    r_eff=rng.uniform(0.5, 5) * kpc,
                    sersic_index=rng.uniform(1, 4),
                    x_0=0 * kpc,
                    y_0=0 * kpc,
                ),
            )
        )
    return pops


def get_obj_size_manual(obj):
    """Estimate object size in bytes, handling nested dicts and numpy arrays.

    This avoids bugs in pympler.asizeof with certain numpy versions.
    """
    size = 0
    if isinstance(obj, dict):
        for v in obj.values():
            size += get_obj_size_manual(v)
    elif isinstance(obj, (np.ndarray, unyt_array)):
        size += obj.nbytes
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            size += get_obj_size_manual(v)
    else:
        size += sys.getsizeof(obj)
    return size
