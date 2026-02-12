"""Helper module to build test galaxies and instruments for profiling.

This module provides factory functions to create reproducible galaxies,
instruments, and emission models for Pipeline profiling runs. Functions
work offline (no network calls) and produce consistent results given the
same random seed.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
from unyt import Msun, Myr, km, kpc, s

from synthesizer.emission_models import PacmanEmission
from synthesizer.grid import Grid
from synthesizer.instruments.premade import JWSTNIRCamWide
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars

# Define the instrument file path
INSTRUMENT_PATH = "jwst_pipeline_perf_inst.hdf5"


def build_test_galaxies(
    grid: Grid,
    nparticles: int,
    ngalaxies: int,
    seed: int = 42,
) -> list[Galaxy]:
    """Build a list of random particle galaxies for profiling.

    Each galaxy will have stars, gas, and black holes with:
        - Random masses, ages, metallicities
        - Coordinates centered at origin
        - Velocities for Doppler shifts
        - Smoothing lengths for imaging

    Args:
        grid (Grid): The SPS grid to use for sampling ages/metallicities.
        nparticles (int): Number of stellar particles per galaxy.
        ngalaxies (int): Number of galaxies to create.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list[Galaxy]: List of Galaxy objects ready for Pipeline processing.
    """
    rng = np.random.default_rng(seed)
    galaxies = []

    for _ in range(ngalaxies):
        # Generate stellar particles
        nstars = nparticles
        star_masses = rng.uniform(1e4, 1e6, nstars) * Msun
        star_ages = rng.uniform(1e6, 1e10, nstars) * Myr
        star_metallicities = rng.uniform(0.001, 0.02, nstars)
        star_coords = rng.normal(0, 10, (nstars, 3)) * kpc
        star_vels = rng.normal(0, 100, (nstars, 3)) * km / s
        star_smls = rng.uniform(0.1, 1.0, nstars) * kpc

        stars = Stars(
            initial_masses=star_masses,
            ages=star_ages,
            metallicities=star_metallicities,
            redshift=1.0,
            coordinates=star_coords,
            velocities=star_vels,
            smoothing_lengths=star_smls,
            tau_v=0.5,  # V-band optical depth for dust attenuation
        )

        # Generate gas particles (fewer than stars)
        ngas = max(1, nstars // 5)
        gas_masses = rng.uniform(1e5, 1e7, ngas) * Msun
        gas_metallicities = rng.uniform(0.001, 0.02, ngas)
        gas_coords = rng.normal(0, 10, (ngas, 3)) * kpc
        gas_vels = rng.normal(0, 100, (ngas, 3)) * km / s
        gas_smls = rng.uniform(0.1, 1.0, ngas) * kpc

        gas = Gas(
            masses=gas_masses,
            metallicities=gas_metallicities,
            redshift=1.0,
            coordinates=gas_coords,
            velocities=gas_vels,
            smoothing_lengths=gas_smls,
            dust_to_metal_ratio=0.3,
        )

        # Generate black hole particles (very few)
        nbh = max(1, nstars // 50)
        bh_masses = rng.uniform(1e6, 1e8, nbh) * Msun
        bh_accretion_rates = rng.uniform(0.1, 10, nbh) * Msun / Myr
        bh_coords = rng.normal(0, 5, (nbh, 3)) * kpc
        bh_vels = rng.normal(0, 100, (nbh, 3)) * km / s
        bh_smls = rng.uniform(0.1, 0.5, nbh) * kpc

        black_holes = BlackHoles(
            masses=bh_masses,
            accretion_rates=bh_accretion_rates,
            redshift=1.0,
            coordinates=bh_coords,
            velocities=bh_vels,
            smoothing_lengths=bh_smls,
        )

        # Create galaxy
        centre = stars.coordinates.mean(axis=0)
        galaxy = Galaxy(
            stars=stars,
            gas=gas,
            black_holes=black_holes,
            redshift=1.0,
            centre=centre,
            tau_v=0.5,  # V-band optical depth for dust attenuation
        )
        galaxies.append(galaxy)

    return galaxies


def get_test_instrument(grid: Grid):
    """Get photometry instrument for profiling.

    Returns the JWST NIRCam Wide instrument which has filters and
    angular resolution, suitable for both photometry and imaging operations.
    The Pipeline automatically handles dependencies (imaging calls photometry
    first).

    Args:
        grid (Grid): The SPS grid to use for filter resampling.

    Returns:
        Instrument: JWST NIRCam Wide instrument.
    """
    # Load the instrument if we can
    if os.path.exists(INSTRUMENT_PATH):
        # Load from file if it exists
        with h5py.File(INSTRUMENT_PATH, "r") as hdf:
            photometry_inst = JWSTNIRCamWide.from_hdf5(hdf)
        return photometry_inst

    # Otherwise, create a new instance and save it for future use
    photometry_inst = JWSTNIRCamWide(
        filter_lams=grid.lam,
        label="JWST.NIRCam.Wide",
    )

    # Save the instrument to file for future runs
    with h5py.File(INSTRUMENT_PATH, "w") as hdf:
        photometry_inst.to_hdf5(hdf)

    return photometry_inst


def get_test_kernel():
    """Get the default SPH kernel for imaging.

    Returns:
        np.ndarray: The kernel array from Kernel().get_kernel().
    """
    kernel = Kernel()  # Default is sph_anarchy
    return kernel.get_kernel()


def get_test_emission_model(grid: Grid):
    """Get a standard PacmanEmission model for profiling.

    Uses PacmanEmission to test dust attenuation with imaging operations.

    Args:
        grid (Grid): The SPS grid to use.

    Returns:
        EmissionModel: A PacmanEmission model with escape fraction.
    """
    model = PacmanEmission(grid=grid, tau_v="tau_v", fesc="fesc")
    model.set_per_particle(True)  # Per-particle emissions
    return model
