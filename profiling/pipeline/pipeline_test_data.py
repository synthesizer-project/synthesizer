"""Helper module to build test galaxies and instruments for profiling.

This module provides factory functions to create reproducible galaxies,
instruments, and emission models for Pipeline profiling runs. Functions
work offline (no network calls) and produce consistent results given the
same random seed.
"""

from __future__ import annotations

import numpy as np
from unyt import Msun, Myr, km, kpc, s

from synthesizer.emission_models import PacmanEmission
from synthesizer.grid import Grid
from synthesizer.instruments.premade import JWSTNIRCamWide
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars


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
        grid (Grid):
            The SPS grid to use for sampling ages/metallicities.
        nparticles (int):
            Number of stellar particles per galaxy.
        ngalaxies (int):
            Number of galaxies to create.
        seed (int):
            Random seed for reproducibility. Default is 42.

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
        )
        galaxies.append(galaxy)

    return galaxies


def get_test_instruments(grid: Grid):
    """Get a standard set of instruments for profiling.

    This function returns instruments suitable for:
        - Photometry (filters)
        - Imaging (filters + resolution)
        - Spectroscopy (wavelength array)
        - Resolved spectroscopy (wavelength array + resolution)

    All instruments use the grid wavelength array and work offline (no SVO).

    Args:
        grid (Grid):
            The SPS grid to use for filter resampling.

    Returns:
        dict: Dictionary with keys:
            - 'photometry': Instrument for photometry/imaging
            - 'spectroscopy': Instrument for unresolved spectroscopy
            - 'ifu': Instrument for resolved spectroscopy (data cubes)
    """
    # Photometry instrument (JWST NIRCam Wide) - no resolution for photometry
    photometry_inst = JWSTNIRCamWide(
        filter_lams=grid.lam,
        label="JWST.NIRCam.Wide",
    )

    # Imaging instrument (same filters, but physical resolution)
    from synthesizer.instruments import Instrument

    imaging_inst = Instrument(
        label="JWST.NIRCam.Wide.Imaging",
        filters=photometry_inst.filters,
        resolution=0.1 * kpc,  # Physical resolution for imaging
    )

    # Spectroscopy instrument (just wavelength array, no filters)
    spectroscopy_inst = Instrument(
        label="Spectrometer",
        lam=grid.lam,
    )

    # IFU instrument (wavelength + resolution for data cubes)
    ifu_inst = Instrument(
        label="IFU",
        lam=grid.lam,
        resolution=0.1 * kpc,  # Physical resolution for data cubes
    )

    return {
        "photometry": photometry_inst,
        "imaging": imaging_inst,
        "spectroscopy": spectroscopy_inst,
        "ifu": ifu_inst,
    }


def get_test_kernel():
    """Get the default SPH kernel for imaging.

    Returns:
        np.ndarray: The kernel array from Kernel().get_kernel().
    """
    kernel = Kernel()  # Default is sph_anarchy
    return kernel.get_kernel()


def get_test_emission_model(grid: Grid):
    """Get a standard PacmanEmission model for profiling.

    Args:
        grid (Grid):
            The SPS grid to use.

    Returns:
        EmissionModel: A PacmanEmission model with typical parameters.
    """
    return PacmanEmission(
        grid,
        tau_v=0.5,
        fesc=0.0,
        fesc_ly_alpha=1.0,
    )
