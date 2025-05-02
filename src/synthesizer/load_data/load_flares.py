"""A submodule for loading FLARES data into Synthesizer.

Example usage:

    from synthesizer.load_data import load_FLARES

    # Load FLARES data
    galaxies = load_FLARES(
        master_file="path/to/master_file.hdf5",
        region="region_name",
        tag="snapshot_tag",
        read_abundances=True,
    )
"""

import h5py
import numpy as np
from unyt import Mpc, Msun, yr

from synthesizer.load_data.utils import get_begin_end_pointers

from ..particle.galaxy import Galaxy


def load_FLARES(master_file, region, tag, read_abundances=False):
    """Load FLARES galaxies from a FLARES master file.

    Args:
        master_file (str):
            The path to the master file.
        region (str):
            The region to load data from.
        tag (str):
            The snapshot tag to load data from.
        read_abundances (bool):
            Whether to read the abundances of the stars.
            If True, the oxygen and hydrogen abundances are loaded.
            If False, only the metallicity is loaded.

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing stars
    """
    zed = float(tag[5:].replace("p", "."))
    scale_factor = 1.0 / (1.0 + zed)

    with h5py.File(master_file, "r") as hf:
        slens = hf[f"{region}/{tag}/Galaxy/S_Length"][:]
        glens = hf[f"{region}/{tag}/Galaxy/G_Length"][:]
        cop = hf[f"{region}/{tag}/Galaxy/COP"][:]  # loading COP

        ages = hf[f"{region}/{tag}/Particle/S_Age"][:]  # Gyr
        coods = (
            hf[f"{region}/{tag}/Particle/S_Coordinates"][:].T * scale_factor
        )  # Mpc (physical)
        masses = hf[f"{region}/{tag}/Particle/S_Mass"][:]  # 1e10 Msol
        imasses = hf[f"{region}/{tag}/Particle/S_MassInitial"][:]  # 1e10 Msol
        s_hsml = hf[f"{region}/{tag}/Particle/S_sml"][:]  # Mpc (physical)

        metallicities = hf[f"{region}/{tag}/Particle/S_Z_smooth"][:]

        if read_abundances:
            s_oxygen = hf[f"{region}/{tag}/Particle/S_Abundance_Oxygen"][:]
            s_hydrogen = hf[f"{region}/{tag}/Particle/S_Abundance_Hydrogen"][:]

        g_sfr = hf[f"{region}/{tag}/Particle/G_SFR"][:]  # Msol / yr
        g_masses = hf[f"{region}/{tag}/Particle/G_Mass"][:]  # 1e10 Msol
        g_metallicities = hf[f"{region}/{tag}/Particle/G_Z_smooth"][:]
        g_coods = (
            hf[f"{region}/{tag}/Particle/G_Coordinates"][:].T * scale_factor
        )  # Mpc (physical)
        g_hsml = hf[f"{region}/{tag}/Particle/G_sml"][:]  # Mpc (physical)

    # Convert units
    ages = ages * 1e9  # yr
    masses = masses * 1e10  # Msol
    imasses = imasses * 1e10  # Msol
    g_masses = g_masses * 1e10  # Msol

    # Get the star particle begin / end indices
    begin, end = get_begin_end_pointers(slens)

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        # Create the individual galaxy objects
        galaxies[i] = Galaxy(redshift=zed)
        galaxies[i].centre = (
            np.array([cop[0][i], cop[1][i], cop[2][i]]) * scale_factor * Mpc
        )
        if read_abundances:
            galaxies[i].load_stars(
                imasses[b:e] * Msun,
                ages[b:e] * yr,
                metallicities[b:e],
                s_oxygen=s_oxygen[b:e],
                s_hydrogen=s_hydrogen[b:e],
                coordinates=coods[b:e, :] * Mpc,
                current_masses=masses[b:e] * Msun,
                smoothing_lengths=s_hsml[b:e] * Mpc,
            )
        else:
            galaxies[i].load_stars(
                imasses[b:e] * Msun,
                ages[b:e] * yr,
                metallicities[b:e],
                coordinates=coods[b:e, :] * Mpc,
                current_masses=masses[b:e] * Msun,
                smoothing_lengths=s_hsml[b:e] * Mpc,
            )

    # Get the gas particle begin / end indices
    begin, end = get_begin_end_pointers(glens)

    for i, (b, e) in enumerate(zip(begin, end)):
        # Use gas particle SFR for star forming mask
        sf_mask = g_sfr[b:e] > 0
        galaxies[i].sf_gas_mass = np.sum(g_masses[b:e][sf_mask]) * Msun

        galaxies[i].sf_gas_metallicity = (
            np.sum(g_masses[b:e][sf_mask] * g_metallicities[b:e][sf_mask])
            / galaxies[i].sf_gas_mass.value
        )

        galaxies[i].load_gas(
            coordinates=g_coods[b:e] * Mpc,
            masses=g_masses[b:e] * Msun,
            metallicities=g_metallicities[b:e],
            smoothing_lengths=g_hsml[b:e] * Mpc,
        )

    return galaxies
