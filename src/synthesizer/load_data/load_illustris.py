"""A submodule for loading IllustrisTNG data into Synthesizer.

This module provides a function to load particle data from the IllustrisTNG
simulation into galaxy objects. The data is loaded from the group and particle
files, and the particles associated with each subhalo are loaded individually.

The module requires the `illustris_python` module, which must be installed
manually. The `load_IllustrisTNG` function loads the particles associated with
each subhalo individually, rather than the whole simulation volume particle
arrays; this can be slower for certain volumes, but avoids memory issues for
higher resolution boxes.

Example usage::

    galaxies, subhalo_mask = load_IllustrisTNG(
        directory="path/to/data",
        snap_number=99,
        stellar_mass_limit=8.5e6,
        verbose=True,
        dtm=0.3,
        physical=True,
        metals=True,
    )
"""

import numpy as np
from astropy.cosmology import Planck15
from tqdm import tqdm
from unyt import Msun, kpc, yr

from synthesizer.exceptions import UnmetDependency
from synthesizer.load_data.utils import age_lookup_table, lookup_age

try:
    import illustris_python as il
except ImportError:
    raise UnmetDependency(
        "The `illustris_python` module is required to load IllustrisTNG data. "
        "Install it via `pip install 'illustris_python @ "
        "git+https://github.com/illustristng/illustris_python.git@master'"
    )


from ..particle.galaxy import Galaxy


def load_IllustrisTNG(
    directory=".",
    snap_number=99,
    stellar_mass_limit=8.5e6,
    verbose=True,
    dtm=0.3,
    physical=True,
    metals=True,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Load IllustrisTNG particle data into galaxy objects.

    Uses the `illustris_python` module, which must be installed manually.
    Loads the particles associated with each subhalo individually, rather than
    the whole simulation volume particle arrays; this can be slower for certain
    volumes, but avoids memory issues for higher resolution boxes.

    Args:
        directory (string):
            Data location of group and particle files.
        snap_number (int):
            Snapshot number.
        stellar_mass_limit (float):
            Stellar mass limit above which to load galaxies.
            In units of solar mass.
        verbose (bool):
            Verbosity flag
        dtm (float):
            Dust-to-metals ratio to apply to all gas particles
        physical (bool):
            Should the coordinates be converted to physical?
            Default: True
        metals (bool):
            Should we load individual stellar element abundances?
            Default: True. This property may not be available for
            all snapshots ( see
            https://www.tng-project.org/data/docs/specifications/#sec1a ).
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (float):
            Scale factor resolution of the age lookup

    Returns:
        galaxies (list):
            List of `ParticleGalaxy` objects, each containing star and
            gas particles
        subhalo_mask (array, bool):
            Boolean array of selected galaxies from the subhalo catalogue.
    """
    # Do some simple argument preparation
    snap_number = int(snap_number)

    if verbose:
        print("Loading header information...")

    # Get header information
    header = il.groupcat.loadHeader(directory, snap_number)
    scale_factor = header["Time"].astype(np.float32)
    redshift = header["Redshift"].astype(np.float32)
    h = header["HubbleParam"]

    if verbose:
        print("Loading subhalo catalogue...")

    # Load subhalo properties (positions and stellar masses)
    fields = ["SubhaloMassType", "SubhaloPos"]
    output = il.groupcat.loadSubhalos(directory, snap_number, fields=fields)

    # Perform stellar mass masking
    stellar_mass = output["SubhaloMassType"][:, 4]
    subhalo_mask = (stellar_mass * 1e10) > stellar_mass_limit

    subhalo_pos = output["SubhaloPos"][subhalo_mask]

    if verbose:
        print(
            f"Loaded {np.sum(subhalo_mask)} galaxies "
            f"above the stellar mass cut ({stellar_mass_limit} M_solar)"
        )

    galaxies = [None] * np.sum(subhalo_mask)

    if verbose:
        print("Loading particle information...")

    for i, (idx, pos) in tqdm(
        enumerate(zip(np.where(subhalo_mask)[0], subhalo_pos)),
        total=np.sum(subhalo_mask),
    ):
        galaxies[i] = Galaxy(verbose=False)
        galaxies[i].redshift = redshift

        if physical:
            pos *= scale_factor

        # Save subhalo centre
        galaxies[i].centre = pos * kpc

        star_fields = [
            "GFM_StellarFormationTime",
            "Coordinates",
            "Masses",
            "GFM_InitialMass",
            "GFM_Metallicity",
            "SubfindHsml",
        ]
        if metals:
            star_fields.append("GFM_Metals")

        output = il.snapshot.loadSubhalo(
            directory, snap_number, idx, "stars", fields=star_fields
        )

        if output["count"] > 0:
            # Mask for wind particles
            mask = output["GFM_StellarFormationTime"] <= 0.0

            # filter particle arrays
            imasses = output["GFM_InitialMass"][~mask]
            form_time = output["GFM_StellarFormationTime"][~mask]
            coods = output["Coordinates"][~mask]
            metallicities = output["GFM_Metallicity"][~mask]
            masses = output["Masses"][~mask]
            hsml = output["SubfindHsml"][~mask]

            masses = (masses * 1e10) / h
            imasses = (imasses * 1e10) / h

            if metals:
                _metals = output["GFM_Metals"][~mask]
                s_oxygen = _metals[:, 4]
                s_hydrogen = 1.0 - np.sum(_metals[:, 1:], axis=1)
            else:
                s_oxygen = None
                s_hydrogen = None

            # Convert comoving coordinates to physical kpc
            if physical:
                coods *= scale_factor
                hsml *= scale_factor

            # convert formation times to ages
            cosmo = Planck15
            universe_age = cosmo.age(1.0 / scale_factor - 1)

            # Are we creating a lookup table for ages?
            if age_lookup:
                # Create the lookup grid
                scale_factors, ages = age_lookup_table(
                    cosmo,
                    redshift=redshift,
                    delta_a=age_lookup_delta_a,
                    low_lim=1e-4,
                )

                # Look up the ages for the particles
                _ages = lookup_age(form_time, scale_factors, ages)
            else:
                # Calculate ages of these explicitly using astropy
                _ages = cosmo.age(1.0 / form_time - 1)

            # Calculate ages at snapshot redshift
            ages = (universe_age - _ages).value * 1e9  # yr

            if hsml is None:
                smoothing_lengths = hsml
            else:
                smoothing_lengths = hsml * kpc

            galaxies[i].load_stars(
                initial_masses=imasses * Msun,
                ages=ages * yr,
                metallicities=metallicities,
                s_oxygen=s_oxygen,
                s_hydrogen=s_hydrogen,
                coordinates=coods * kpc,
                current_masses=masses * Msun,
                smoothing_lengths=smoothing_lengths,
            )

        gas_fields = [
            "StarFormationRate",
            "Coordinates",
            "Masses",
            "GFM_Metallicity",
            "SubfindHsml",
        ]
        output = il.snapshot.loadSubhalo(
            directory, snap_number, idx, "gas", fields=gas_fields
        )

        if output["count"] > 0:
            g_masses = output["Masses"]
            g_sfr = output["StarFormationRate"]
            g_coods = output["Coordinates"]
            g_hsml = output["SubfindHsml"]
            g_metals = output["GFM_Metallicity"]

            g_masses = (g_masses * 1e10) / h
            star_forming = g_sfr > 0.0  # star forming gas particles

            # Convert comoving coordinates to physical kpc
            if physical:
                coods *= scale_factor
                g_coods *= scale_factor
                g_hsml *= scale_factor

            galaxies[i].load_gas(
                coordinates=g_coods * kpc,
                masses=g_masses * Msun,
                metallicities=g_metals,
                star_forming=star_forming,
                smoothing_lengths=g_hsml * kpc,
                dust_to_metal_ratio=dtm,
            )

    return galaxies, subhalo_mask
