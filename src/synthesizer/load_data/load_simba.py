"""A submodule for loading Simba data into Synthesizer.

Load Simba galaxy data from a caesar file and snapshot.

Method for loading galaxy and particle data for
the [Simba](http://simba.roe.ac.uk/) simulation
"""

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from unyt import Msun, kpc, yr

from synthesizer.load_data.utils import age_lookup_table, lookup_age

from ..particle.galaxy import Galaxy


def load_Simba(
    directory=".",
    snap_name="snap_033.hdf5",
    caesar_name="fof_subhalo_tab_033.hdf5",
    caesar_directory=None,
    object_indexes=False,
    load_halo=False,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Load Simba galaxy data from a caesar file and snapshot.

    This function can load all objects or a specific subset based on their
    indices. It uses the `slist` and `glist` datasets from the Caesar file
    to correctly map particles to their host galaxies/halos.

    Args:
        directory (string):
            Data location.
        snap_name (string):
            Snapshot filename.
        caesar_name (string):
            Caesar file name.
        caesar_directory (string):
            Optional path to the Caesar file if it's in a different
            directory from the snapshot.
        object_indexes (list or bool):
            List of object indices to load. If False, all objects are loaded.
        load_halo (bool):
            If True, loads halo objects instead of galaxy objects.
        age_lookup (bool):
            If True, creates and uses a lookup table for stellar ages for
            faster age calculation.
        age_lookup_delta_a (float):
            The resolution of the age lookup table in terms of scale factor.

    Returns:
        galaxies (list):
            A list of `synthesizer.particle.galaxy.Galaxy` objects.
    """
    # check which kind of object we're loading
    if load_halo:
        obj_str = "halo_data"
    else:
        obj_str = "galaxy_data"

    if caesar_directory is None:
        caesar_directory = directory

    # Open the Caesar file to get particle lists for each galaxy
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        slist_all = hf[f"{obj_str}/lists/slist"][:]
        slist_start_all = hf[f"{obj_str}/slist_start"][:]
        slist_end_all = hf[f"{obj_str}/slist_end"][:]

        glist_all = hf[f"{obj_str}/lists/glist"][:]
        glist_start_all = hf[f"{obj_str}/glist_start"][:]
        glist_end_all = hf[f"{obj_str}/glist_end"][:]

        centres_all = hf[f"{obj_str}/pos"][:]

    # Get the particle indices for the selected galaxies
    if object_indexes is False:
        galaxy_indices = np.arange(len(slist_start_all))

        star_particle_indices = slist_all
        gas_particle_indices = glist_all
    else:
     else:
         galaxy_indices = np.array(object_indexes)
         # Validate indices
         if np.any(galaxy_indices < 0) or np.any(galaxy_indices >= len(slist_start_all)):
             raise ValueError(
                 f"Invalid object indices. Must be in range [0, {len(slist_start_all) - 1}]"
             )
        star_particle_indices = np.concatenate(
            [
                slist_all[slist_start_all[i] : slist_end_all[i]]
                for i in galaxy_indices
            ]
        )
        gas_particle_indices = np.concatenate(
            [
                glist_all[glist_start_all[i] : glist_end_all[i]]
                for i in galaxy_indices
            ]
        )

    centres = centres_all[galaxy_indices]

    with h5py.File(f"{directory}/{snap_name}", "r") as hf:
        scale_factor = hf["Header"].attrs["Time"]
        Om0 = hf["Header"].attrs["Omega0"]
        h = hf["Header"].attrs["HubbleParam"]

        # Load star data for selected particles
        form_time = hf["PartType4/StellarFormationTime"][:][
            star_particle_indices
        ]
        coods = (
            hf["PartType4/Coordinates"][:][star_particle_indices]
            * scale_factor
            / h
        )
        masses = hf["PartType4/Masses"][:][star_particle_indices]
        imasses = np.ones(len(masses)) * 0.00155
        _metals = hf["PartType4/Metallicity"][:][star_particle_indices]

        # Load gas data for selected particles
        g_masses = hf["PartType0/Masses"][:][gas_particle_indices]
        g_metals_all = hf["PartType0/Metallicity"][:][gas_particle_indices]
        g_metals = g_metals_all[:, 0]
        g_coods = (
            hf["PartType0/Coordinates"][:][gas_particle_indices]
            * scale_factor
            / h
        )
        g_hsml = hf["PartType0/SmoothingLength"][:][gas_particle_indices]
        g_dustmass = hf["PartType0/Dust_Masses"][:][gas_particle_indices]
        g_h2fraction = hf["PartType0/FractionH2"][:][gas_particle_indices]

    redshift = (1.0 / scale_factor) - 1

    # Convert units
    masses = (masses * 1e10) / h
    imasses = (imasses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    g_dustmass = (g_dustmass * 1e10) / h

    # Get individual and summed metallicity components for stars
    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metallicity = _metals[:, 0]

    # Convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

    if age_lookup:
        scale_factors, ages_table = age_lookup_table(
            cosmo, redshift=redshift, delta_a=age_lookup_delta_a, low_lim=1e-4
        )
        _ages = lookup_age(form_time, scale_factors, ages_table)
    else:
        _ages = cosmo.age(1.0 / form_time - 1)

    ages = (universe_age - _ages).value * 1e9  # yr

    # Prepare to create Galaxy objects
    galaxies = []
    s_particle_counts = [
        slist_end_all[i] - slist_start_all[i] for i in galaxy_indices
    ]
    g_particle_counts = [
        glist_end_all[i] - glist_start_all[i] for i in galaxy_indices
    ]
    s_offsets = np.concatenate([[0], np.cumsum(s_particle_counts)[:-1]])
    g_offsets = np.concatenate([[0], np.cumsum(g_particle_counts)[:-1]])

    for i, gal_idx in enumerate(galaxy_indices):
        galaxy = Galaxy()
        galaxy.redshift = redshift

        s_start, s_end = s_offsets[i], s_offsets[i] + s_particle_counts[i]
        g_start, g_end = g_offsets[i], g_offsets[i] + g_particle_counts[i]

        galaxy.load_stars(
            initial_masses=imasses[s_start:s_end] * Msun,
            ages=ages[s_start:s_end] * yr,
            metallicities=metallicity[s_start:s_end],
            s_oxygen=s_oxygen[s_start:s_end],
            s_hydrogen=s_hydrogen[s_start:s_end],
            coordinates=coods[s_start:s_end, :] * kpc,
            current_masses=masses[s_start:s_end] * Msun,
            centre=centres[i] * kpc,
        )

        h2_masses = g_masses[g_start:g_end] * g_h2fraction[g_start:g_end]
        galaxy.sf_gas_mass = np.sum(h2_masses) * Msun
        if galaxy.sf_gas_mass > 0:
            galaxy.sf_gas_metallicity = (
                np.sum(h2_masses * g_metals[g_start:g_end])
                / galaxy.sf_gas_mass
            )
        else:
            galaxy.sf_gas_metallicity = 0.0

        galaxy.load_gas(
            coordinates=g_coods[g_start:g_end] * kpc,
            masses=g_masses[g_start:g_end] * Msun,
            metallicities=g_metals[g_start:g_end],
            smoothing_lengths=g_hsml[g_start:g_end] * kpc,
            dust_masses=g_dustmass[g_start:g_end] * Msun,
        )
        galaxies.append(galaxy)

    return galaxies
