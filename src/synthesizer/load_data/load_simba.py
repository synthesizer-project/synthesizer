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
    load_halo=False,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Load Simba galaxy data from a caesar file and snapshot.

    Args:
        directory (string):
            data location
        snap_name (string):
            snapshot filename
        caesar_name (string):
            Subfind / FOF filename
        caesar_directory (string):
            optional argument specifying location of fof file
            if different to snapshot
        load_halo (bool, false):
            optional argument, whether to load galaxy or halo objects.
            If False (default), will load individual galaxies.
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (float):
            Scale factor resolution of the age lookup

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particles
    """
    with h5py.File(f"{directory}/{snap_name}", "r") as hf:
        scale_factor = hf["Header"].attrs["Time"]
        Om0 = hf["Header"].attrs["Omega0"]
        h = hf["Header"].attrs["HubbleParam"]

        form_time = hf["PartType4/StellarFormationTime"][:]
        coods = hf["PartType4/Coordinates"][:] * scale_factor  # Mpc (physical)
        masses = hf["PartType4/Masses"][:]

        imasses = np.ones(len(masses)) * 0.00155
        # * hf['Header'].attrs['MassTable'][1]

        _metals = hf["PartType4/Metallicity"][:]

        # g_sfr = hf["PartType0/StarFormationRate"][:]
        g_h2fraction = hf["PartType0/FractionH2"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/Metallicity"][:][:, 0]
        g_coods = (
            hf["PartType0/Coordinates"][:] * scale_factor
        )  # Mpc (physical)
        g_hsml = hf["PartType0/SmoothingLength"][:]  # Mpc

        g_dustmass = hf["PartType0/Dust_Masses"][:]

    redshift = (1.0 / scale_factor) - 1

    # convert units
    masses = (masses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    g_dustmass = (g_dustmass * 1e10) / h
    imasses = (imasses * 1e10) / h

    # get individual and summed metallicity components
    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metallicity = _metals[:, 0]

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
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

    # check which kind of object we're loading
    if load_halo:
        obj_str = "halo_data"
    else:
        obj_str = "galaxy_data"

    if caesar_directory is None:
        caesar_directory = directory

    # get the star particle begin / end indices
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        begin = hf[f"{obj_str}/slist_start"][:]
        end = hf[f"{obj_str}/slist_end"][:]

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        # create the individual galaxy objects
        galaxies[i] = Galaxy()
        galaxies[i].redshift = redshift
        galaxies[i].load_stars(
            initial_masses=imasses[b:e] * Msun,
            ages=ages[b:e] * yr,
            metallicities=metallicity[b:e],
            s_oxygen=s_oxygen[b:e],
            s_hydrogen=s_hydrogen[b:e],
            coordinates=coods[b:e, :] * kpc,
            current_masses=masses[b:e] * Msun,
        )

    # get the gas particle begin / end indices
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        begin = hf[f"{obj_str}/glist_start"][:]
        end = hf[f"{obj_str}/glist_end"][:]

    for i, (b, e) in enumerate(zip(begin, end)):
        # Use the H2 masses computed in Simba directly to
        # estimate the star forming gas mass and metallicity.
        # Alternative to setting star_forming property on
        # each gas particle
        h2_masses = g_masses[b:e] * g_h2fraction[b:e]
        galaxies[i].sf_gas_mass = np.sum(h2_masses) * Msun
        galaxies[i].sf_gas_metallicity = (
            np.sum(h2_masses * g_metals[b:e]) / galaxies[i].sf_gas_mass
        )

        galaxies[i].load_gas(
            coordinates=g_coods[b:e] * kpc,
            masses=g_masses[b:e] * Msun,
            metallicities=g_metals[b:e],
            smoothing_lengths=g_hsml[b:e] * kpc,
            dust_masses=g_dustmass[b:e] * Msun,
        )

    return galaxies
