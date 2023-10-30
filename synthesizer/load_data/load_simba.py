import h5py
import numpy as np

from astropy.cosmology import FlatLambdaCDM

from ..particle.galaxy import Galaxy


def load_Simba(
    directory=".",
    snap_name="snap_033.hdf5",
    caesar_name="fof_subhalo_tab_033.hdf5",
    caesar_directory=None,
    load_halo=False,
):
    """
    Load CAMELS-SIMBA galaxies

    Args:
        directory (string):
            data location
        snap_name (string):
            snapshot filename
        fof_name (string):
            Subfind / FOF filename
        fof_directory (string):
            optional argument specifying location of fof file
            if different to snapshot
        dtm (float):
            dust to metals ratio for all gas particles

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particle
    """

    with h5py.File(f"{directory}/{snap_name}", "r") as hf:
        form_time = hf["PartType4/StellarFormationTime"][:]
        coods = hf["PartType4/Coordinates"][:]
        masses = hf["PartType4/Masses"][:]

        imasses = np.ones(len(masses)) * 0.00155
        # * hf['Header'].attrs['MassTable'][1]

        _metals = hf["PartType4/Metallicity"][:]

        g_sfr = hf["PartType0/StarFormationRate"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/Metallicity"][:][:, 0]
        g_coods = hf["PartType0/Coordinates"][:]
        g_hsml = hf["PartType0/SmoothingLength"][:]

        g_dustmass = hf["PartType0/Dust_Masses"][:]

        scale_factor = hf["Header"].attrs["Time"]
        Om0 = hf["Header"].attrs["Omega0"]
        h = hf["Header"].attrs["HubbleParam"]

    # convert units
    masses = (masses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    g_dustmass = (g_dustmass * 1e10) / h
    imasses = (imasses * 1e10) / h

    # create mask of star forming gas particles
    star_forming = g_sfr > 0.0

    # get individual and summed metallicity components
    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metallicity = _metals[:, 0]

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(1.0 / scale_factor - 1)
    _ages = cosmo.age(1.0 / form_time - 1)
    ages = (universe_age - _ages).value * 1e9  # yr

    # check which kind of object we're loading
    if load_halo:
        obj_str = 'halo_data'
    else:
        obj_str = 'galaxy_data'

    # get the star particle begin / end indices
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        begin = hf[f"{obj_str}/slist_start"][:]
        end = hf[f"{obj_str}/slist_end"][:]

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        # create the individual galaxy objects
        galaxies[i] = Galaxy()

        galaxies[i].load_stars(
            imasses[b:e],
            ages[b:e],
            metallicity[b:e],
            s_oxygen=s_oxygen[b:e],
            s_hydrogen=s_hydrogen[b:e],
            coordinates=coods[b:e, :],
            current_masses=masses[b:e],
        )

    # get the gas particle begin / end indices
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        begin = hf[f"{obj_str}/glist_start"][:]
        end = hf[f"{obj_str}/glist_end"][:]

    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i].load_gas(
            coordinates=g_coods[b:e],
            masses=g_masses[b:e],
            metals=g_metals[b:e],
            star_forming=star_forming[b:e],
            smoothing_lengths=g_hsml[b:e],
            dust_masses=g_dustmass[b:e],
        )

    return galaxies
