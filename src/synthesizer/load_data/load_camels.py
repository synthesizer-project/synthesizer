"""A submodule for loading CAMELS data into the synthesizer.

Example usage:

    # Load CAMELS-IllustrisTNG data
    galaxies = load_CAMELS_IllustrisTNG(
        _dir="path/to/data",
        snap_name="snap_033.hdf5",
        group_name="fof_subhalo_tab_033.hdf5",
        group_dir="path/to/group",
        verbose=True,
        dtm=0.3,
        physical=True,
        age_lookup=True,
        age_lookup_delta_a=1e-4,
    )
"""

from functools import partial

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from unyt import Mpc, Msun, kpc, yr

from synthesizer.exceptions import UnmetDependency
from synthesizer.load_data.utils import (
    age_lookup_table,
    get_begin_end_pointers,
    lookup_age,
)

from ..particle.galaxy import Galaxy


def _load_CAMELS(
    lens,
    imasses,
    ages,
    metallicities,
    s_oxygen,
    s_hydrogen,
    coods,
    masses,
    g_coods,
    g_masses,
    g_metallicities,
    g_hsml,
    star_forming,
    redshift,
    centre,
    s_hsml=None,
    dtm=0.3,
):
    """Load CAMELS galaxies into a galaxy object.

    Arbitrary back end for different CAMELS simulation suites

    Args:
        lens (np.ndarray of int):
            subhalo particle length array
        imasses (np.ndarray of float):
            initial masses particle array
        ages (np.ndarray of float):
            particle ages array
        metallicities (array):
            particle summed metallicities array
        s_oxygen (array):
            particle oxygen abundance array
        s_hydrogen (array):
            particle hydrogen abundance array
        s_hsml (array):
            star particle smoothing lengths array, comoving
        coods (array):
            particle coordinates array, comoving
        masses (array):
            current mass particle array
        g_coods (array):
            gas particle coordinates array, comoving
        g_masses (array):
            gas particle masses array
        g_metallicities (array):
            gas particle overall metallicities array
        g_hsml (array):
            gas particle smoothing lengths array, comoving
        star_forming (array):
            boolean array flagging star forming gas particles
        redshift (float):
            Galaxies redshift
        centre (array):
            Coordinates of the galaxies centre. Can be defined
            as required (e.g. can be centre of mass)
        dtm (float):
            dust-to-metals ratio to apply to all particles

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing specified
            stars and gas objects
    """
    begin, end = get_begin_end_pointers(lens[:, 4])
    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i] = Galaxy()
        galaxies[i].redshift = redshift
        galaxies[i].centre = centre[i] * kpc

        if s_hsml is None:
            smoothing_lengths = s_hsml
        else:
            smoothing_lengths = s_hsml[b:e] * kpc

        galaxies[i].load_stars(
            initial_masses=imasses[b:e] * Msun,
            ages=ages[b:e] * yr,
            metallicities=metallicities[b:e],
            s_oxygen=s_oxygen[b:e],
            s_hydrogen=s_hydrogen[b:e],
            coordinates=coods[b:e, :] * kpc,
            current_masses=masses[b:e] * Msun,
            smoothing_lengths=smoothing_lengths,
        )

    begin, end = get_begin_end_pointers(lens[:, 0])
    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i].load_gas(
            coordinates=g_coods[b:e] * kpc,
            masses=g_masses[b:e] * Msun,
            metallicities=g_metallicities[b:e],
            star_forming=star_forming[b:e],
            smoothing_lengths=g_hsml[b:e] * kpc,
            dust_to_metal_ratio=dtm,
        )

    return galaxies


def load_CAMELS_IllustrisTNG(
    _dir=".",
    snap_name="snap_033.hdf5",
    group_name="fof_subhalo_tab_033.hdf5",
    group_dir=None,
    verbose=False,
    dtm=0.3,
    physical=True,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
    **kwargs,
):
    """Load CAMELS-IllustrisTNG galaxies.

    Args:
        dir (string):
            data location
        snap_name (string):
            snapshot filename
        group_name (string):
            subfind / FOF filename
        group_dir (string):
            optional argument specifying lcoation of fof file
            if different to snapshot
        verbose (bool):
            verbosity flag
        dtm (float):
            dust-to-metals ratio to apply to all gas particles
        physical (bool):
            Should the coordinates be converted to physical?
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (float):
            Scale factor resolution of the age lookup
        **kwargs (dict):
            Additional keyword arguments to pass to the
            `_load_CAMELS` function.

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particle
    """
    with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
        scale_factor = hf["Header"].attrs["Time"]
        redshift = 1.0 / scale_factor - 1
        h = hf["Header"].attrs["HubbleParam"]
        Om0 = hf["Header"].attrs["Omega0"]

        form_time = hf["PartType4/GFM_StellarFormationTime"][:]
        coods = hf["PartType4/Coordinates"][:]  # kpc (comoving)
        masses = hf["PartType4/Masses"][:]
        imasses = hf["PartType4/GFM_InitialMass"][:]
        _metals = hf["PartType4/GFM_Metals"][:]
        metallicity = hf["PartType4/GFM_Metallicity"][:]
        hsml = hf["PartType4/SubfindHsml"][:]

        g_sfr = hf["PartType0/StarFormationRate"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/GFM_Metallicity"][:]
        g_coods = hf["PartType0/Coordinates"][:]  # kpc (physical)
        g_hsml = hf["PartType0/SubfindHsml"][:]

    if group_dir:
        _dir = group_dir  # replace if symlinks for fof files are broken
    with h5py.File(f"{_dir}/{group_name}", "r") as hf:
        lens = hf["Subhalo/SubhaloLenType"][:]
        pos = hf["Subhalo/SubhaloPos"][:]  # kpc (comoving)

    """
    remove wind particles
    """
    mask = form_time <= 0.0  # mask for wind particles

    if verbose:
        print("Wind particles:", np.sum(mask))

    # change len indexes
    for m in np.where(mask)[0]:
        # create array of end indexes
        cum_lens = np.append(0, np.cumsum(lens[:, 4]))

        # which halo does this wind particle belong to?
        which_halo = np.where(m < cum_lens)[0]

        # check we're not at the end of the array
        if len(which_halo) > 0:
            # reduce the length of *this* halo
            lens[which_halo[0] - 1, 4] -= 1

    # filter particle arrays
    imasses = imasses[~mask]
    form_time = form_time[~mask]
    coods = coods[~mask]
    metallicity = metallicity[~mask]
    masses = masses[~mask]
    _metals = _metals[~mask]
    hsml = hsml[~mask]

    masses = (masses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    imasses = (imasses * 1e10) / h

    star_forming = g_sfr > 0.0

    s_oxygen = _metals[:, 4]
    s_hydrogen = 1.0 - np.sum(_metals[:, 1:], axis=1)

    # Convert comoving coordinates to physical kpc
    if physical:
        coods *= scale_factor
        g_coods *= scale_factor
        hsml *= scale_factor
        g_hsml *= scale_factor
        pos *= scale_factor

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

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
    ages = (universe_age - _ages).to("yr").value

    return _load_CAMELS(
        lens=lens,
        imasses=imasses,
        ages=ages,
        metallicities=metallicity,
        s_oxygen=s_oxygen,
        s_hydrogen=s_hydrogen,
        s_hsml=hsml,
        coods=coods,
        masses=masses,
        g_coods=g_coods,
        g_masses=g_masses,
        g_metallicities=g_metals,
        g_hsml=g_hsml,
        star_forming=star_forming,
        redshift=redshift,
        centre=pos,
        dtm=dtm,
    )


def load_CAMELS_Astrid(
    _dir=".",
    snap_name="snap_090.hdf5",
    group_name="fof_subhalo_tab_090.hdf5",
    group_dir=None,
    dtm=0.3,
    physical=True,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
    **kwargs,
):
    """Load CAMELS-Astrid galaxies.

    Args:
        dir (string):
            data location
        snap_name (string):
            snapshot filename
        group_name (string):
            subfind / FOF filename
        group_dir (string):
            optional argument specifying lcoation of fof file
            if different to snapshot
        dtm (float):
            dust to metals ratio for all gas particles
        physical (bool):
            Should properties be converted to physical?
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (float):
            Scale factor resolution of the age lookup
        **kwargs (dict):
            Additional keyword arguments to pass to the
            `_load_CAMELS` function.

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particle
    """
    with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
        redshift = hf["Header"].attrs["Redshift"].astype(np.float32)[0]
        scale_factor = hf["Header"].attrs["Time"].astype(np.float32)[0]
        h = hf["Header"].attrs["HubbleParam"][0]
        Om0 = hf["Header"].attrs["Omega0"][0]

        form_time = hf["PartType4/GFM_StellarFormationTime"][:]
        coods = hf["PartType4/Coordinates"][:]  # kpc (comoving)
        masses = hf["PartType4/Masses"][:]

        imasses = np.ones(len(masses)) * 1.27e-4

        _metals = hf["PartType4/GFM_Metals"][:]
        metallicity = hf["PartType4/GFM_Metallicity"][:]

        g_sfr = hf["PartType0/StarFormationRate"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/GFM_Metallicity"][:]
        g_coods = hf["PartType0/Coordinates"][:]  # kpc (comoving)
        g_hsml = hf["PartType0/SmoothingLength"][:]

    masses = (masses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    imasses = (imasses * 1e10) / h

    star_forming = g_sfr > 0.0

    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

    # Are we creating a lookup table for ages?
    if age_lookup:
        # Create the lookup grid
        scale_factors, ages = age_lookup_table(
            cosmo, redshift=redshift, delta_a=age_lookup_delta_a
        )

        # Look up the ages for the particles
        _ages = lookup_age(form_time, scale_factors, ages)
    else:
        # Calculate ages of these explicitly using astropy
        _ages = cosmo.age(1.0 / form_time - 1)

    # Calculate ages at snapshot redshift
    ages = (universe_age - _ages).to("yr").value

    if group_dir:
        _dir = group_dir  # replace if symlinks for fof files are broken
    with h5py.File(f"{_dir}/{group_name}", "r") as hf:
        lens = hf["Subhalo/SubhaloLenType"][:]
        pos = hf["Subhalo/SubhaloPos"][:]  # kpc (comoving)

    # Convert comoving coordinates to physical kpc
    if physical:
        coods *= scale_factor
        g_coods *= scale_factor
        g_hsml *= scale_factor
        pos *= scale_factor

    return _load_CAMELS(
        redshift=redshift,
        lens=lens,
        imasses=imasses,
        ages=ages,
        metallicities=metallicity,
        s_oxygen=s_oxygen,
        s_hydrogen=s_hydrogen,
        coods=coods,
        masses=masses,
        g_coods=g_coods,
        g_masses=g_masses,
        g_metallicities=g_metals,
        g_hsml=g_hsml,
        star_forming=star_forming,
        dtm=dtm,
        centre=pos,
    )


def load_CAMELS_Simba(
    _dir=".",
    snap_name="snap_033.hdf5",
    group_name="fof_subhalo_tab_033.hdf5",
    group_dir=None,
    dtm=0.3,
    physical=True,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
    **kwargs,
):
    """Load CAMELS-SIMBA galaxies.

    Args:
        dir (string):
            data location
        snap_name (string):
            snapshot filename
        group_name (string):
            subfind / FOF filename
        group_dir (string):
            optional argument specifying lcoation of fof file
            if different to snapshot
        dtm (float):
            dust to metals ratio for all gas particles
        physical (bool):
            Should properties be converted to physical?
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (float):
            Scale factor resolution of the age lookup
        **kwargs (dict):
            Additional keyword arguments to pass to the
            `_load_CAMELS` function.

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particle
    """
    with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
        redshift = hf["Header"].attrs["Redshift"]
        scale_factor = hf["Header"].attrs["Time"]
        h = hf["Header"].attrs["HubbleParam"]
        Om0 = hf["Header"].attrs["Omega0"]

        form_time = hf["PartType4/StellarFormationTime"][:]
        coods = hf["PartType4/Coordinates"][:]  # kpc (comoving)
        masses = hf["PartType4/Masses"][:]
        imasses = (
            np.ones(len(masses)) * 0.00155
        )  # * hf['Header'].attrs['MassTable'][1]
        _metals = hf["PartType4/Metallicity"][:]

        g_sfr = hf["PartType0/StarFormationRate"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/Metallicity"][:][:, 0]
        g_coods = hf["PartType0/Coordinates"][:]  # kpc (comoving)
        g_hsml = hf["PartType0/SmoothingLength"][:]

    masses = (masses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    imasses = (imasses * 1e10) / h

    star_forming = g_sfr > 0.0

    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metallicity = _metals[:, 0]

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

    # Are we creating a lookup table for ages?
    if age_lookup:
        # Create the lookup grid
        scale_factors, ages = age_lookup_table(
            cosmo, redshift=redshift, delta_a=age_lookup_delta_a
        )

        # Look up the ages for the particles
        _ages = lookup_age(form_time, scale_factors, ages)
    else:
        # Calculate ages of these explicitly using astropy
        _ages = cosmo.age(1.0 / form_time - 1)

    # Calculate ages at snapshot redshift
    ages = (universe_age - _ages).to("yr").value

    if group_dir:
        _dir = group_dir  # replace if symlinks for fof files are broken
    with h5py.File(f"{_dir}/{group_name}", "r") as hf:
        lens = hf["Subhalo/SubhaloLenType"][:]
        pos = hf["Subhalo/SubhaloPos"][:]  # kpc (comoving)

    # Convert comoving coordinates to physical kpc
    if physical:
        coods *= scale_factor
        g_coods *= scale_factor
        g_hsml *= scale_factor
        pos *= scale_factor

    return _load_CAMELS(
        redshift=redshift,
        lens=lens,
        imasses=imasses,
        ages=ages,
        metallicities=metallicity,
        s_oxygen=s_oxygen,
        s_hydrogen=s_hydrogen,
        coods=coods,
        masses=masses,
        g_coods=g_coods,
        g_masses=g_masses,
        g_metallicities=g_metals,
        g_hsml=g_hsml,
        star_forming=star_forming,
        dtm=dtm,
        centre=pos,
    )


def load_CAMELS_SwiftEAGLE_subfind(
    _dir=".",
    snap_name="snapshot_033.hdf5",
    group_name="groups_033.hdf5",
    group_dir=None,
    dtm=0.3,
    physical=True,
    min_star_part=10,
    num_threads=-1,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
    **kwargs,
):
    """Load CAMELS-Swift-EAGLE galaxies.

    Args:
        dir (string):
            data location
        snap_name (string):
            snapshot filename
        group_name (string):
            subfind / FOF filename
        group_dir (string):
            optional argument specifying location of fof file
            if different to snapshot
        verbose (bool):
            verbosity flag
        dtm (float):
            dust-to-metals ratio to apply to all gas particles
        physical (bool):
            Should the coordinates be converted to physical?
        min_star_part (int):
            minimum number of star particles required to load galaxy
        num_threads (int):
            number of threads to use for multiprocessing.
            Default is -1, i.e. use all available cores.
        age_lookup (bool):
            Create a lookup table for ages
        age_lookup_delta_a (int):
            Scale factor resolution of the age lookup
        **kwargs (dict):
            Additional keyword arguments to pass to the
            `_load_CAMELS` function.

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing star and gas particle
    """
    try:
        import schwimmbad
    except ImportError:
        raise UnmetDependency(
            "Loading Swift-EAGLE CAMELS data requires the `schwimmbad`"
            "package. You currently do not have schwimmbad installed. "
            "Install it via `pip install schwimmbad`"
        )

    if num_threads == 1:
        pool = schwimmbad.SerialPool()
    elif num_threads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=num_threads)

    # Check if snapshot and subfind files in same directory
    if group_dir is None:
        group_dir = _dir

    # Load cosmology information
    with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
        scale_factor = hf["Cosmology"].attrs["Scale-factor"][0]
        redshift = hf["Cosmology"].attrs["Redshift"][0]
        Om0 = hf["Cosmology"].attrs["Omega_m"][0]
        h = hf["Cosmology"].attrs["h"][0]

    # get subfind particle info (lens and IDs) for subsetting snapshot info
    with h5py.File(f"{group_dir}/{group_name}", "r") as hf:
        lentype = hf["Subhalo/SubhaloLenType"][:]
        grp_lentype = hf["Group/GroupLenType"][:]
        grpn = hf["Subhalo/SubhaloGrNr"][:]
        grp_firstsub = hf["Group/GroupFirstSub"][:]
        ids = hf["IDs/ID"][:]
        pos = hf["Subhalo/SubhaloPos"][:] / h  # kpc (comoving)

    with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
        # Load star particle information
        star_ids = hf["PartType4/ParticleIDs"][:]
        form_time = hf["PartType4/BirthScaleFactors"][:]
        coods = hf["PartType4/Coordinates"][:]
        masses = hf["PartType4/Masses"][:]
        imasses = hf["PartType4/InitialMasses"][:]
        _metals = hf["PartType4/SmoothedElementMassFractions"][:]
        metallicity = hf["PartType4/SmoothedMetalMassFractions"][:]
        hsml = hf["PartType4/SmoothingLengths"][:]

        # Load gas particle information
        gas_ids = hf["PartType0/ParticleIDs"][:]
        g_sfr = hf["PartType0/StarFormationRates"][:]
        g_masses = hf["PartType0/Masses"][:]
        g_metals = hf["PartType0/SmoothedMetalMassFractions"][:]
        g_coods = hf["PartType0/Coordinates"][:]
        g_hsml = hf["PartType0/SmoothingLengths"][:]

    masses = masses * 1e10
    imasses = imasses * 1e10
    g_masses = g_masses * 1e10

    # Convert comoving coordinates to physical kpc
    if physical:
        coods *= scale_factor
        g_coods *= scale_factor
        hsml *= scale_factor
        g_hsml *= scale_factor
        pos *= scale_factor

    # Get subhalos with minimum number of star particles
    mask = np.where(lentype[:, 4] > min_star_part)[0]

    # Convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

    # Are we creating a lookup table for ages?
    if age_lookup:
        # Create the lookup grid
        scale_factors, ages = age_lookup_table(
            cosmo, redshift=redshift, delta_a=age_lookup_delta_a
        )

        # Look up the ages for the particles
        _ages = lookup_age(form_time, scale_factors, ages)
    else:
        # Calculate ages of these explicitly using astropy
        _ages = cosmo.age(1.0 / form_time - 1)

    # Calculate ages at snapshot redshift
    ages = (universe_age - _ages).to("yr").value

    def swifteagle_particle_assignment(
        idx,
        redshift,
        pos,
        grpn,
        grp_lentype,
        grp_firstsub,
        lentype,
        ids,
        star_ids,
        gas_ids,
        ages,
        coods,
        masses,
        imasses,
        metallicity,
        hsml,
        g_sfr,
        g_masses,
        g_metals,
        g_coods,
        g_hsml,
    ):
        gal = Galaxy(name=idx)
        gal.redshift = redshift
        gal.centre = pos[idx] * kpc
        gal.index = idx

        # Find star particles in this subhalo
        ptype = 4

        # First find group number for this subhalo
        _grpn = grpn[idx]

        # Find particle index for preceding groups
        grp_lowi = np.sum(grp_lentype[:_grpn])

        # Find particle index for preceding subhalos in same group
        sh_lowi = np.sum(lentype[grp_firstsub[_grpn] : idx])

        # Find particle index upto this subhalo
        lowi = grp_lowi + sh_lowi + np.sum(lentype[idx, :ptype])

        # Find upper index for this subhalo particle type
        uppi = lowi + lentype[idx, ptype] + 1

        # Filter particle IDs for this subhalo
        part_ids = np.where(np.in1d(star_ids, ids[lowi:uppi]))[0]

        # Filter particle arrays for this subhalo
        sh_ages = ages[part_ids]
        sh_coods = coods[part_ids]
        sh_masses = masses[part_ids]
        sh_imasses = imasses[part_ids]
        sh_metallicity = metallicity[part_ids]
        sh_hsml = hsml[part_ids]

        # Get individual element abundances
        s_oxygen = _metals[part_ids, 4]
        s_hydrogen = 1.0 - np.sum(_metals[part_ids, 1:], axis=1)

        # Check for smoothing lengths
        if sh_hsml is None:
            smoothing_lengths = sh_hsml
        else:
            smoothing_lengths = sh_hsml * Mpc

        gal.load_stars(
            initial_masses=sh_imasses * Msun,
            ages=sh_ages * yr,
            metallicities=sh_metallicity,
            s_oxygen=s_oxygen,
            s_hydrogen=s_hydrogen,
            coordinates=sh_coods * Mpc,
            current_masses=sh_masses * Msun,
            smoothing_lengths=smoothing_lengths,
        )

        # Check there are gas particles in this subhalo
        if lentype[idx, 0] > 0:
            # Find gas particles in this subhalo
            ptype = 0

            # Find particle index upto this subhalo
            lowi = grp_lowi + sh_lowi + np.sum(lentype[idx, :ptype])

            # Find upper index for this subhalo particle type
            uppi = lowi + lentype[idx, ptype] + 1

            # Filter particle IDs for this subhalo
            part_ids = np.where(np.in1d(gas_ids, ids[lowi:uppi]))[0]

            sh_g_sfr = g_sfr[part_ids]
            sh_g_masses = g_masses[part_ids]
            sh_g_metals = g_metals[part_ids]
            sh_g_coods = g_coods[part_ids]
            sh_g_hsml = g_hsml[part_ids]

            star_forming = sh_g_sfr > 0.0

            gal.load_gas(
                coordinates=sh_g_coods * Mpc,
                masses=sh_g_masses * Msun,
                metallicities=sh_g_metals,
                star_forming=star_forming,
                smoothing_lengths=sh_g_hsml * Mpc,
                dust_to_metal_ratio=dtm,
            )

        return gal

    _f = partial(
        swifteagle_particle_assignment,
        redshift=redshift,
        pos=pos,
        grpn=grpn,
        grp_lentype=grp_lentype,
        grp_firstsub=grp_firstsub,
        lentype=lentype,
        ids=ids,
        star_ids=star_ids,
        gas_ids=gas_ids,
        ages=ages,
        coods=coods,
        masses=masses,
        imasses=imasses,
        metallicity=metallicity,
        hsml=hsml,
        g_sfr=g_sfr,
        g_masses=g_masses,
        g_metals=g_metals,
        g_coods=g_coods,
        g_hsml=g_hsml,
    )

    galaxies = list(pool.map(_f, mask))
    pool.close()

    return galaxies
