"""
A module for interfacing with the outputs of the EAGLE
hydrodynamical simulations suite.

The EAGLE hdf5 data loading scripts have been taken from
the `eagle_io` package (https://github.com/flaresimulations/eagle_IO/).
This has been to make the call as self-contained as possible.

"""

import glob
import re
import sys
import timeit
from functools import partial

import h5py
import numpy as np
from astropy.cosmology import Planck13 as cosmo

from synthesizer import exceptions

try:
    import schwimmbad
except exceptions.UnmetDependency:
    print(
        "Loading eagle data requires the schwimmbad package"
        "You currently do not have schwimmbad installed."
        "Install it via 'pip install schwimmbad'"
    )
    sys.exit()

from unyt import Mpc, Msun, yr

from ..particle.galaxy import Galaxy


def load_EAGLE(
    fileloc, tag, numThreads=1, chunk=1, tot_chunks=256, mshared=False
):
    """
    Load EAGLE required EAGLE galaxy properties
    for generating their SEDs
    Most useful for running on high-z snaps

    Args:
        fileloc (string):
            eagle data file location
        tag (string):
            snapshot tag to load
        numThreads (int)
            number of threads to use
        chunk (int)
            file number to process
        tot_chunks (int)
            total number of files to process
        mshared (bool)
            if memory is shared across cpus/nodes

    """
    zed = float(tag[5:].replace("p", "."))
    if chunk > tot_chunks:
        exceptions.InconsistentArguments(
            "Value specified by 'chunk' should be lower"
            "than the total chunks (tot_chunks)"
        )

    with h5py.File(
        f"{fileloc}/groups_{tag}/eagle_subfind_tab_{tag}.{chunk-1}.hdf5", "r"
    ) as hf:
        sgrpno = np.array(hf.get("/Subhalo/SubGroupNumber"))
        grpno = np.array(hf.get("/Subhalo/GroupNumber"))

    if mshared:
        print("Not implemented yet!")

    else:
        # Get required star particle properties
        s_sgrpno = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/SubGroupNumber",
            numThreads=numThreads,
        )
        s_grpno = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/GroupNumber",
            numThreads=numThreads,
        )
        # Create mask for particles in the current chunk
        ok = np.in1d(s_sgrpno, sgrpno) * np.in1d(s_grpno, grpno)
        s_sgrpno = s_sgrpno[ok]
        s_grpno = s_grpno[ok]

        s_coords = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/Coordinates",
            noH=True,
            physicalUnits=True,
            numThreads=numThreads,
        )[ok]  # physical Mpc
        s_imasses = (
            read_array(
                "PARTDATA",
                fileloc,
                tag,
                "/PartType4/InitialMass",
                noH=True,
                physicalUnits=True,
                numThreads=numThreads,
            )[ok]
            * 1e10
        )  #  Msun
        S_ages = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/StellarFormationTime",
            numThreads=numThreads,
        )[ok]
        S_ages = get_age(S_ages, zed, numThreads=numThreads)  # Gyr
        s_Zsmooth = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/SmoothedMetallicity",
            numThreads=numThreads,
        )[ok]
        s_oxygen = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/ElementAbundance/Oxygen",
            numThreads=numThreads,
        )[ok]
        s_hydrogen = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/ElementAbundance/Hydrogen",
            numThreads=numThreads,
        )[ok]

        # Get gas particle properties
        g_sgrpno = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/SubGroupNumber",
            numThreads=numThreads,
        )
        g_grpno = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/GroupNumber",
            numThreads=numThreads,
        )
        # Create mask for particles in the current chunk
        ok = np.in1d(g_sgrpno, sgrpno) * np.in1d(g_grpno, grpno)
        g_sgrpno = g_sgrpno[ok]
        g_grpno = g_grpno[ok]

        g_coords = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/Coordinates",
            noH=True,
            physicalUnits=True,
            numThreads=numThreads,
        )[ok]  # physical Mpc
        g_masses = (
            read_array(
                "PARTDATA",
                fileloc,
                tag,
                "/PartType0/Mass",
                noH=True,
                physicalUnits=True,
                numThreads=numThreads,
            )[ok]
            * 1e10
        )  # Msun
        g_sfr = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/StarFormationRate",
            numThreads=numThreads,
        )[ok]  # Msol / yr
        g_Zsmooth = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/SmoothedMetallicity",
            numThreads=numThreads,
        )[ok]
        g_hsml = read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType0/SmoothingLength",
            noH=True,
            physicalUnits=True,
            numThreads=numThreads,
        )[ok]  # physical Mpc

    galaxies = [None] * len(sgrpno)

    for ii in range(len(sgrpno)):
        # Create the individual galaxy objects
        galaxies[ii] = Galaxy(redshift=zed)

        # Fill individual galaxy objects with star particles
        # mask for current galaxy
        ok = (s_grpno == grpno[ii]) * (s_sgrpno == sgrpno[ii])
        galaxies[ii].load_stars(
            initial_masses=s_imasses[ok] * Msun,
            ages=S_ages[ok] * 1e9 * yr,
            metallicities=s_Zsmooth[ok],
            coordinates=s_coords[ok],
            s_oxygen=s_oxygen[ok],
            s_hydrogen=s_hydrogen[ok],
        )

        # Fill individual galaxy objects with gas particles
        sfr_flag = g_sfr > 0
        # mask for current galaxy
        ok = (g_grpno == grpno[ii]) * (g_sgrpno == sgrpno[ii])
        galaxies[ii].load_gas(
            masses=g_masses[ok] * Msun,
            metallicities=g_Zsmooth[ok],
            star_forming=sfr_flag[ok],
            coordinates=g_coords[ok] * Mpc,
            smoothing_lengths=g_hsml[ok] * Mpc,
        )

    return galaxies


def read_hdf5(f, dataset):
    with h5py.File(f, "r") as hf:
        dat = np.array(hf.get(dataset))
        if dat.ndim == 0:
            return np.array([])

    return dat


def get_files(fileType, directory, tag):
    """
    Fetch filename

    Args:
        fileType (str)
        directory (str)
        tag (str)
    """

    if fileType in ["FOF", "FOF_PARTICLES"]:
        files = glob.glob(
            "%s/groups_%s/group_tab_%s*.hdf5" % (directory, tag, tag)
        )
    elif fileType in ["SNIP_FOF", "SNIP_FOF_PARTICLES"]:
        files = glob.glob(
            "%s/groups_snip_%s/group_snip_tab_%s*.hdf5" % (directory, tag, tag)
        )
    elif fileType in ["SUBFIND", "SUBFIND_GROUP", "SUBFIND_IDS"]:
        files = glob.glob(
            "%s/groups_%s/eagle_subfind_tab_%s*.hdf5" % (directory, tag, tag)
        )
    elif fileType in [
        "SNIP_SUBFIND",
        "SNIP_SUBFIND_GROUP",
        "SNIP_SUBFIND_IDS",
    ]:
        files = glob.glob(
            "%s/groups_snip_%s/eagle_subfind_snip_tab_%s*.hdf5"
            % (directory, tag, tag)
        )
    elif fileType in ["SNIP", "SNIPSHOT"]:
        files = glob.glob(
            "%s/snipshot_%s/snip_%s*.hdf5" % (directory, tag, tag)
        )
    elif fileType in ["SNAP", "SNAPSHOT"]:
        files = glob.glob(
            "%s/snapshot_%s/snap_%s*.hdf5" % (directory, tag, tag)
        )
    elif fileType in ["PARTDATA"]:
        files = glob.glob(
            "%s/particledata_%s/eagle_subfind_particles_%s*.hdf5"
            % (directory, tag, tag)
        )
    elif fileType in ["SNIP_PARTDATA"]:
        files = glob.glob(
            "%s/particledata_snip_%s/eagle_subfind_snip_particles_%s*.hdf5"
            % (directory, tag, tag)
        )
    else:
        raise ValueError("Type of files not supported")

    return sorted(files, key=lambda x: int(re.findall(r"(\d+)", x)[-2]))


def apply_physicalUnits_conversion(f, dataset, dat, verbose=True):
    """

    Args:
        f (str)
        dat (array)
    """
    with h5py.File(f, "r") as hf:
        exponent = hf[dataset].attrs["aexp-scale-exponent"]
        a = hf["Header"].attrs["ExpansionFactor"]

    if exponent != 0.0:
        if verbose:
            print(
                "Converting to physical units."
                "(Multiplication by a^%f, a=%f)" % (exponent, a)
            )
        return dat * pow(a, exponent)
    else:
        if verbose:
            print("Converting to physical units. No conversion needed!")
        return dat


def apply_hfreeUnits_conversion(f, dataset, dat, verbose=True):
    """

    Args:
        f (str)
        dat (array)
    """
    with h5py.File(f, "r") as hf:
        exponent = hf[dataset].attrs["h-scale-exponent"]
        h = hf["Header"].attrs["HubbleParam"]

    if exponent != 0.0:
        if verbose:
            print(
                "Converting to h-free units."
                "(Multiplication by h^%f, h=%f)" % (exponent, h)
            )
        return dat * pow(h, exponent)
    else:
        if verbose:
            print("Converting to h-free units. No conversion needed!")
        return dat


def apply_CGSUnits_conversion(f, dataset, dat, verbose=True):
    """
    A function to convert to CGS units.

    :param f:
    :param dataset:
    :param dat:
    :param verbose:
    :return:
    """

    with h5py.File(f, "r") as hf:
        cgs = hf[dataset].attrs["CGSConversionFactor"]

    if cgs != 1.0:
        if verbose:
            print(
                "Converting to CGS units."
                "(Multiplication by CGS conversion factor = %f)" % cgs
            )
        return dat * cgs
    else:
        if verbose:
            print("Converting to CGS units. No conversion needed!")
        return dat


def read_array(
    ftype,
    directory,
    tag,
    dataset,
    numThreads=1,
    noH=False,
    physicalUnits=False,
    CGS=False,
    verbose=True,
):
    """

    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
        numThreads (int)
        noH (bool)
        physicalUnits (bool)
    """
    start = timeit.default_timer()

    files = get_files(ftype, directory, tag)

    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    lg = partial(read_hdf5, dataset=dataset)
    dat = list(pool.map(lg, files))

    # get indices of non-empty arrays
    non_empty_file_indices = [i for i, d in enumerate(dat) if d.shape[0] != 0]

    # ignore files with no data
    dat = [d for d in dat if d.shape[0] != 0]
    dat = np.concatenate(dat, axis=0)

    pool.close()

    stop = timeit.default_timer()

    if verbose:
        print(
            "Reading in '{}' for z = {} using {} thread(s) took {}s".format(
                dataset,
                np.round(
                    read_header(ftype, directory, tag, dataset="Redshift"), 3
                ),
                numThreads,
                np.round(stop - start, 6),
            )
        )

    if noH:
        dat = apply_hfreeUnits_conversion(
            files[non_empty_file_indices[0]], dataset, dat, verbose=verbose
        )

    if physicalUnits:
        dat = apply_physicalUnits_conversion(
            files[non_empty_file_indices[0]], dataset, dat, verbose=verbose
        )
    if CGS:
        dat = apply_CGSUnits_conversion(
            files[non_empty_file_indices[0]], dataset, dat, verbose=verbose
        )
    return dat


def read_header(ftype, directory, tag, dataset):
    """
    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
    """

    files = get_files(ftype, directory, tag)
    with h5py.File(files[0], "r") as hf:
        hdr = hf["Header"].attrs[dataset]

    return hdr


def get_star_formation_time(scale_factor):
    SFz = (1 / scale_factor) - 1.0  # convert scale factor to z
    return cosmo.age(SFz).value


def get_age(scale_factors, z, numThreads=4):
    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    Age = cosmo.age(z).value - np.array(
        list(pool.map(get_star_formation_time, scale_factors))
    )
    pool.close()

    return Age
