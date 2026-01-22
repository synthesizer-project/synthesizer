"""A submodule for loading EAGLE data into Synthesizer.

The EAGLE hdf5 data loading scripts have been taken from
the `eagle_io` package (https://github.com/flaresimulations/eagle_IO/).
This has been to make the call as self-contained as possible.
"""

import glob
import re
import timeit
from collections import namedtuple
from functools import partial
from typing import Any, Dict, List, Union

import h5py
import numpy as np
from astropy.cosmology import LambdaCDM
from numpy.typing import NDArray
from typing_extensions import Never

from synthesizer.exceptions import InconsistentArguments, UnmetDependency

try:
    import schwimmbad
except ImportError:
    raise UnmetDependency(
        "The `schwimmbad` module is required to load eagle data. "
        "Install synthesizer with the `eagle` extra dependencies:"
        " `pip install .[eagle]`."
    )

from unyt import Mpc, Msun, km, s, unyt_array, unyt_quantity, yr

from synthesizer.load_data.utils import lookup_age

from ..particle.galaxy import Galaxy

# define EAGLE cosmology
cosmo = LambdaCDM(Om0=0.307, Ode0=0.693, H0=67.77, Ob0=0.04825)
norm = np.linalg.norm


def load_EAGLE(
    fileloc: str,
    args: namedtuple,
    tot_chunks: int = 1535,
    verbose: bool = False,
) -> List[Union[Galaxy, Never]]:
    """Load EAGLE required EAGLE galaxy properties.

    Args:
        fileloc (string):
            eagle data file location
        args (namedtuple):
            parser arguments passed on to this job
        tot_chunks (int):
            total number of files to process
        verbose (bool):
            Are we talking?

    Returns:
        a dictionary of Galaxy objects with stars and gas components
    """
    tag = args.tag
    numThreads = args.nthreads
    chunk = args.chunk

    if chunk > tot_chunks:
        raise InconsistentArguments(
            "Value specified by 'chunk' should be lower"
            "than the total chunks (tot_chunks)"
        )

    # Read some of the simulation box information
    h = read_header("SUBFIND", fileloc, tag, "HubbleParam")
    boxl = read_header("SUBFIND", fileloc, tag, "BoxSize") / h
    exp_fac = read_header("SUBFIND", fileloc, tag, "ExpansionFactor")
    zed = (1.0 / exp_fac) - 1.0
    boxl = boxl / (1 + zed)

    with h5py.File(
        f"{fileloc}/groups_{tag}/eagle_subfind_tab_{tag}.{chunk}.hdf5", "r"
    ) as hf:
        sgrpno = np.array(hf.get("/Subhalo/SubGroupNumber"), dtype=np.int32)
        grpno = np.array(hf.get("/Subhalo/GroupNumber"), dtype=np.int32)
        # centre of potential of subhalo in h-less physical Mpc
        cop = np.array(
            hf.get("/Subhalo/CentreOfPotential"), dtype=np.float32
        ) / (h * (1 + zed))

    if grpno.dtype == object:
        return []

    # Get required star particle properties
    s_sgrpno = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/SubGroupNumber",
        numThreads=numThreads,
        verbose=verbose,
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
        verbose=verbose,
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
            verbose=verbose,
        )[ok]
        * 1e10
    )  #  Msun
    s_masses = (
        read_array(
            "PARTDATA",
            fileloc,
            tag,
            "/PartType4/Mass",
            noH=True,
            physicalUnits=True,
            numThreads=numThreads,
            verbose=verbose,
        )[ok]
        * 1e10
    )  #  Msun
    s_ages = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/StellarFormationTime",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]
    s_ages = get_age(s_ages, zed, args, numThreads=numThreads)  # Gyr
    s_Zsmooth = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/SmoothedMetallicity",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]
    s_hsml = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/SmoothingLength",
        noH=True,
        physicalUnits=True,
        numThreads=numThreads,
        verbose=verbose,
    )[ok]  # physical Mpc
    s_oxygen = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/ElementAbundance/Oxygen",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]
    s_hydrogen = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/ElementAbundance/Hydrogen",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]

    # get velocities
    s_velocities = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType4/Velocity",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]

    # Get gas particle properties
    g_sgrpno = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType0/SubGroupNumber",
        numThreads=numThreads,
        verbose=verbose,
    )
    g_grpno = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType0/GroupNumber",
        numThreads=numThreads,
        verbose=verbose,
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
        verbose=verbose,
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
            verbose=verbose,
        )[ok]
        * 1e10
    )  # Msun
    g_sfr = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType0/StarFormationRate",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]  # Msol / yr
    g_Zsmooth = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType0/SmoothedMetallicity",
        numThreads=numThreads,
        verbose=verbose,
    )[ok]
    g_hsml = read_array(
        "PARTDATA",
        fileloc,
        tag,
        "/PartType0/SmoothingLength",
        noH=True,
        physicalUnits=True,
        numThreads=numThreads,
        verbose=verbose,
    )[ok]  # physical Mpc

    _f = partial(
        assign_galaxy_prop,
        zed=zed,
        boxl=boxl,
        aperture=args.aperture,
        grpno=grpno,
        sgrpno=sgrpno,
        cop=cop,
        s_grpno=s_grpno,
        s_sgrpno=s_sgrpno,
        s_imasses=s_imasses,
        s_masses=s_masses,
        s_ages=s_ages,
        s_Zsmooth=s_Zsmooth,
        s_coords=s_coords,
        s_hsml=s_hsml,
        s_oxygen=s_oxygen,
        s_hydrogen=s_hydrogen,
        s_velocities=s_velocities,
        g_grpno=g_grpno,
        g_sgrpno=g_sgrpno,
        g_masses=g_masses,
        g_Zsmooth=g_Zsmooth,
        g_sfr=g_sfr,
        g_coords=g_coords,
        g_hsml=g_hsml,
        verbose=verbose,
    )

    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    with schwimmbad.MultiPool(numThreads) as pool:
        galaxies = pool.map(_f, np.arange(len(grpno)))

    return galaxies


def load_EAGLE_shm(
    chunk: int,
    fileloc: str,
    tag: str,
    s_len: int,
    g_len: int,
    args: namedtuple,
    numThreads: int = 1,
    dtype: str = "float32",
    tot_chunks: int = 1535,
    verbose: bool = False,
) -> List[Union[Galaxy, Never]]:
    """Load EAGLE required EAGLE galaxy properties.

    Note, this function will use a memory mapped array to read in the data
    from the hdf5 files. This is much faster than reading in the data
    directly from the hdf5 files.

    Args:
        chunk (int):
            file number to process
        fileloc (string):
            eagle data file location
        tag (string):
            snapshot tag to load
        s_len (int):
            total number of star particles
        g_len (int):
            total number of gas particles
        args (namedtuple):
            parser arguments passed on to this job
        numThreads (int):
            number of threads to use
        dtype (numpy object):
            data type of the array in memory
        tot_chunks (int):
            total number of files to process
        verbose (bool):
            Are we talking?

    Returns:
        a dictionary of Galaxy objects with stars and gas components
    """
    # get the redshift from the given eagle tag
    zed = float(tag[5:].replace("p", "."))
    h = 0.6777
    if chunk > tot_chunks:
        InconsistentArguments(
            "Value specified by 'chunk' should be lower"
            "than the total chunks (tot_chunks)"
        )

    # Read some of the simulation box information
    h = read_header("SUBFIND", fileloc, tag, "HubbleParam")
    boxl = read_header("SUBFIND", fileloc, tag, "BoxSize") / h
    exp_fac = read_header("SUBFIND", fileloc, tag, "ExpansionFactor")
    zed = (1.0 / exp_fac) - 1.0

    with h5py.File(
        f"{fileloc}/groups_{tag}/eagle_subfind_tab_{tag}.{chunk}.hdf5", "r"
    ) as hf:
        sgrpno = np.array(hf.get("/Subhalo/SubGroupNumber"), dtype=np.int32)
        grpno = np.array(hf.get("/Subhalo/GroupNumber"), dtype=np.int32)
        # centre of potential of subhalo in h-less physical Mpc
        cop = np.array(
            hf.get("/Subhalo/CentreOfPotential"), dtype=np.float32
        ) / (h * (1 + zed))

    if grpno.dtype == object:
        return []

    # read in required star particle shared memory arrays
    s_grpno = np_shm_read(
        f"{args.shm_prefix}s_grpno{args.shm_suffix}", (s_len,), "int32"
    )
    s_sgrpno = np_shm_read(
        f"{args.shm_prefix}s_sgrpno{args.shm_suffix}", (s_len,), "int32"
    )

    # Create mask for particles in the current chunk
    ok = np.in1d(s_sgrpno, sgrpno) * np.in1d(s_grpno, grpno)
    s_sgrpno = s_sgrpno[ok]
    s_grpno = s_grpno[ok]

    s_imasses = np_shm_read(
        f"{args.shm_prefix}s_imasses{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_masses = np_shm_read(
        f"{args.shm_prefix}s_masses{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_ages = np_shm_read(
        f"{args.shm_prefix}s_ages{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_Zsmooth = np_shm_read(
        f"{args.shm_prefix}s_Zsmooth{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_coords = np_shm_read(
        f"{args.shm_prefix}s_coords{args.shm_suffix}", (s_len, 3), dtype
    )[ok]
    s_hsml = np_shm_read(
        f"{args.shm_prefix}s_hsml{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_oxygen = np_shm_read(
        f"{args.shm_prefix}s_oxygen{args.shm_suffix}", (s_len,), dtype
    )[ok]
    s_hydrogen = np_shm_read(
        f"{args.shm_prefix}s_hydrogen{args.shm_suffix}", (s_len,), dtype
    )[ok]

    # read in required gas particle shared memory arrays
    g_grpno = np_shm_read(
        f"{args.shm_prefix}g_grpno{args.shm_suffix}", (g_len,), "int32"
    )
    g_sgrpno = np_shm_read(
        f"{args.shm_prefix}g_sgrpno{args.shm_suffix}", (g_len,), "int32"
    )

    # Create mask for particles in the current chunk
    ok = np.in1d(g_sgrpno, sgrpno) * np.in1d(g_grpno, grpno)
    g_sgrpno = g_sgrpno[ok]
    g_grpno = g_grpno[ok]

    g_masses = np_shm_read(
        f"{args.shm_prefix}g_masses{args.shm_suffix}", (g_len,), dtype
    )[ok]
    g_sfr = np_shm_read(
        f"{args.shm_prefix}g_sfr{args.shm_suffix}", (g_len,), dtype
    )[ok]
    g_Zsmooth = np_shm_read(
        f"{args.shm_prefix}g_Zsmooth{args.shm_suffix}", (g_len,), dtype
    )[ok]
    g_coords = np_shm_read(
        f"{args.shm_prefix}g_coords{args.shm_suffix}", (g_len, 3), dtype
    )[ok]
    g_hsml = np_shm_read(
        f"{args.shm_prefix}g_hsml{args.shm_suffix}", (g_len,), dtype
    )[ok]

    _f = partial(
        assign_galaxy_prop,
        zed=zed,
        boxl=boxl,
        aperture=args.aperture,
        grpno=grpno,
        sgrpno=sgrpno,
        cop=cop,
        s_grpno=s_grpno,
        s_sgrpno=s_sgrpno,
        s_imasses=s_imasses,
        s_masses=s_masses,
        s_ages=s_ages,
        s_Zsmooth=s_Zsmooth,
        s_coords=s_coords,
        s_hsml=s_hsml,
        s_oxygen=s_oxygen,
        s_hydrogen=s_hydrogen,
        g_grpno=g_grpno,
        g_sgrpno=g_sgrpno,
        g_masses=g_masses,
        g_Zsmooth=g_Zsmooth,
        g_sfr=g_sfr,
        g_coords=g_coords,
        g_hsml=g_hsml,
        verbose=verbose,
    )

    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    galaxies = list(pool.map(_f, np.arange(len(grpno))))
    pool.close()

    return galaxies


def np_shm_read(name: str, array_shape: tuple, dtype: str) -> NDArray[Any]:
    """Read the required numpy memmap array.

    Args:
        name (str):
            location of the memmap
        array_shape (tuple):
            shape of the memmap array
        dtype (str):
            data type of the memmap array

    Returns:
        Required eagle array from numpy memmap location
    """
    tmp = np.memmap(name, dtype=dtype, mode="r", shape=array_shape)
    return tmp


def get_files(fileType: str, directory: str, tag: str) -> List[str]:
    """Fetch filename for the different eagle outputs for reading.

    Args:
        fileType (str):
            type of file in the eagle outputs
        directory (str):
            eagle data file location
        tag (string):
            snapshot tag to load

    Returns:
        array of files to read
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
    elif fileType in ["SUBFIND_PHOTOMETRY", "PHOTOMETRY"]:
        files = glob.glob(
            "%s/photometry_%s/eagle_subfind_photometry_%s*.hdf5"
            % (directory, tag, tag)
        )
    else:
        raise ValueError("Type of files not supported")

    return sorted(files, key=lambda x: int(re.findall(r"(\d+)", x)[-2]))


def read_hdf5(filename: str, dataset: str) -> NDArray[Any]:
    """Read the required dataset from the eagle hdf5 file.

    Args:
        filename (str):
            name of the hdf5 file
        dataset (str):
            name of the dataset to extract

    Returns:
        Numpy array of the required hdf5 dataset
    """
    with h5py.File(filename, "r") as hf:
        dat = np.array(hf.get(dataset))
        # return empty array if empty dataset
        if dat.ndim == 0:
            return np.array([])

    return dat


def read_array(
    ftype: str,
    directory: str,
    tag: str,
    dataset: str,
    numThreads: int = 1,
    noH: bool = False,
    physicalUnits: bool = False,
    CGS: bool = False,
    photUnits: bool = False,
    verbose: bool = True,
) -> Union[NDArray[Any], unyt_array[unyt_quantity]]:
    """Read the required dataset from the eagle hdf5 file.

    Args:
        ftype (str):
            eagle file type to read
        directory (str):
            location of the eagle simulation directory
        tag (str):
            snapshot tag to read
        dataset (str):
            name of the dataset to read
        numThreads (int):
            number threads to use
        noH (bool):
            remove any reduced Hubble factors
        physicalUnits (bool):
            return in physical units
        CGS (bool):
            return in CGS units
        photUnits (bool):
            return in photometric units
        verbose (bool):
            verbose condition

    Returns:
        Numpy/unyt array from eagle hdf5 filetype
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
                float(tag[5:].replace("p", ".")),
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
    if photUnits:
        dat = apply_PhotUnits(
            files[non_empty_file_indices[0]], dataset, dat, verbose=verbose
        )
    return dat


def read_header(
    ftype: str,
    directory: str,
    tag: str,
    dataset: str,
) -> NDArray[Any]:
    """Read the required header dataset from the eagle hdf5 file.

    Ftype (str):
        eagle file type to read
    directory (str):
        location of the eagle simulation directory
    tag (str):
        snapshot tag to read
    dataset (str):
        name of the dataset to read

    Returns:
        Reads in required eagle hdf5 header data
    """
    files = get_files(ftype, directory, tag)
    with h5py.File(files[0], "r") as hf:
        hdr = hf["Header"].attrs[dataset]

    return hdr


def apply_physicalUnits_conversion(
    filename: str,
    dataset: str,
    dat: NDArray[Any],
    verbose: bool = True,
) -> NDArray[Any]:
    """Apply physical conversion to the dataset.

    Args:
        filename (str):
            filename to read from
        dataset (str):
            dataset to read attribute
        dat (array):
            dataset array to apply conversion
        verbose (bool):
            verbose condition

    Returns:
        Numpy array of the dataset converted to physical units
    """
    with h5py.File(filename, "r") as hf:
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


def apply_hfreeUnits_conversion(
    filename: str,
    dataset: str,
    dat: NDArray[Any],
    verbose: bool = True,
) -> NDArray[Any]:
    """Read the required dataset from the eagle hdf5 file.

    Args:
        filename (str):
            filename to read from
        dataset (str):
            dataset to read attribute
        dat (array):
            dataset array to apply conversion
        verbose (bool):
            verbose condition

    Returns:
        Numpy array of the dataset converted to `h` free units
    """
    with h5py.File(filename, "r") as hf:
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


def apply_CGSUnits_conversion(
    filename: str,
    dataset: str,
    dat: NDArray[Any],
    verbose: bool = True,
) -> NDArray[Any]:
    """Apply CGS conversion to the dataset.

    Args:
        filename (str):
            filename to read from
        dataset (str):
            dataset to read attribute
        dat (array):
            dataset array to apply conversion
        verbose (bool):
            verbose condition

    Returns:
        Numpy array of the dataset converted to CGS units
    """
    with h5py.File(filename, "r") as hf:
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


def apply_PhotUnits(
    filename: str,
    dataset: str,
    dat: NDArray[Any],
    verbose: bool = True,
) -> unyt_array[unyt_quantity]:
    """Apply photometric units to the dataset.

    Args:
        filename (str):
            filename to read from
        dataset (str):
            dataset to read attribute
        dat (array):
            dataset array to apply conversion
        verbose (bool):
            verbose condition

    Returns:
        Unyt array of the dataset in the photometric units
    """
    with h5py.File(filename, "r") as hf:
        unit = hf[dataset].attrs["Units"]

    return unyt_array(dat, unit)


def get_star_formation_time(scale_factor: float) -> float:
    """Convert scale factor to z.

    Args:
        scale_factor (float):
            scale factor of the star particle

    Returns:
        age of the star particle in years
    """
    SFz = (1.0 / scale_factor) - 1.0
    return cosmo.age(SFz).value


def get_age(
    scale_factors: NDArray[Any],
    z: float,
    args: namedtuple,
    numThreads: int = 4,
) -> NDArray[Any]:
    """Convert scale factor to z.

    Args:
        scale_factors (array):
            scale factor of the star particle
        z (float):
            redshift
        args (namedtuple):
            parser arguments passed on to this job
        numThreads (int):
            number of threads to use

    Returns:
        array of ages of the star particles in years
    """
    if args.age_lookup:
        table = np.load(args.age_lookup)
        scale_factor = table["scale_factor"]
        ages = table["ages"]
        Age = cosmo.age(z).value - lookup_age(
            scale_factors, scale_factor, ages
        )

    else:
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


def assign_galaxy_prop(
    ii: int,
    zed: float,
    boxl: float,
    aperture: float,
    grpno: NDArray[np.int32],
    sgrpno: NDArray[np.int32],
    cop: NDArray[np.float32],
    s_grpno: NDArray[np.int32],
    s_sgrpno: NDArray[np.int32],
    s_imasses: NDArray[np.float32],
    s_masses: NDArray[np.float32],
    s_ages: NDArray[np.float32],
    s_Zsmooth: NDArray[np.float32],
    s_coords: NDArray[np.float32],
    s_hsml: NDArray[np.float32],
    s_oxygen: NDArray[np.float32],
    s_hydrogen: NDArray[np.float32],
    s_velocities: NDArray[np.float32],
    g_grpno: NDArray[np.int32],
    g_sgrpno: NDArray[np.int32],
    g_masses: NDArray[np.float32],
    g_Zsmooth: NDArray[np.float32],
    g_sfr: NDArray[np.float32],
    g_coords: NDArray[np.float32],
    g_hsml: NDArray[np.float32],
    verbose: bool,
    s_kwargs: Dict = {},
    g_kwargs: Dict = {},
) -> Galaxy:
    """Load stellar and gas particle data into synthesizer galaxy object.

    Args:
        ii (int):
            galaxy number
        zed (float):
            redshift
        boxl (float):
            simulation box side length
        aperture (float):
            aperture to use from centre of potential
        grpno (array):
            Group numbers in this chunk
        sgrpno (array):
            Subgroup numbers in this chunk
        cop (array):
            centre of potential of subhalos in this chunk
        s_grpno (array):
            Stellar particle group numbers
        s_sgrpno (array):
            Stellar particle subgroup numbers
        s_imasses (np.ndarray):
            Stellar particle initial masses in Msun
        s_masses (np.ndarray):
            Stellar particle current masses in Msun
        s_ages (np.ndarray):
            Stellar particle ages in Gyr
        s_Zsmooth (np.ndarray):
            Stellar particle smoothed metallicity
        s_coords (np.ndarray):
            Stellar particle coordinates in pMpc
        s_hsml(np.ndarray):
            Stellar particle smoothing length in pMpc
        s_oxygen (np.ndarray):
            Stellar particle abundance in oxygen
        s_hydrogen (np.ndarray):
            Stellar particle abundance in hydrogen
        s_velocities (np.ndarray):
            Stellar Velocities
        g_grpno (np.ndarray):
            Gas particle group number
        g_sgrpno (np.ndarray):
            Gas particle subgroup number
        g_masses (np.ndarray):
            Gas particle masses in Msun
        g_Zsmooth (np.ndarray):
            Gas particle smoothed metallicity
        g_sfr (np.ndarray):
            Gas particle instantaneous SFR in Msun/yr
        g_coords: (np.ndarray):
            Gas particle coordinates in pMpc
        g_hsml(np.ndarray):
            Gas particle smoothing length in pMpc
        verbose (bool):
            Are we talking?
        s_kwargs (dict):
            kwargs for stars
        g_kwargs (dict):
            kwargs for gas

    Returns:
        Galaxy
            synthesizer galaxy object
    """
    galaxy = Galaxy(redshift=zed, verbose=verbose)

    # set the bounds of the box length
    bounds = np.array([boxl, boxl, boxl])

    # Fill individual galaxy objects with star particles
    # mask for current galaxy
    ok = np.where((s_grpno == grpno[ii]) * (s_sgrpno == sgrpno[ii]))[0]
    # correcting for periodic boundary
    r = cop[ii] - s_coords[ok]
    r = np.sign(r) * np.min(np.dstack(((r) % bounds, (-r) % bounds)), axis=2)
    # mask for aperture
    ok = ok[norm(r, axis=1) <= aperture]

    # Assign stellar properties
    galaxy.load_stars(
        initial_masses=s_imasses[ok] * Msun,
        current_masses=s_masses[ok] * Msun,
        ages=s_ages[ok] * 1e9 * yr,
        metallicities=s_Zsmooth[ok],
        coordinates=s_coords[ok] * Mpc,
        smoothing_lengths=s_hsml[ok] * Mpc,
        s_oxygen=s_oxygen[ok],
        s_hydrogen=s_hydrogen[ok],
        velocities=s_velocities[ok] * km / s,
        **s_kwargs,
    )

    # Fill individual galaxy objects with gas particles
    sfr_flag = g_sfr > 0
    # mask for current galaxy
    ok = np.where((g_grpno == grpno[ii]) * (g_sgrpno == sgrpno[ii]))[0]
    # correcting for periodic boundary
    r = cop[ii] - g_coords[ok]
    r = np.sign(r) * np.min(np.dstack(((r) % bounds, (-r) % bounds)), axis=2)
    # # mask for aperture
    ok = ok[norm(r, axis=1) <= aperture]

    # Assign gas particle properties
    galaxy.load_gas(
        masses=g_masses[ok] * Msun,
        metallicities=g_Zsmooth[ok],
        star_forming=sfr_flag[ok],
        coordinates=g_coords[ok] * Mpc,
        smoothing_lengths=g_hsml[ok] * Mpc,
        **g_kwargs,
    )

    galaxy.gas.dust_to_metal_ratio = galaxy.dust_to_metal_vijayan19()

    return galaxy
