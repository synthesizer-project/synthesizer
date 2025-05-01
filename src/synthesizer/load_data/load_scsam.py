"""A submodule for loading SC-SAM data into Synthesizer.

Currently implemented are loading methods for
- SC-SAM (using a parametric method)
- SC-SAM (using a particle method)
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator as NNI
from scipy.interpolate import RegularGridInterpolator as RGI
from unyt import Msun, yr

from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars


def load_SCSAM(fname, method, grid=None, verbose=False):
    r"""Read an SC-SAM star formation data file.

    Returns a list of galaxy objects, halo indices, and birth halo IDs.
    Adapted from code by Aaron Yung.

    Args:
        fname (str):
            The SC-SAM star formation data file to be read.
        method (str):
            'particle', 'parametric_NNI' or 'parametric_RGI', depending on how
            you wish to model your SFZH. 'particle' treats each age-Z bin as a
            particle. 'parametric_NNI' uses scipy's nearest ND interpolator to
            interpolate the grid for a parametric SFH 'parametric_RGI' uses
            scipy's regular grid interpolator to interpolate the grid for a
            parametric SFH.
        grid (grid object):
            Grid object to extract from (needed for parametric galaxies).
        verbose (bool):
            Are we talking?

    Returns:
        tuple:
            galaxies (list):
                list of galaxy objects
            halo_ind_list (list):
                list of halo indices
            birthhalo_id_list (list):
                birth halo indices
    """
    # Prepare to read SFHist file
    sfhist = open(fname, "r")
    lines = sfhist.readlines()

    # Set up halo index, birth halo ID, redshift and galaxy object lists
    halo_ind_list = []
    birthhalo_id_list = []
    redshift_list = []
    galaxies = []

    # Line counter
    count = 0
    count_gal = 0

    # Read file line by line as it can be large
    for line in lines:
        # Get age-metallicity grid structure
        if count == 0:
            Z_len, age_len = [int(i) for i in line.split()]
            if verbose:
                print(
                    f"There are {Z_len} metallicity bins and "
                    f"{age_len} age bins."
                )

        # Get metallicity bins
        if count == 1:
            Z_lst = [float(i) for i in line.split()]  # logZ in solar units
            if verbose:
                print(f"Z_lst (log10Zsun): {Z_lst}")
            # Check that this agrees with the expected grid structure
            if len(Z_lst) != Z_len:
                print("Wrong number of Z bins.")
                break
            Z_sun = 0.02  # Solar metallicity
            Z_lst = 10 ** np.array(Z_lst) * Z_sun  # Unitless
            if verbose:
                print(f"Z_lst (unitless): {Z_lst}")

        # Get age bins
        if count == 2:
            age_lst = [float(i) for i in line.split()]  # Gyr
            if verbose:
                print(f"age_lst: {age_lst}")
            # Check that this agrees with the expected grid structure
            if len(age_lst) != age_len:
                print("Wrong number of age bins.")
                break

        # Get galaxy data
        # The data chunk for each galaxy consists of one header line,
        # followed by the age x Z grid.
        # Thus it takes up age_len+1 lines.
        if (count - 3) % (age_len + 1) == 0:
            # The line preceding each age x Z grid contains:
            halo_ind = int(line.split()[0])
            birthhalo_id = int(line.split()[1])
            redshift = float(line.split()[2])

            # Append this information to its respective list
            halo_ind_list.append(halo_ind)
            birthhalo_id_list.append(birthhalo_id)
            redshift_list.append(redshift)

            # Start a new SFH array, specific to the galaxy
            SFH = []

        # If the age x Z grid of a galaxy is being read:
        if (count - 3) % (age_len + 1) != 0 and count > 3:
            _grid = [float(i) for i in line.split()]
            SFH.append(_grid)

        # If the last line of an age x Z grid has been read:
        # (i.e we now have the full grid of a single galaxy)
        if (count - 3) % (age_len + 1) == age_len and count > 3:
            count_gal += 1

            # Create galaxy object
            if method == "particle":
                galaxy = _load_SCSAM_particle_galaxy(
                    SFH, age_lst, Z_lst, verbose=verbose
                )
            elif method == "parametric_NNI":
                galaxy = _load_SCSAM_parametric_galaxy(
                    SFH, age_lst, Z_lst, "NNI", grid, verbose=verbose
                )
            elif method == "parametric_RGI":
                galaxy = _load_SCSAM_parametric_galaxy(
                    SFH, age_lst, Z_lst, "RGI", grid, verbose=verbose
                )

            # Append to list of galaxy objects
            galaxies.append(galaxy)

        count += 1

    return galaxies, halo_ind_list, birthhalo_id_list


def _load_SCSAM_particle_galaxy(SFH, age_lst, Z_lst, verbose=False):
    """Treat each age-Z bin as a particle.

    Args:
        SFH (unyt_array):
            age x Z SFH array as given by SC-SAM for a single galaxy.
        age_lst (list):
            age bins in the SFH array (Gyr).
        Z_lst (list):
            metallicity bins in the SFH array (unitless).
        verbose (bool):
            Are we talking?
    """
    # Initialise arrays for storing particle information
    p_imass = []  # initial mass
    p_age = []  # age
    p_Z = []  # metallicity

    # Get length of arrays
    age_len = len(age_lst)
    Z_len = len(Z_lst)

    # Iterate through every point on the grid
    if verbose:
        print("Iterating through grid...")
    for age_ind in range(age_len):
        for Z_ind in range(Z_len):
            if SFH[age_ind][Z_ind] == 0:
                continue
            else:
                p_imass.append(SFH[age_ind][Z_ind])  # 10^9 Msun
                p_age.append(age_lst[age_ind])  # Gyr
                p_Z.append(Z_lst[Z_ind])  # unitless

    # Convert units
    if verbose:
        print("Converting units...")
    p_imass = np.array(p_imass) * 10**9  # Msun
    p_age = np.array(p_age) * 10**9  # yr
    p_Z = np.array(p_Z)  # unitless

    if verbose:
        print("Generating SED...")

    # Create stars object
    stars = ParticleStars(
        initial_masses=p_imass * Msun, ages=p_age * yr, metallicities=p_Z
    )

    if verbose:
        print("Creating galaxy object...")
    # Create galaxy object
    particle_galaxy = ParticleGalaxy(stars=stars)

    return particle_galaxy


def _load_SCSAM_parametric_galaxy(
    SFH,
    age_lst,
    Z_lst,
    method,
    grid,
    verbose=False,
):
    """Obtain galaxy SED using the parametric method.

    This is done by interpolating the grid.
    Returns a galaxy object.
    Adapted from code by Kartheik Iyer.

    Args:
        SFH (unyt_array):
            age x Z SFH array as given by SC-SAM for a single galaxy.
        age_lst (list):
            age bins in the SFH array (Gyr).
        Z_lst (list):
            metallicity bins in the SFH array (unitless).
        method (str):
            'parametric_NNI' or 'parametric_RGI', depending on how you wish to
            model your SFZH.
        grid (grid object):
            Grid object to extract from.
        verbose (bool):
            Are we talking?
    """
    # This the grid that we want to interpolate to
    new_age = 10**grid.log10age  # yr
    new_Z = np.log10(grid.metallicity)  # log10Z

    # This is the old grid, to be interpolated
    old_age = np.array(age_lst) * 10**9  # yr
    old_Z = np.log10(Z_lst)  # log10Z

    # Convert SFH units
    SFH = np.array(SFH) * 10**9  # Msun
    # sum_SFH = np.log10(np.sum(SFH))

    # Using regular grid interpolator
    if method == "RGI":
        # Get coords for new grid
        new_X, new_Y = np.meshgrid(new_age, new_Z, indexing="ij")
        # Set up the old grid for interpolation
        interp_obj = RGI(
            points=(old_age, old_Z), values=SFH, bounds_error=False
        )
        # Interpolate to new grid coordinates and reshape
        new_SFH = interp_obj((new_X.ravel(), new_Y.ravel()))
        new_SFH = new_SFH.reshape(len(new_age), len(new_Z))
        # Convert NaNs to zero
        new_SFH[np.isnan(new_SFH)] = 0.0

    # Using nearest ND interpolator
    if method == "NNI":
        # Get coords for new grid
        new_X, new_Y = np.meshgrid(new_age, new_Z, indexing="ij")
        # Get list of coordinates of old grid
        old_X, old_Y = np.meshgrid(old_age, old_Z, indexing="ij")
        old_coords = np.vstack([old_X.ravel(), old_Y.ravel()]).T
        # Set up the old grid for interpolation
        interp_obj = NNI(old_coords, SFH.flatten(), rescale=True)
        # Interpolate to new grid
        new_SFH = interp_obj(new_X, new_Y)

    # Normalise the new grid
    norm_SFH = np.sum(SFH) / np.sum(new_SFH)
    new_SFH *= norm_SFH

    # Create Binned SFZH object
    stars = ParametricStars(
        log10ages=grid.log10age,
        metallicities=grid.metallicity,
        sfzh=new_SFH,
    )

    # Create galaxy object
    parametric_galaxy = ParametricGalaxy(stars)

    return parametric_galaxy
