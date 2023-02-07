"""
Script containing all the functions to calculate the line of sight attenuation.

...

Licence..

"""
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange, types, config

config.THREADING_LAYER = "threadsafe"


def kd_los(
    s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins, dimens=(0, 1, 2)
):
    """
    KD-Tree flavour of LOS calculation.
    Compute the los metal surface density (in Msun/Mpc^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Parameters
    ----------
    s_cood: float (Nx3) array
        star particle coordinates, in physical Mpc
    g_cood: float (Nx3) array
        gas particle coordinate, in physical Mpc
    g_mass: float array
        gas mass, in Msolar
    g_Z: float array
        smoothed gas metallicity, unit-less
    g_sml: float array
        smoothing length, in physical Mpc
    lkernel: float array
        kernel value at position
    kbins: int
        number bins of kernel values
    dimens: int
        (0,1,2) -> x, y, z coordinates

    Returns
    -------
    Z_los_SD: float array
        1D array of line-of-sight metal column density of star particles
        along the z-direction
    """

    # Generalise dimensions (function assume LOS along z-axis)
    xdir, ydir, zdir = dimens

    # Get how many stars
    nstar = s_cood.shape[0]

    # Lets build the kd tree from star positions
    tree = cKDTree(s_cood[:, (xdir, ydir)])

    # Query the tree for all gas particles (can now supply multiple rs!)
    query = tree.query_ball_point(g_cood[:, (xdir, ydir)], r=g_sml, p=1)

    # Now we just need to collect each stars neighbours
    star_gas_nbours = {s: [] for s in range(nstar)}
    for g_ind, sparts in enumerate(query):
        for s_ind in sparts:
            star_gas_nbours[s_ind].append(g_ind)

    # Initialise line of sight metal density array
    Z_los_SD = np.zeros(nstar)

    # Loop over stars
    for s_ind in range(nstar):

        # Extract gas particles to consider
        g_inds = star_gas_nbours.pop(s_ind)

        # Extract data for these particles
        thisspos = s_cood[s_ind]
        thisgpos = g_cood[g_inds]
        thisgsml = g_sml[g_inds]
        thisgZ = g_Z[g_inds]
        thisgmass = g_mass[g_inds]

        # We only want to consider particles "in-front" of the star
        ok = np.where(thisgpos[:, zdir] > thisspos[zdir])[0]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgZ = thisgZ[ok]
        thisgmass = thisgmass[ok]

        # Get radii and divide by smooting length
        b = np.linalg.norm(
            thisgpos[:, (xdir, ydir)] - thisspos[((xdir, ydir),)], axis=-1
        )
        boverh = b / thisgsml

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh])

        # Finally get LOS metal surface density in units of Msun/pc^2
        Z_los_SD[s_ind] = np.sum(
            (thisgmass * thisgZ / (thisgsml * thisgsml)) * kernel_vals
        )

    return Z_los_SD


@njit(
    (
        types.float32[:, :],
        types.float32[:, :],
        types.float32[:],
        types.float32[:],
        types.float32[:],
        types.float32[:],
        types.int32,
    ),
    parallel=True,
    nogil=True,
)
def numba_los(
    s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins, dimens=(0, 1, 2)
):
    """
    Original LOS calculation. Faster for smaller particle numbers.
    Compute the los metal surface density (in Msun/Mpc^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Parameters
    ----------
    s_cood: float (Nx3) array
        star particle coordinates, in physical Mpc
    g_cood: float (Nx3) array
        gas particle coordinate, in physical Mpc
    g_mass: float array
        gas mass, in Msolar
    g_Z: float array
        smoothed gas metallicity, unit-less
    g_sml: float array
        smoothing length, in physical Mpc
    lkernel: float array
        kernel value at position
    kbins: int
        number bins of kernel values
    dimens: int
        (0,1,2) -> x, y, z coordinates

    Returns
    -------
    Z_los_SD: float array
        1D array of line-of-sight metal column density of star particles
        along the z-direction
    """

    nstar = len(s_cood)

    # Initialise line of sight metal density array
    Z_los_SD = np.zeros(nstar, dtype=np.float32)

    # Generalise dimensions (function assume LOS along z-axis)
    xdir, ydir, zdir = dimens

    for s_ind in prange(nstar):

        # Extract data for these particles
        thisspos = s_cood[s_ind]
        ok = g_cood[:, zdir] > thisspos[zdir]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]

        # We only want to consider particles "in-front" of the star
        x = thisgpos[:, xdir] - thisspos[xdir]
        y = thisgpos[:, ydir] - thisspos[ydir]

        # Calculate the impact parameter of the LOS on the gas particles
        boverh = np.sqrt(x * x + y * y) / thisgsml

        # LOS that pass through the gas particles
        ok = boverh <= 1.0

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh[ok]])

        # in units of Msun/Mpc^2
        Z_los_SD[s_ind] = np.sum(
            (thisgmass[ok] * thisgZ[ok] / (thisgsml[ok] * thisgsml[ok]))
            * kernel_vals
        )

    return Z_los_SD


def calc_los(npart, kernel_func, kernel_dict):
    """User facing wrapper for line of sight calculations.

    This will choose the best LOS implementation to apply to the
    situaition.
    """

    # If lightweight use lightweight calculation
    if npart < 100:
        los = numba_los()
    else:
        los = kd_los()

    return los
