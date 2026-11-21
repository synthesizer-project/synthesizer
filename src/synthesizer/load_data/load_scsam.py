"""A submodule for loading SC-SAM data into Synthesizer.

SC-SAM ``sfhist`` files tabulate, per galaxy, the stellar mass formed in
each (age, metallicity) bin. The file starts with three header lines
(the grid shape, the metallicity bins in log10(Z / 0.02) and the age-bin
centres in Gyr), then per galaxy a line of ``halo_ind birthhalo_id
redshift`` followed by the age x Z mass grid in 10^9 Msun.

Example usage::

    from synthesizer.load_data.load_scsam import load_SCSAM

    galaxies, halo_inds, birthhalo_ids = load_SCSAM(
        "sfhist.dat", "particle", grid
    )
"""

import warnings

import numpy as np
from unyt import Msun, yr

from synthesizer import exceptions
from synthesizer.load_data.utils import split_age_bins
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars

_ZSUN = 0.02  # SC-SAM solar metallicity


def load_SCSAM(fname, method, grid, verbose=False, dtype=np.float64):
    """Read an SC-SAM star formation history file.

    Each (age bin, Z bin) cell is a top-hat of star formation between the
    bin edges (midpoints between the tabulated centres). Adapted from
    code by Aaron Yung.

    Args:
        fname (str):
            The SC-SAM ``sfhist`` file to read.
        method (str):
            'particle' returns particle galaxies with one particle per
            non-zero cell, split at every grid age the cell's age bin
            contains so young star formation is resolved. 'parametric'
            returns parametric galaxies holding the cells exactly (via
            ``parametric.Stars.from_binned``, each tabulated metallicity
            a single value). 'parametric_NNI' and 'parametric_RGI' are
            deprecated aliases for 'parametric'.
        grid (Grid):
            Grid whose age and metallicity axes define the SFZH, and
            whose age spacing sets how finely the SC-SAM bins are split.
        verbose (bool):
            Are we talking?
        dtype (type):
            The numpy dtype to cast all numerical particle arrays to.
            Defaults to np.float64 to match standard SPS grids. Set to
            np.float32 (with Grid(use_precision=np.float32)) to reduce
            memory.

    Returns:
        tuple:
            galaxies (list): particle.Galaxy or parametric.Galaxy
                objects in file order, each carrying its redshift.
            halo_ind_list (list): halo indices.
            birthhalo_id_list (list): birth halo IDs.

    Raises:
        InconsistentArguments:
            If method is unknown or the header or payload is inconsistent.
    """
    if method in ("parametric_NNI", "parametric_RGI"):
        warnings.warn(
            f"method='{method}' is deprecated, use 'parametric'",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "parametric"
    if method not in ("particle", "parametric"):
        raise exceptions.InconsistentArguments(
            f"Unknown method '{method}' (use 'particle' or 'parametric')"
        )

    with open(fname) as f:
        lines = f.read().splitlines()
    nz, nage = (int(i) for i in lines[0].split())
    zs = 10 ** np.array(lines[1].split(), dtype=float) * _ZSUN
    centres = np.array(lines[2].split(), dtype=float) * 1e9  # yr
    if zs.size != nz or centres.size != nage:
        raise exceptions.InconsistentArguments(
            f"Header promises {nz} Z and {nage} age bins but lists "
            f"{zs.size} and {centres.size}"
        )

    # Bin edges: zero, the midpoints, and the last bin symmetric about its
    # centre (so a single bin spans zero to twice its centre)
    lo = np.concatenate(([0.0], 0.5 * (centres[1:] + centres[:-1])))
    hi = np.append(lo[1:], 2 * centres[-1] - lo[-1])
    grid_ages = 10**grid.log10ages

    if method == "particle":
        # Resolve bins wider than the grid spacing into one particle per
        # grid interval; narrower bins stay one particle at their midpoint
        b, age, frac = split_age_bins(lo, hi, grid_ages)
    else:
        # Each tabulated metallicity is a single value (a zero width bin)
        zorder = np.argsort(zs)
        metal_edges = np.repeat(zs[zorder], 2)
        age_edges = np.append(lo, hi[-1])

    galaxies, halo_inds, birthhalo_ids = [], [], []
    block = nage + 1
    if (len(lines) - 3) % block:
        raise exceptions.InconsistentArguments(
            f"{len(lines) - 3} galaxy lines is not a whole number of "
            f"{block}-line galaxy blocks"
        )
    for start in range(3, len(lines), block):
        halo_ind, birthhalo_id, redshift = lines[start].split()
        redshift = float(redshift)
        sfh = np.loadtxt(lines[start + 1 : start + block], ndmin=2) * 1e9
        if sfh.shape != (nage, nz):
            raise exceptions.InconsistentArguments(
                f"Halo {halo_ind} SFH has shape {sfh.shape}, "
                f"expected {(nage, nz)}"
            )

        stars = None
        if method == "particle":
            m = sfh[b] * frac[:, None]  # (nsegment, nz)
            keep = m > 0
            if keep.any():
                seg, iz = np.nonzero(keep)
                stars = ParticleStars(
                    initial_masses=(m[keep] * Msun)
                    .in_base("galactic")
                    .astype(dtype),
                    ages=(age[seg] * yr).astype(dtype),
                    metallicities=zs[iz].astype(dtype),
                    redshift=redshift,
                )
            galaxy = ParticleGalaxy(stars=stars, redshift=redshift)
        else:
            if sfh.any():
                masses = np.zeros((nage, 2 * nz - 1))
                masses[:, ::2] = sfh[:, zorder]
                stars = ParametricStars.from_binned(
                    grid.log10ages,
                    grid.metallicities,
                    age_edges,
                    metal_edges,
                    masses,
                )
            galaxy = ParametricGalaxy(stars=stars, redshift=redshift)

        galaxies.append(galaxy)
        halo_inds.append(int(halo_ind))
        birthhalo_ids.append(int(birthhalo_id))

    if verbose:
        print(f"Loaded {len(galaxies)} galaxies")

    return galaxies, halo_inds, birthhalo_ids
