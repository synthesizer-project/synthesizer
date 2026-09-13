"""A submodule for loading SHARK data into Synthesizer.

SHARK (Lagos+ 2018, https://github.com/ICRAR/shark) writes per-galaxy
star formation and metal enrichment histories to
``star_formation_histories.hdf5`` when run with
``output_sf_histories = true``. Each galaxy's history is stored per
component (disk, merger-driven bulge, disk-instability-driven bulge) as
the star formation rate and the metallicity of stars formed in each
inter-snapshot time bin.

Example usage::

    from synthesizer.load_data.load_shark import load_SHARK

    # Particle galaxies (one "particle" per non-zero time bin, split
    # across the grid ages where the bins are wider than the grid spacing)
    galaxies = load_SHARK("star_formation_histories.hdf5", grid)

    # Parametric galaxies binned onto the SPS grid
    galaxies = load_SHARK(
        "star_formation_histories.hdf5", grid, method="parametric"
    )

Notes:
    - The SFHs are gross star formation (mass formed), so total initial
      mass exceeds the (post-recycling) stellar masses in the SHARK
      ``galaxies.hdf5`` catalogue by a factor 1 / (1 - recycle).
    - Rows in ``star_formation_histories.hdf5`` are a subset of the
      ``galaxies.hdf5`` catalogue in a different order; join catalogue
      properties using the ``id_galaxy`` attribute attached to each
      returned galaxy, never by row position.
    - SFRs here are in Msun / yr / h, unlike the catalogue ``sfr_*``
      datasets which are in Msun / Gyr / h.
    - The time bins are contiguous and the last one ends at the output
      snapshot, so ages are built from ``delta_t`` alone. (``lbt_mean``
      is the lookback time from z=0, not from the output.)
"""

import h5py
import numpy as np
from unyt import Msun, yr

from synthesizer import exceptions
from synthesizer.load_data.utils import (
    bin_overlap_matrix,
    cic_matrix,
    split_age_bins,
)
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars

_COMPONENTS = ("disks", "bulges_mergers", "bulges_diskins")


def load_SHARK(
    fname,
    grid,
    method="particle",
    components=_COMPONENTS,
    verbose=False,
):
    """Read a SHARK star formation histories file.

    Each (time bin, component) entry is a top-hat of star formation with
    mass ``SFR * delta_t / h`` (converted to Msun) and the metallicity of
    the stars formed in that bin. Bins with zero star formation are
    dropped (SHARK zero-fills bins before a galaxy forms).

    Args:
        fname (str):
            The SHARK ``star_formation_histories.hdf5`` file to read.
        grid (Grid):
            Grid whose age and metallicity axes define the SFZH, and
            whose age spacing sets how finely the SHARK bins are split.
        method (str):
            'particle' (default) returns particle galaxies with one
            particle per non-zero bin per component, split at every grid
            age the bin contains so young star formation is resolved.
            'parametric' integrates each bin over the grid age cells
            (the ``parametric.Stars`` convention) and returns parametric
            galaxies. Note the per-particle component tags are lost in
            the combined SFZH; pass e.g. ``components=("disks",)`` for
            per-component parametric galaxies.
        components (tuple):
            SHARK components to include, a subset of
            ('disks', 'bulges_mergers', 'bulges_diskins').
        verbose (bool):
            Are we talking?

    Returns:
        list: particle.Galaxy (method='particle') or parametric.Galaxy
            (method='parametric') objects, in file row order, each with
            an ``id_galaxy`` attribute. Particle stars carry a
            ``star_component`` array indexing ``components``.

    Raises:
        InconsistentArguments:
            If method is unknown or components contains an unknown
            component.
    """
    if method not in ("particle", "parametric"):
        raise exceptions.InconsistentArguments(
            f"Unknown method '{method}' (use 'particle' or 'parametric')"
        )
    unknown = set(components) - set(_COMPONENTS)
    if unknown:
        raise exceptions.InconsistentArguments(
            f"Unknown components {unknown} (available: {_COMPONENTS})"
        )

    with h5py.File(fname, "r") as hf:
        h = float(hf["cosmology/h"][()])
        redshift = float(hf["run_info/redshift"][()])
        delta_t = hf["delta_t"][:].astype(np.float64) * 1e9  # yr
        gal_ids = hf["galaxies/id_galaxy"][:]
        sfrs = [
            hf[f"{comp}/star_formation_rate_histories"][:]
            for comp in components
        ]  # Msun / yr / h
        zmets = [
            hf[f"{comp}/metallicity_histories"][:] for comp in components
        ]  # absolute Z of stars formed per bin

    ncomp = len(components)

    # Mass formed per bin [Msun]: SFR [Msun/yr/h] * delta_t [yr]
    mass = np.concatenate(sfrs, axis=1) * np.tile(delta_t / h, ncomp)
    zmet = np.concatenate(zmets, axis=1).astype(np.float64)
    tags = np.repeat(np.arange(ncomp, dtype=np.int8), delta_t.size)

    # Bin edges as ages at the output [yr]: the bins are contiguous and
    # the last one ends at the output snapshot
    hi = np.cumsum(delta_t[::-1])[::-1]
    lo = np.tile(hi - delta_t, ncomp)
    hi = np.tile(hi, ncomp)
    grid_ages = 10**grid.log10ages

    if method == "particle":
        # Resolve bins wider than the grid spacing into one particle per
        # grid interval; narrower bins stay one particle at their midpoint
        b, age, frac = split_age_bins(lo, hi, grid_ages)
        mass, zmet, tags = mass[:, b] * frac, zmet[:, b], tags[b]
    else:
        overlap = bin_overlap_matrix(lo, hi, grid_ages)
        log10zs = np.log10(grid.metallicities)

    if verbose:
        print(
            f"Loading {gal_ids.size} galaxies "
            f"({np.count_nonzero(mass)} particles) at z={redshift:.3f}"
        )

    galaxies = []
    for gal_id, m, z in zip(gal_ids, mass, zmet):
        keep = m > 0
        stars = None
        if not keep.any():
            if verbose:
                print(f"Galaxy {gal_id} formed no stars in {components}")
        elif method == "particle":
            stars = ParticleStars(
                initial_masses=m[keep] * Msun,
                ages=age[keep] * yr,
                metallicities=z[keep],
                redshift=redshift,
                star_component=tags[keep],
            )
        else:
            # Z = 0 bins clamp to the lowest grid metallicity
            with np.errstate(divide="ignore"):
                zw = cic_matrix(np.log10(z[keep]), log10zs)
            stars = ParametricStars(
                grid.log10ages,
                grid.metallicities,
                sfzh=overlap[keep].T @ (m[keep, None] * zw),
            )

        galaxy_cls = (
            ParticleGalaxy if method == "particle" else ParametricGalaxy
        )
        galaxies.append(
            galaxy_cls(stars=stars, redshift=redshift, id_galaxy=int(gal_id))
        )

    return galaxies
