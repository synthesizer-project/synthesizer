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

    # Particle galaxies (one "particle" per non-zero time bin)
    galaxies = load_SHARK("star_formation_histories.hdf5")

    # Parametric galaxies binned onto an SPS grid
    galaxies = load_SHARK(
        "star_formation_histories.hdf5", method="parametric", grid=grid
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
"""

import h5py
import numpy as np
from unyt import Msun, yr

from synthesizer import exceptions
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars

_COMPONENTS = ("disks", "bulges_mergers", "bulges_diskins")


def load_SHARK(
    fname,
    method="particle",
    grid=None,
    components=_COMPONENTS,
    verbose=False,
):
    """Read a SHARK star formation histories file.

    Each non-zero (time bin, component) entry becomes one stellar
    "particle" with initial mass ``SFR * delta_t / h`` (converted to
    Msun), age equal to the lookback time to the bin midpoint
    (``lbt_mean``), and the metallicity of stars formed in that bin.
    Bins with zero star formation are dropped (SHARK zero-fills bins
    before a galaxy forms).

    Args:
        fname (str):
            The SHARK ``star_formation_histories.hdf5`` file to read.
        method (str):
            'particle' (default) returns particle galaxies with one
            particle per non-zero time bin per component. 'parametric'
            additionally bins those particles onto the given SPS grid
            axes (mass conserving) and returns parametric galaxies.
            Note the per-particle component tags are lost in the
            combined SFZH; pass e.g. ``components=("disks",)`` for
            per-component parametric galaxies.
        grid (Grid):
            Grid whose age and metallicity axes define the SFZH
            (required for method='parametric').
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
            If method is unknown, or method='parametric' without a
            grid, or components contains an unknown component.
    """
    if method not in ("particle", "parametric"):
        raise exceptions.InconsistentArguments(
            f"Unknown method '{method}' (use 'particle' or 'parametric')"
        )
    if method == "parametric" and grid is None:
        raise exceptions.InconsistentArguments(
            "method='parametric' requires a grid"
        )
    unknown = set(components) - set(_COMPONENTS)
    if unknown:
        raise exceptions.InconsistentArguments(
            f"Unknown components {unknown} (available: {_COMPONENTS})"
        )

    with h5py.File(fname, "r") as hf:
        h = float(hf["cosmology/h"][()])
        redshift = float(hf["run_info/redshift"][()])
        lbt_mean = hf["lbt_mean"][:].astype(np.float64)  # Gyr
        delta_t = hf["delta_t"][:].astype(np.float64)  # Gyr
        gal_ids = hf["galaxies/id_galaxy"][:]
        sfrs = [
            hf[f"{comp}/star_formation_rate_histories"][:]
            for comp in components
        ]  # Msun / yr / h
        zmets = [
            hf[f"{comp}/metallicity_histories"][:] for comp in components
        ]  # absolute Z of stars formed per bin

    nbins = lbt_mean.size
    ncomp = len(components)

    # Mass formed per bin [Msun]: SFR [Msun/yr/h] * delta_t [Gyr]
    mass = np.concatenate(
        [sfr.astype(np.float64) * delta_t * 1e9 / h for sfr in sfrs],
        axis=1,
    )  # (ngal, ncomp * nbins)
    zmet = np.concatenate(zmets, axis=1).astype(np.float64)

    # Stellar age at the output snapshot = lookback time to bin midpoint
    ages = np.tile(lbt_mean * 1e9, ncomp)  # yr
    tags = np.repeat(np.arange(ncomp, dtype=np.int8), nbins)

    # Only bins with star formation become particles (zeros pad bins
    # before the galaxy formed, and would otherwise be metallicity
    # floored)
    mask = mass > 0

    # Boolean-masking a 2D array flattens in row-major order, so
    # particles are galaxy-contiguous; index galaxy i via cumulative
    # per-row counts
    begin = np.zeros(mask.shape[0] + 1, dtype=np.int64)
    np.cumsum(mask.sum(axis=1), out=begin[1:])
    flat_mass = mass[mask]
    flat_z = zmet[mask]
    flat_age = np.broadcast_to(ages, mass.shape)[mask]
    flat_tag = np.broadcast_to(tags, mass.shape)[mask]

    if verbose:
        print(
            f"Loading {gal_ids.size} galaxies "
            f"({flat_mass.size} particles) at z={redshift:.3f}"
        )

    galaxies = []
    for i, gal_id in enumerate(gal_ids):
        b, e = begin[i], begin[i + 1]
        stars = None
        if e > b:
            stars = ParticleStars(
                initial_masses=flat_mass[b:e] * Msun,
                ages=flat_age[b:e] * yr,
                metallicities=flat_z[b:e],
                redshift=redshift,
                star_component=flat_tag[b:e],
            )
        elif verbose:
            print(f"Galaxy {gal_id} formed no stars in {components}")

        if method == "parametric":
            if stars is not None:
                stars = stars.get_sfzh(grid.log10ages, grid.metallicities)
            gal = ParametricGalaxy(stars=stars, redshift=redshift)
        else:
            gal = ParticleGalaxy(stars=stars, redshift=redshift)
        gal.id_galaxy = int(gal_id)
        galaxies.append(gal)

    return galaxies
