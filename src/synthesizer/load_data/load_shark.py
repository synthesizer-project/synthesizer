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
    - ``lbt_mean`` is the lookback time from z=0, not from the output
      snapshot; ages are shifted so they are relative to the output.
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
    Msun), age equal to the time from the bin midpoint to the output
    snapshot, and the metallicity of stars formed in that bin. Bins
    with zero star formation are dropped (SHARK zero-fills bins before
    a galaxy forms).

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
        lbt_mean = hf["lbt_mean"][:].astype(np.float64)  # Gyr, from z=0
        delta_t = hf["delta_t"][:].astype(np.float64)  # Gyr
        gal_ids = hf["galaxies/id_galaxy"][:]
        sfrs = [
            hf[f"{comp}/star_formation_rate_histories"][:]
            for comp in components
        ]  # Msun / yr / h
        zmets = [
            hf[f"{comp}/metallicity_histories"][:] for comp in components
        ]  # absolute Z of stars formed per bin

    ncomp = len(components)

    # Mass formed per bin [Msun]: SFR [Msun/yr/h] * delta_t [Gyr]
    mass = np.concatenate(sfrs, axis=1) * np.tile(delta_t * 1e9 / h, ncomp)
    zmet = np.concatenate(zmets, axis=1).astype(np.float64)

    # Stellar age at the output snapshot [yr]. lbt_mean is measured from
    # z=0 and the last bin ends at the output, so subtract the lookback
    # time to the output (the last bin's lower edge)
    ages = np.tile(lbt_mean - (lbt_mean[-1] - 0.5 * delta_t[-1]), ncomp)
    ages *= 1e9
    tags = np.repeat(np.arange(ncomp, dtype=np.int8), delta_t.size)

    # Only bins with star formation become particles (zeros pad bins
    # before the galaxy formed, and would otherwise be metallicity
    # floored). nonzero is row-major so particles are galaxy-contiguous
    gi, bi = np.nonzero(mass > 0)
    begin = np.searchsorted(gi, np.arange(gal_ids.size + 1))
    flat_mass, flat_z = mass[gi, bi], zmet[gi, bi]
    flat_age, flat_tag = ages[bi], tags[bi]

    if verbose:
        print(
            f"Loading {gal_ids.size} galaxies "
            f"({gi.size} particles) at z={redshift:.3f}"
        )

    galaxy_cls = ParametricGalaxy if method == "parametric" else ParticleGalaxy
    galaxies = []
    for gal_id, b, e in zip(gal_ids, begin[:-1], begin[1:]):
        stars = None
        if e > b:
            stars = ParticleStars(
                initial_masses=flat_mass[b:e] * Msun,
                ages=flat_age[b:e] * yr,
                metallicities=flat_z[b:e],
                redshift=redshift,
                star_component=flat_tag[b:e],
            )
            if method == "parametric":
                stars = stars.get_sfzh(grid.log10ages, grid.metallicities)
        elif verbose:
            print(f"Galaxy {gal_id} formed no stars in {components}")

        galaxies.append(
            galaxy_cls(stars=stars, redshift=redshift, id_galaxy=int(gal_id))
        )

    return galaxies
