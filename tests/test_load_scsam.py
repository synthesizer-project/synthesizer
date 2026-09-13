"""Tests for the SC-SAM load_data submodule."""

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.load_data.load_scsam import load_SCSAM
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy

LOG10Z = [-1.0, 0.0]  # log10(Z / 0.02)
CENTRES = [0.005, 0.015, 0.025, 0.035]  # Gyr, 10 Myr bins
SFH = {  # (halo_ind, birthhalo_id, z): (nage, nz) mass in 1e9 Msun
    (37, 100, 5.0): [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.5, 0.5]],
    (41, 200, 5.0): [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
    (45, 300, 5.0): [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
}


@pytest.fixture
def scsam_file(tmp_path):
    """Write a synthetic SC-SAM sfhist file."""
    lines = [
        f"{len(LOG10Z)} {len(CENTRES)}",
        " ".join(map(str, LOG10Z)),
        " ".join(map(str, CENTRES)),
    ]
    for (halo, birth, z), sfh in SFH.items():
        lines.append(f"{halo} {birth} {z}")
        lines += [" ".join(map(str, row)) for row in sfh]
    fname = tmp_path / "sfhist.dat"
    fname.write_text("\n".join(lines) + "\n")
    return str(fname)


def test_particle(scsam_file, test_grid):
    """Cells become particles split across the grid ages they contain."""
    galaxies, halo_inds, birth_ids = load_SCSAM(
        scsam_file, "particle", test_grid
    )
    assert halo_inds == [37, 41, 45]
    assert birth_ids == [100, 200, 300]
    assert all(g.redshift == 5.0 for g in galaxies)
    assert galaxies[2].stars is None

    # Galaxy 41: one cell in [20, 30] Myr at Z = 0.002; the bin contains
    # the grid age 10^7.4 yr so it becomes two particles split by width
    stars = galaxies[1].stars
    assert stars.nstars == 2
    cut = 10**7.4
    np.testing.assert_allclose(
        np.sort(stars.initial_masses.to("Msun").value),
        np.sort([cut - 2e7, 3e7 - cut]) / 1e7 * 3e9,
    )
    np.testing.assert_allclose(stars.metallicities, 0.002)
    np.testing.assert_allclose(
        stars.ages.to("yr").value, [0.5 * (2e7 + cut), 0.5 * (cut + 3e7)]
    )

    # Galaxy 37: mass and metallicity per Z bin are conserved
    stars = galaxies[0].stars
    mass = stars.initial_masses.to("Msun").value
    np.testing.assert_allclose(mass[stars.metallicities == 0.002].sum(), 1.5e9)
    np.testing.assert_allclose(mass[stars.metallicities == 0.02].sum(), 2.5e9)


def test_parametric(scsam_file, test_grid):
    """Parametric method conserves mass and agrees with the particles."""
    particle, _, _ = load_SCSAM(scsam_file, "particle", test_grid)
    parametric, _, _ = load_SCSAM(scsam_file, "parametric", test_grid)
    ages = 10**test_grid.log10ages
    assert parametric[2].stars is None
    for pgal, gal in zip(particle[:2], parametric[:2]):
        assert isinstance(gal, ParametricGalaxy)
        pmass = pgal.stars.initial_masses.to("Msun").value
        np.testing.assert_allclose(gal.stars.sfzh.sum(), pmass.sum())
        sfh = gal.stars.sfzh.sum(axis=1)
        np.testing.assert_allclose(
            np.average(ages, weights=sfh),
            np.average(pgal.stars.ages.to("yr").value, weights=pmass),
            rtol=0.05,
        )


def test_deprecated_aliases(scsam_file, test_grid):
    """The old interpolation method names warn and route to parametric."""
    for method in ("parametric_NNI", "parametric_RGI"):
        with pytest.warns(DeprecationWarning):
            galaxies, _, _ = load_SCSAM(scsam_file, method, test_grid)
        assert isinstance(galaxies[0], ParametricGalaxy)


def test_bad_method(scsam_file, test_grid):
    """An unknown method raises."""
    with pytest.raises(exceptions.InconsistentArguments):
        load_SCSAM(scsam_file, "spherical_cow", test_grid)
