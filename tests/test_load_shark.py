"""Tests for the SHARK load_data submodule."""

import h5py
import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.load_data.load_shark import load_SHARK
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy

H = 0.7
DELTA_T = np.array([2.0, 2.0, 2.0, 2.0])  # Gyr; bins end at the output
# Age range (Gyr) of each bin at the output, oldest first
BINS = np.array([[6.0, 8.0], [4.0, 6.0], [2.0, 4.0], [0.0, 2.0]])
IDS = np.array([10, 42, 7], dtype=np.int32)
SFR = {  # Msun / yr / h, (ngal, nbins)
    "disks": np.array(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.5, 0.5, 0.0, 0.0]]
    ),
    "bulges_mergers": np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
    "bulges_diskins": np.array(
        [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ),
}


def _zmet(comp_idx):
    """Distinct metallicity per (component, galaxy, bin), 0 where no SF."""
    comp = list(SFR)[comp_idx]
    ngal, nbins = SFR[comp].shape
    z = 0.001 * (comp_idx + 1) * np.arange(1, nbins + 1) * np.ones((ngal, 1))
    return np.where(SFR[comp] > 0, z, 0.0)


@pytest.fixture
def shark_file(tmp_path):
    """Write a synthetic SHARK star_formation_histories.hdf5."""
    fname = tmp_path / "star_formation_histories.hdf5"
    with h5py.File(fname, "w") as hf:
        hf["galaxies/id_galaxy"] = IDS
        hf["delta_t"] = DELTA_T.astype(np.float32)
        hf["cosmology/h"] = H
        hf["run_info/redshift"] = 0.25
        for i, comp in enumerate(SFR):
            hf[f"{comp}/star_formation_rate_histories"] = SFR[comp].astype(
                np.float32
            )
            hf[f"{comp}/metallicity_histories"] = _zmet(i).astype(np.float32)
    return str(fname)


def _bin_mass(stars, lo, hi):
    """Total mass and mass-weighted mean age of particles aged in [lo, hi)."""
    ages = stars.ages.to("Gyr").value
    mass = stars.initial_masses.to("Msun").value
    sel = (ages >= lo) & (ages < hi)
    return mass[sel].sum(), np.average(ages[sel], weights=mass[sel])


def test_particle_roundtrip(shark_file, test_grid):
    """Masses, ages, metallicities and ids survive the round trip."""
    galaxies = load_SHARK(shark_file, test_grid)
    assert [g.id_galaxy for g in galaxies] == [10, 42, 7]
    assert all(g.redshift == 0.25 for g in galaxies)

    # Galaxy 10: disk bins 0 and 2, merger-bulge bin 1. Each bin is split
    # across the grid ages it contains, conserving mass and mean age
    stars = galaxies[0].stars
    for ibin, sfr in [(0, 1.0), (2, 2.0), (1, 1.0)]:
        mass, mean_age = _bin_mass(stars, *BINS[ibin])
        np.testing.assert_allclose(mass, sfr * 2e9 / H, rtol=1e-6)
        np.testing.assert_allclose(mean_age, BINS[ibin].mean(), rtol=1e-6)
    assert set(np.round(stars.metallicities, 6)) == {0.001, 0.003, 0.004}
    # Only the [4, 6] Gyr bin is merger-bulge star formation
    ages = stars.ages.to("Gyr").value
    in_bin1 = (ages >= 4) & (ages < 6)
    np.testing.assert_array_equal(stars.star_component[in_bin1], 1)
    np.testing.assert_array_equal(stars.star_component[~in_bin1], 0)

    # Zero-SFR bins never become particles
    np.testing.assert_allclose(
        galaxies[1].stars.initial_masses.to("Msun").value.sum(),
        5.0 * 2e9 / H,
        rtol=1e-6,
    )


def test_component_subset(shark_file, test_grid):
    """A components subset only loads those components."""
    galaxies = load_SHARK(
        shark_file, test_grid, components=("bulges_diskins",)
    )
    # Galaxy 10 forms no disk-instability stars
    assert galaxies[0].stars is None
    stars = galaxies[1].stars
    # One bin, [6, 8] Gyr, contains grid ages 10^9.8 and 10^9.9
    assert stars.nstars == 3
    np.testing.assert_allclose(
        stars.initial_masses.to("Msun").value.sum(), 2.0 * 2e9 / H, rtol=1e-6
    )
    np.testing.assert_array_equal(stars.star_component, [0, 0, 0])


def test_parametric(shark_file, test_grid):
    """Parametric method conserves mass and agrees with the particles."""
    particle = load_SHARK(shark_file, test_grid)
    parametric = load_SHARK(shark_file, test_grid, method="parametric")
    ages = 10**test_grid.log10ages
    for pgal, gal in zip(particle, parametric):
        assert isinstance(gal, ParametricGalaxy)
        assert gal.id_galaxy == pgal.id_galaxy
        pmass = pgal.stars.initial_masses.to("Msun").value
        np.testing.assert_allclose(
            np.sum(gal.stars.sfzh), pmass.sum(), rtol=1e-6
        )
        # Same mass-weighted mean age to within the grid spacing
        sfh = gal.stars.sfzh.sum(axis=1)
        np.testing.assert_allclose(
            np.average(ages, weights=sfh),
            np.average(pgal.stars.ages.to("yr").value, weights=pmass),
            rtol=0.05,
        )


def test_bad_arguments(shark_file, test_grid):
    """Unknown method and component raise."""
    with pytest.raises(exceptions.InconsistentArguments):
        load_SHARK(shark_file, test_grid, method="spherical_cow")
    with pytest.raises(exceptions.InconsistentArguments):
        load_SHARK(shark_file, test_grid, components=("disks", "bar"))
