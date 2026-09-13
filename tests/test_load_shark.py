"""Tests for the SHARK load_data submodule."""

import h5py
import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.load_data.load_shark import load_SHARK
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy

H = 0.7
DELTA_T = np.array([2.0, 2.0, 2.0, 2.0])  # Gyr
# Lookback time from z=0 to each bin midpoint; the output snapshot sits
# 3 Gyr back, so ages at the output are [7, 5, 3, 1] Gyr
LBT_MEAN = np.array([10.0, 8.0, 6.0, 4.0])  # Gyr
AGES = np.array([7.0, 5.0, 3.0, 1.0])  # Gyr
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
        hf["lbt_mean"] = LBT_MEAN.astype(np.float32)
        hf["delta_t"] = DELTA_T.astype(np.float32)
        hf["cosmology/h"] = H
        hf["run_info/redshift"] = 0.25
        for i, comp in enumerate(SFR):
            hf[f"{comp}/star_formation_rate_histories"] = SFR[comp].astype(
                np.float32
            )
            hf[f"{comp}/metallicity_histories"] = _zmet(i).astype(np.float32)
    return str(fname)


def test_particle_roundtrip(shark_file):
    """Masses, ages, metallicities and ids survive the round trip."""
    galaxies = load_SHARK(shark_file)
    assert [g.id_galaxy for g in galaxies] == [10, 42, 7]
    assert all(g.redshift == 0.25 for g in galaxies)

    # Galaxy 10: disk bins 0 and 2, then merger-bulge bin 1
    stars = galaxies[0].stars
    assert stars.nstars == 3
    np.testing.assert_allclose(
        stars.initial_masses.to("Msun").value,
        np.array([1.0, 2.0, 1.0]) * 2e9 / H,
    )
    np.testing.assert_allclose(
        stars.ages.to("yr").value, AGES[[0, 2, 1]] * 1e9, rtol=1e-6
    )
    np.testing.assert_allclose(
        stars.metallicities, [0.001, 0.003, 0.004], rtol=1e-6
    )
    np.testing.assert_array_equal(stars.star_component, [0, 0, 1])

    # Zero-SFR bins never become particles
    assert galaxies[1].stars.nstars == 2
    assert galaxies[2].stars.nstars == 3


def test_component_subset(shark_file):
    """A components subset only loads those components."""
    galaxies = load_SHARK(shark_file, components=("bulges_diskins",))
    # Galaxy 10 forms no disk-instability stars
    assert galaxies[0].stars is None
    stars = galaxies[1].stars
    assert stars.nstars == 1
    np.testing.assert_allclose(
        stars.initial_masses.to("Msun").value, [2.0 * 2e9 / H]
    )
    np.testing.assert_array_equal(stars.star_component, [0])


def test_parametric(shark_file, test_grid):
    """Parametric method conserves mass on the grid."""
    particle = load_SHARK(shark_file)
    parametric = load_SHARK(shark_file, method="parametric", grid=test_grid)
    for pgal, gal in zip(particle, parametric):
        assert isinstance(gal, ParametricGalaxy)
        assert gal.id_galaxy == pgal.id_galaxy
        np.testing.assert_allclose(
            np.sum(gal.stars.sfzh),
            np.sum(pgal.stars.initial_masses.to("Msun").value),
            rtol=1e-6,
        )


def test_bad_arguments(shark_file):
    """Unknown method/component and missing grid raise."""
    with pytest.raises(exceptions.InconsistentArguments):
        load_SHARK(shark_file, method="parametric")
    with pytest.raises(exceptions.InconsistentArguments):
        load_SHARK(shark_file, method="spherical_cow")
    with pytest.raises(exceptions.InconsistentArguments):
        load_SHARK(shark_file, components=("disks", "bar"))
