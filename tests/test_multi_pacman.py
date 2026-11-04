"""Tests for the MultiPacman emission models."""

import numpy as np
import pytest
from unyt import K, Msun, Myr, km, s

from synthesizer.emission_models import (
    BimodalMultiPacmanEmission,
    BimodalPacmanEmission,
    MultiPacmanEmission,
    PacmanEmission,
)
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.particle import Stars


@pytest.fixture
def stars():
    """Return particle stars spanning both sides of the age pivot."""
    nstars = 20
    return Stars(
        initial_masses=np.full(nstars, 1e6) * Msun,
        ages=np.logspace(0, 3, nstars) * Myr,
        metallicities=np.full(nstars, 0.01),
    )


def _lnu(stars, model, label):
    """Get the integrated spectrum for one label of a model."""
    stars.clear_all_emissions()
    stars.get_spectra(model, out_dtype=np.float64)
    return stars.spectra[label].lnu.sum(axis=0).value


def _greybody():
    return Greybody(temperature=30 * K, emissivity=1.6)


@pytest.mark.parametrize("with_dust_emission", (False, True))
def test_reduces_to_pacman(test_grid, stars, with_dust_emission):
    """Without attenuating escaped emission or fesc_dust it is Pacman."""
    common = dict(
        tau_v=0.5, dust_curve=Calzetti2000(), fesc=0.2, fesc_ly_alpha=0.5
    )
    multi = MultiPacmanEmission(
        test_grid,
        fesc_dust=0.0,
        attenuate_incident_escaped=False,
        dust_emission=_greybody() if with_dust_emission else None,
        **common,
    )
    pacman = PacmanEmission(
        test_grid,
        dust_emission=_greybody() if with_dust_emission else None,
        **common,
    )
    label = "total" if with_dust_emission else "emergent"
    assert multi.label == label
    np.testing.assert_allclose(
        _lnu(stars, multi, label), _lnu(stars, pacman, label), rtol=1e-10
    )


@pytest.mark.parametrize("attenuate_incident_escaped", (False, True))
def test_dust_escape_split(test_grid, stars, attenuate_incident_escaped):
    """fesc_dust splits what reaches the dust into escaped and obscured."""
    model = MultiPacmanEmission(
        test_grid,
        tau_v=0.5,
        fesc=0.2,
        fesc_dust=0.3,
        fesc_ly_alpha=0.5,
        attenuate_incident_escaped=attenuate_incident_escaped,
    )
    stars.get_spectra(model, out_dtype=np.float64)
    spectra = {k: v.lnu.sum(axis=0).value for k, v in stars.spectra.items()}

    reaching = spectra["reprocessed"] + (
        spectra["escaped"] if attenuate_incident_escaped else 0.0
    )
    np.testing.assert_allclose(spectra["dust_escaped"], 0.3 * reaching)
    np.testing.assert_allclose(spectra["obscured"], 0.7 * reaching)

    leaving = spectra["attenuated"] + spectra["dust_escaped"]
    if not attenuate_incident_escaped:
        leaving = leaving + spectra["escaped"]
    np.testing.assert_allclose(spectra["emergent"], leaving)


def test_attenuating_escaped_reduces_emergent(test_grid, stars):
    """Attenuating the escaped incident emission removes light."""
    kwargs = dict(tau_v=0.5, fesc=0.2, fesc_dust=0.0, fesc_ly_alpha=0.5)
    with_dust = _lnu(
        stars,
        MultiPacmanEmission(test_grid, **kwargs),
        "emergent",
    )
    without_dust = _lnu(
        stars,
        MultiPacmanEmission(
            test_grid, attenuate_incident_escaped=False, **kwargs
        ),
        "emergent",
    )
    assert with_dust.sum() < without_dust.sum()


def test_bimodal_reduces_to_pacman(test_grid, stars):
    """The bimodal model also reduces to its Pacman equivalent."""
    common = dict(
        tau_v_ism=0.3,
        tau_v_birth=0.6,
        fesc=0.2,
        fesc_ly_alpha=0.5,
    )
    multi = BimodalMultiPacmanEmission(
        test_grid,
        fesc_dust=0.0,
        attenuate_incident_escaped=False,
        dust_emission_ism=_greybody(),
        dust_emission_birth=_greybody(),
        **common,
    )
    pacman = BimodalPacmanEmission(
        test_grid,
        dust_emission_ism=_greybody(),
        dust_emission_birth=_greybody(),
        **common,
    )
    for label in ("young_emergent", "old_emergent", "emergent", "total"):
        np.testing.assert_allclose(
            _lnu(stars, multi, label), _lnu(stars, pacman, label), rtol=1e-10
        )


@pytest.mark.parametrize("attenuate_incident_escaped", (False, True))
def test_bimodal_young_reaches_ism(
    test_grid, stars, attenuate_incident_escaped
):
    """Young emission reaching the ISM is birth attenuated (+ escaped)."""
    model = BimodalMultiPacmanEmission(
        test_grid,
        tau_v_ism=0.3,
        tau_v_birth=0.6,
        fesc=0.2,
        fesc_dust=0.3,
        fesc_ly_alpha=0.5,
        attenuate_incident_escaped=attenuate_incident_escaped,
    )
    assert model.label == "emergent"
    stars.get_spectra(model, out_dtype=np.float64)
    spectra = {k: v.lnu.sum(axis=0).value for k, v in stars.spectra.items()}

    reaching = spectra["young_attenuated_nebular"] + (
        spectra["young_escaped"] if attenuate_incident_escaped else 0.0
    )
    np.testing.assert_allclose(spectra["young_obscured"], 0.7 * reaching)
    np.testing.assert_allclose(
        spectra["emergent"],
        spectra["young_emergent"] + spectra["old_emergent"],
    )


@pytest.mark.parametrize(
    "multi_cls, pacman_cls, kwargs",
    (
        (MultiPacmanEmission, PacmanEmission, dict(tau_v=0.5)),
        (
            BimodalMultiPacmanEmission,
            BimodalPacmanEmission,
            dict(tau_v_ism=0.3, tau_v_birth=0.6),
        ),
    ),
)
def test_velocity_dispersion(test_grid, stars, multi_cls, pacman_cls, kwargs):
    """Broadening matches Pacman and only the final emission is saved."""
    common = dict(
        fesc=0.2, fesc_ly_alpha=0.5, velocity_dispersion=100 * km / s, **kwargs
    )
    multi = multi_cls(
        test_grid, fesc_dust=0.0, attenuate_incident_escaped=False, **common
    )
    pacman = pacman_cls(test_grid, **common)

    assert multi.label == "emergent"
    assert multi.transformer.__class__.__name__ == "DopplerBroadening"
    assert not multi["emergent_predispersion"].save
    np.testing.assert_allclose(
        _lnu(stars, multi, "emergent"),
        _lnu(stars, pacman, "emergent"),
        rtol=1e-10,
    )


def test_dust_generators_not_modified(test_grid):
    """The caller's dust generators are copied, not given energy balance."""
    single, birth, ism = _greybody(), _greybody(), _greybody()
    MultiPacmanEmission(test_grid, tau_v=0.5, dust_emission=single)
    BimodalMultiPacmanEmission(
        test_grid,
        tau_v_ism=0.3,
        tau_v_birth=0.6,
        dust_emission_ism=ism,
        dust_emission_birth=birth,
    )
    assert not any(g.is_energy_balance for g in (single, birth, ism))


def test_bimodal_requires_tau_v_birth(test_grid):
    """The bimodal model needs a birth cloud."""
    with pytest.raises(ValueError):
        BimodalMultiPacmanEmission(test_grid, tau_v_ism=0.3, tau_v_birth=None)
