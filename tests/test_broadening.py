"""Tests for spectral broadening helpers and transformers."""

# ruff: noqa: D101, D103, D107

import numpy as np
import pytest
from unyt import Hz, K, amu, angstrom, erg, km, s

from synthesizer import exceptions
from synthesizer.emission_models.transformers import (
    DopplerBroadening,
    ThermalBroadening,
)
from synthesizer.emission_models.utils import get_param
from synthesizer.emissions.sed import Sed


class DummyModel:
    label = "dummy"

    def __init__(self, **fixed_parameters):
        self.fixed_parameters = fixed_parameters


def make_line_sed(shape=()):
    lam = np.linspace(1000.0, 2000.0, 200) * angstrom
    lnu = np.zeros((*shape, lam.size))
    lnu[..., lam.size // 2] = 1.0

    return Sed(lam, lnu * erg / s / Hz)


def test_doppler_broaden_accepts_array_sigma_v():
    sed = make_line_sed(shape=(2,))

    broadened = sed.doppler_broaden(np.array([100.0, 300.0]) * km / s)

    assert broadened.lnu.shape == sed.lnu.shape
    assert broadened.lnu[1].max() < broadened.lnu[0].max()


def test_doppler_broaden_masks_spectra():
    sed = make_line_sed(shape=(2,))

    broadened = sed.doppler_broaden(300.0 * km / s, mask=[True, False])

    assert broadened.lnu[0].max() < sed.lnu[0].max()
    np.testing.assert_allclose(broadened.lnu[1], sed.lnu[1])


def test_doppler_broaden_rejects_bad_sigma_v_shape():
    sed = make_line_sed(shape=(2,))

    with pytest.raises(exceptions.InconsistentArguments):
        sed.doppler_broaden(np.ones((2, 1)) * km / s)


def test_thermally_broaden_accepts_array_temperature():
    sed = make_line_sed(shape=(2,))

    broadened = sed.thermally_broaden(
        np.array([1.0e6, 4.0e6]) * K,
        mu=np.array([1.0, 1.0]) * amu,
    )

    assert broadened.lnu.shape == sed.lnu.shape
    assert broadened.lnu[1].max() < broadened.lnu[0].max()


def test_doppler_broadening_transformer_uses_fixed_parameter():
    sed = make_line_sed()
    model = DummyModel(velocity_dispersion=300.0 * km / s)
    transformer = DopplerBroadening(sigma_v_attr="velocity_dispersion")

    broadened = transformer._transform(sed, None, model, None, None)

    assert broadened.lnu.max() < sed.lnu.max()


def test_get_param_preserve_units_keeps_fixed_parameter_units():
    model = DummyModel(velocity_dispersion=300.0 * km / s)

    default_value = get_param("velocity_dispersion", model, None, None)
    unit_value = get_param(
        "velocity_dispersion",
        model,
        None,
        None,
        preserve_units=True,
    )

    assert not hasattr(default_value, "units")
    assert unit_value.units == km / s


def test_broadening_transformer_preserves_fixed_parameter_units():
    model = DummyModel(velocity_dispersion=300.0 * km / s)
    transformer = DopplerBroadening(
        sigma_v_attr="velocity_dispersion",
        apply_units=False,
    )

    params = transformer._extract_params(
        model,
        None,
        None,
        preserve_units=True,
    )

    assert params["velocity_dispersion"].units == km / s


def test_thermal_broadening_transformer_uses_fixed_parameters():
    sed = make_line_sed()
    model = DummyModel(temperature=1.0e6 * K, mu=1.0 * amu)
    transformer = ThermalBroadening(mu_attr="mu")

    broadened = transformer._transform(sed, None, model, None, None)

    assert broadened.lnu.max() < sed.lnu.max()


def test_broadening_transformer_rejects_lam_mask():
    sed = make_line_sed()
    model = DummyModel(sigma_v=300.0 * km / s)
    transformer = DopplerBroadening()

    with pytest.raises(exceptions.UnimplementedFunctionality):
        transformer._transform(sed, None, model, None, np.ones(sed.lam.size))


def test_broadening_transformer_rejects_non_sed():
    model = DummyModel(sigma_v=300.0 * km / s)
    transformer = DopplerBroadening()

    with pytest.raises(exceptions.InconsistentArguments):
        transformer._transform(object(), None, model, None, None)
