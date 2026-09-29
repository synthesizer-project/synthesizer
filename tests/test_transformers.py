"""A test suite for the transformers module."""

import numpy as np
from unyt import K, angstrom

from synthesizer.emission_models import (
    AttenuatedEmission,
    DustEmission,
    IntrinsicEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TransmittedEmission,
)
from synthesizer.emission_models.attenuation import Calzetti2000, PowerLaw
from synthesizer.emission_models.generators.dust.greybody import Greybody


def test_fesc_model_level(
    test_grid,
    random_part_stars,
):
    """Test setting fesc on the emission model.

    This test ensures that the fesc set during instantiation of a model
    is set correctly. We only do this for the IntrinsicEmission model here
    since it encompasses all the models that relate to fesc with some parent
    models derived from it.
    """
    # Get the spectra with fesc=0
    transmitted0 = TransmittedEmission(
        test_grid,
        label="transmitted",
        fesc=0.0,
    )
    model = ReprocessedEmission(
        test_grid,
        label="reprocessed",
        transmitted=transmitted0,
    )
    spec_fesc0 = random_part_stars.get_spectra(model)
    random_part_stars.clear_all_emissions()

    # Get the spectra with fesc=1
    transmitted1 = TransmittedEmission(
        test_grid,
        label="transmitted",
        fesc=1.0,
    )
    reprocessed1 = ReprocessedEmission(
        test_grid,
        label="reprocessed",
        transmitted=transmitted1,
    )
    model = IntrinsicEmission(
        test_grid,
        label="intrinsic",
        reprocessed=reprocessed1,
        escaped=transmitted1,
    )
    spec_fesc1 = random_part_stars.get_spectra(model)

    # Ensure the two differ
    assert not np.all(spec_fesc0.lnu == spec_fesc1.lnu), (
        "The spectra are the same with different fesc values"
    )


def test_fesc_override(
    test_grid,
    random_part_stars,
    nebular_emission_model,
    reprocessed_emission_model,
    intrinsic_emission_model,
    pacman_emission_model,
):
    """Test the get_spectra fesc override in the emission model.

    This test ensures that the fesc override on get_spectra methods is working
    correctly. Unlike above, we test all the models here to ensure the
    override is correctly passed through different levels of complexity.
    """
    # For this test we need a singular tau_V value (we're only doing integrated
    # spectra)
    random_part_stars.tau_v = 0.1

    # Loop over the models and test
    for model in [
        reprocessed_emission_model,
        intrinsic_emission_model,
        pacman_emission_model,
    ]:
        # Get the spectra with fesc=0
        spec_fesc0 = random_part_stars.get_spectra(
            model,
            fesc=0,
        )
        random_part_stars.clear_all_emissions()

        # Get the spectra with fesc=1
        spec_fesc1 = random_part_stars.get_spectra(
            model,
            fesc=1,
        )

        # Ensure the two differ
        assert not np.all(spec_fesc0.lnu == spec_fesc1.lnu), (
            f"[{model.__class__.__name__}]: The spectra "
            "are the same with different fesc values"
        )


def _attenuated_lnu(stars, test_grid, dust_curve, **model_params):
    """Return the attenuated spectrum for a dust curve and model parameters.

    Args:
        stars (Stars): The emitter to generate the spectra with.
        test_grid (Grid): The grid to extract the incident emission from.
        dust_curve (AttenuationLaw): The dust curve to attenuate with.
        **model_params (dict): Parameters to set on the EmissionModel itself.

    Returns:
        unyt_array: The attenuated luminosity.
    """
    incident = StellarEmissionModel(
        label="incident",
        grid=test_grid,
        extract="incident",
    )
    attenuated = AttenuatedEmission(
        label="attenuated",
        dust_curve=dust_curve,
        apply_to=incident,
        tau_v=0.5,
        emitter="stellar",
        **model_params,
    )
    stars.clear_all_emissions()
    return stars.get_spectra(attenuated).lnu


def test_transformer_param_on_model_overrides_curve(
    test_grid,
    random_part_stars,
):
    """Test a parameter set on the model overrides the dust curve value."""
    # The curve value should be used when the model says nothing about it
    default = _attenuated_lnu(
        random_part_stars,
        test_grid,
        PowerLaw(slope=-1.0),
    )
    explicit = _attenuated_lnu(
        random_part_stars,
        test_grid,
        PowerLaw(slope=-0.5),
    )
    assert not np.allclose(default.value, explicit.value), (
        "Changing the curve slope should change the attenuation"
    )

    # And a value set on the model should win over the curve value
    overridden = _attenuated_lnu(
        random_part_stars,
        test_grid,
        PowerLaw(slope=-1.0),
        slope=-0.5,
    )
    assert np.allclose(overridden.value, explicit.value), (
        "A slope set on the EmissionModel should override the curve value"
    )


def test_transformer_unit_param_on_model_overrides_curve(
    test_grid,
    random_part_stars,
):
    """Test unit carrying parameters can also be overridden on the model."""
    # Calzetti2000 declares slope, cent_lam, ampl and gamma alongside tau_v,
    # with cent_lam and gamma carrying units
    default = _attenuated_lnu(
        random_part_stars,
        test_grid,
        Calzetti2000(ampl=1.0, cent_lam=2175 * angstrom),
    )
    explicit = _attenuated_lnu(
        random_part_stars,
        test_grid,
        Calzetti2000(ampl=1.0, cent_lam=2500 * angstrom),
    )
    assert not np.allclose(default.value, explicit.value), (
        "Changing the curve cent_lam should change the attenuation"
    )

    overridden = _attenuated_lnu(
        random_part_stars,
        test_grid,
        Calzetti2000(ampl=1.0, cent_lam=2175 * angstrom),
        cent_lam=2500 * angstrom,
    )
    assert np.allclose(overridden.value, explicit.value), (
        "A cent_lam set on the EmissionModel should override the curve value"
    )


def test_transformer_string_param_is_still_an_alias(
    test_grid,
    random_part_stars,
):
    """Test a string parameter is looked up under the aliased name."""
    # A string on the curve means "look this up under this name", so setting
    # the aliased name on the model should behave like setting slope directly
    explicit = _attenuated_lnu(
        random_part_stars,
        test_grid,
        PowerLaw(slope=-0.5),
    )
    aliased = _attenuated_lnu(
        random_part_stars,
        test_grid,
        PowerLaw(slope="slope_young"),
        slope_young=-0.5,
    )
    assert np.allclose(aliased.value, explicit.value), (
        "An aliased slope should be looked up under the aliased name"
    )


def test_generator_param_on_model_overrides_generator(
    test_grid,
    random_part_stars,
):
    """Test a parameter on the model overrides a generator value.

    This is the mirror of the transformer tests above, locking both sides of
    the EmissionModel to the same rule.
    """
    incident = StellarEmissionModel(
        label="incident",
        grid=test_grid,
        extract="incident",
    )
    attenuated = AttenuatedEmission(
        label="attenuated",
        dust_curve=PowerLaw(slope=-1.0),
        apply_to=incident,
        tau_v=0.5,
        emitter="stellar",
    )

    def dust_lnu(curve_temperature, **model_params):
        dust = DustEmission(
            dust_emission_model=Greybody(
                temperature=curve_temperature,
                emissivity=1.5,
            ),
            emitter="stellar",
            label="dust_emission",
            dust_lum_intrinsic=incident,
            dust_lum_attenuated=attenuated,
            **model_params,
        )
        random_part_stars.clear_all_emissions()
        return random_part_stars.get_spectra(dust).lnu

    default = dust_lnu(30 * K)
    explicit = dust_lnu(60 * K)
    assert not np.allclose(default.value, explicit.value), (
        "Changing the greybody temperature should change the emission"
    )

    overridden = dust_lnu(30 * K, temperature=60 * K)
    assert np.allclose(overridden.value, explicit.value), (
        "A temperature set on the EmissionModel should override the "
        "generator value"
    )
