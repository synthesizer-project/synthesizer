"""A suite of tests for the emission model operations."""

import numpy as np

from synthesizer.emission_models import StellarEmissionModel
from synthesizer.emission_models.transformers import PowerLaw


def test_single_star_extraction(
    single_star_particle,
    single_star_parametric,
    test_grid,
    incident_emission_model,
    transmitted_emission_model,
    nebular_emission_model,
    reprocessed_emission_model,
):
    """Test extraciton of a single star's emission.

    This will use and compare a single star for a particle Stars object and a
    single SFZH bin for a parametric Stars object. These two descriptions
    should be equivalent.
    """
    # First ensure the sfzh's are equivalent
    single_star_particle.get_sfzh(
        test_grid.log10ages,
        test_grid.log10metallicities,
    )
    assert np.isclose(np.sum(single_star_particle.sfzh.sfzh), 1.0), (
        "The unit particle SFZH does not sum to 1"
        f" (sum={np.sum(single_star_particle.sfzh.sfzh.sum())})"
    )
    assert np.isclose(np.sum(single_star_parametric.sfzh), 1.0), (
        "The unit parametric SFZH does not sum to 1"
        f" (sum={np.sum(single_star_parametric.sfzh)})"
    )
    assert np.allclose(
        single_star_particle.sfzh.sfzh,
        single_star_parametric.sfzh,
    ), (
        f"The SFZH's are not equivalent (non-zero elements: "
        f"particle={np.where(single_star_particle.sfzh.sfzh > 0)}, "
        f"parametric={np.where(single_star_parametric.sfzh > 0)})"
    )

    # Ok, we know the SFZH's are equivalent, let's now get the spectra
    # and compare them for a range of emission model complexities.

    # Loop over the emission models
    for model in [
        incident_emission_model,
        transmitted_emission_model,
        nebular_emission_model,
        reprocessed_emission_model,
    ]:
        # Loop over grid look up methods too
        for method in ["ngp", "cic"]:
            part_sed = single_star_particle.get_spectra(
                model,
                grid_assignment_method=method,
            )
            param_sed = single_star_parametric.get_spectra(
                model,
                grid_assignment_method=method,
            )
            assert np.allclose(part_sed.shape, param_sed.shape), (
                f"[{model.__class__.__name__}] (with {method}): "
                "The SED shapes are not equivalent "
                f"(particle={part_sed.shape}, "
                f"parametric={param_sed.shape})"
            )
            resi = np.sum(part_sed.lnu) - np.sum(param_sed.lnu)
            assert np.allclose(
                part_sed.lnu,
                param_sed.lnu,
            ), (
                f"[{model.__class__.__name__}] (with {method}): "
                "The SEDs are not equivalent (part_sed.sum - param_sed.sum = "
                f"{np.sum(part_sed.lnu)} - {np.sum(param_sed.lnu)} = {resi}, "
            )


def test_attenuation_transform(unit_sed):
    """Test attenuating an sed."""
    # Get the attenuation law
    dcurve = PowerLaw(slope=0.0)

    att_unit_sed = unit_sed.apply_attenuation(
        tau_v=0.1,
        dust_curve=dcurve,
    )

    # Ensure the shape is the same
    assert np.allclose(
        unit_sed.lnu.shape,
        att_unit_sed.lnu.shape,
    ), (
        "The attenuated SED shape is not the same as the original"
        f" (original={unit_sed.lnu.shape},"
        f" attenuated={att_unit_sed.lnu.shape})"
    )

    # Ensure the attenuation is correct
    assert np.allclose(
        att_unit_sed.lnu,
        unit_sed.lnu * np.exp(-0.1),
    ), (
        "The attenuated SED is not correct"
        f" (original={unit_sed.lnu}, attenuated={att_unit_sed.lnu})"
    )


def test_combination_spectra(
    random_part_stars,
    test_grid,
    incident_emission_model,
    transmitted_emission_model,
):
    """Test the combination of spectra."""
    # Create an emission model that will combine the incident and transmitted
    # emission models
    model = StellarEmissionModel(
        label="combined",
        combine=(incident_emission_model, transmitted_emission_model),
    )

    # Get the spectra
    combined_spec = random_part_stars.get_spectra(model)

    # Explicitly add the spectra together
    explicit_spectra = (
        random_part_stars.spectra["incident"].lnu
        + random_part_stars.spectra["transmitted"].lnu
    )

    # Ensure the shapes are the same
    assert np.allclose(
        combined_spec.lnu.shape,
        explicit_spectra.shape,
    ), (
        "The combined spectra shape is not the same as the explicit sum"
        f" (combined={combined_spec.lnu.shape}, "
        f"explicit={explicit_spectra.shape})"
    )

    # Ensure the spectra are the same
    assert np.allclose(
        combined_spec.lnu,
        explicit_spectra,
    ), (
        "The combined spectra are not the same as the explicit sum"
        f" (combined={combined_spec.lnu}, explicit={explicit_spectra})"
    )
