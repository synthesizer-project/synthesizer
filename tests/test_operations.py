"""A suite of tests for the emission model operations."""

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.emission_models import (
    AttenuatedEmission,
    StellarEmissionModel,
)
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
    """Test extraction of a single star's emission.

    This will use and compare a single star for a particle Stars object and a
    single SFZH bin for a parametric Stars object. These two descriptions
    should be equivalent.
    """
    # First ensure the sfzh's are equivalent
    single_star_particle.get_sfzh(
        test_grid.log10ages,
        test_grid.metallicities,
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
        combine=(
            transmitted_emission_model["incident"],
            transmitted_emission_model,
        ),
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


def test_transformation_with_string_label(stars_with_fake_spectra):
    """Test transformation operations using string labels for apply_to."""
    stars = stars_with_fake_spectra

    # Test applying transformation using string label
    att_model = AttenuatedEmission(
        label="attenuated_string",
        dust_curve=PowerLaw(slope=-1.0),
        apply_to="intrinsic",  # String label instead of model
        tau_v=0.5,
        emitter="stellar",
    )

    # Get the attenuated spectra using string label
    att_spec = stars.get_spectra(att_model)

    # Ensure the string-based transformation is stored correctly
    assert "attenuated_string" in stars.spectra
    assert att_spec is not None
    # Verify the result is properly attenuated (should be less than original)
    assert np.all(att_spec.lnu <= stars.spectra["intrinsic"].lnu)


def test_invalid_string_labels(stars_with_fake_spectra):
    """Test error handling for invalid string labels."""
    stars = stars_with_fake_spectra

    # Test transformation with non-existent label
    with pytest.raises(exceptions.InconsistentArguments):
        invalid_model = AttenuatedEmission(
            label="invalid_transform",
            dust_curve=PowerLaw(slope=-1.0),
            apply_to="nonexistent_spectrum",  # Invalid label
            tau_v=0.5,
            emitter="stellar",
        )
        stars.get_spectra(invalid_model)


def test_unsaved_shared_dependency_survives_until_last_consumer(
    random_part_stars,
    test_grid,
):
    """Test unsaved shared dependencies live until all consumers run."""
    # Measure the reference incident spectrum before building the shared DAG.
    incident_reference = StellarEmissionModel(
        label="incident_reference",
        grid=test_grid,
        extract="incident",
    )
    reference_spectra = random_part_stars.get_spectra(incident_reference)
    random_part_stars.clear_all_emissions()

    # Build an unsaved extraction that is consumed both directly and through a
    # downstream transformation.
    incident = StellarEmissionModel(
        label="incident_unsaved",
        grid=test_grid,
        extract="incident",
        save=False,
    )
    attenuated = AttenuatedEmission(
        label="attenuated_unsaved",
        dust_curve=PowerLaw(slope=0.0),
        apply_to=incident,
        tau_v=0.1,
        save=False,
        emitter="stellar",
    )
    total = StellarEmissionModel(
        label="total_saved",
        combine=(incident, attenuated),
    )

    # Generate the spectra and keep the root result for a direct check.
    total_spectra = random_part_stars.get_spectra(total)

    # The unsaved intermediate models should have been deleted eagerly.
    assert "incident_unsaved" not in random_part_stars.spectra
    assert "attenuated_unsaved" not in random_part_stars.spectra
    assert "total_saved" in random_part_stars.spectra

    # The shared dependency must still have survived long enough to build the
    # correct final combined spectrum.
    expected_scaling = 1.0 + np.exp(-0.1)
    assert np.allclose(
        total_spectra.lnu,
        reference_spectra.lnu * expected_scaling,
    )


def test_related_models_are_executed_in_same_queue(
    random_part_stars,
    test_grid,
):
    """Test related models are executed as part of the same queue."""
    # Build a saved extraction and a related attenuation model that depends on
    # the same extracted spectrum.
    incident = StellarEmissionModel(
        label="incident_root",
        grid=test_grid,
        extract="incident",
    )
    attenuated = AttenuatedEmission(
        label="incident_related",
        dust_curve=PowerLaw(slope=0.0),
        apply_to=incident,
        tau_v=0.2,
        emitter="stellar",
    )
    incident.related_models.add(attenuated)

    # Generate the root model and ensure the related model is also produced.
    incident_spectra = random_part_stars.get_spectra(incident)

    assert "incident_root" in random_part_stars.spectra
    assert "incident_related" in random_part_stars.spectra
    assert np.allclose(
        incident_spectra.lnu, random_part_stars.spectra["incident_root"].lnu
    )
    assert np.all(
        random_part_stars.spectra["incident_related"].lnu
        <= random_part_stars.spectra["incident_root"].lnu
    )


def test_nested_related_models_are_executed_in_same_queue(
    random_part_stars,
    test_grid,
):
    """Test nested related models are collected recursively."""
    # Build a root extraction and two related attenuations chained together.
    incident = StellarEmissionModel(
        label="nested_incident_root",
        grid=test_grid,
        extract="incident",
    )
    attenuated_once = AttenuatedEmission(
        label="nested_incident_related_1",
        dust_curve=PowerLaw(slope=0.0),
        apply_to=incident,
        tau_v=0.2,
        emitter="stellar",
    )
    attenuated_twice = AttenuatedEmission(
        label="nested_incident_related_2",
        dust_curve=PowerLaw(slope=0.0),
        apply_to=attenuated_once,
        tau_v=0.3,
        emitter="stellar",
    )
    incident.related_models.add(attenuated_once)
    attenuated_once.related_models.add(attenuated_twice)

    # Generate the root model and ensure both related models are also produced.
    random_part_stars.get_spectra(incident)

    assert "nested_incident_root" in random_part_stars.spectra
    assert "nested_incident_related_1" in random_part_stars.spectra
    assert "nested_incident_related_2" in random_part_stars.spectra
