"""Tests for generating spectra."""

import numpy as np
from unyt import Myr


def test_integrated_generation_ngp(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the integrated and per particle machinery
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    per_particle_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra are different
    assert np.allclose(
        integrated_spec._lnu, per_particle_spec._lnu
    ), "The integrated and summed per particle spectra are not the same."


def test_integrated_generation_cic(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the integrated and per particle machinery
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    per_particle_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra are different
    assert np.allclose(
        integrated_spec._lnu, per_particle_spec._lnu
    ), "The integrated and summed per particle spectra are not the same."


def test_threaded_generation_ngp_per_particle(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(True)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_ngp_integrated(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(False)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_cic_per_particle(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(True)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_cic_integrated(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(False)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_reusing_weights_ngp(nebular_emission_model, random_part_stars):
    """Test reusing weights to calculate another spectra for the same grid."""

    # Compute the spectra the first time
    first_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure we have the weights
    assert hasattr(
        random_part_stars, "_grid_weights"
    ), "The grid weights are not stored."
    assert (
        "test_grid" in random_part_stars._grid_weights["ngp"]
    ), "The grid weights are not stored."

    # Compute the spectra the second time which will reuse the weights
    random_part_stars.clear_all_emissions()
    second_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        first_spec._lnu,
        second_spec._lnu,
    ), "The first and second spectra are not the same."


def test_reusing_weights_cic(nebular_emission_model, random_part_stars):
    """Test reusing weights to calculate another spectra for the same grid."""

    # Compute the spectra the first time
    first_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    # Ensure we have the weights
    assert hasattr(
        random_part_stars, "_grid_weights"
    ), "The grid weights are not stored."
    assert (
        "test_grid" in random_part_stars._grid_weights["cic"]
    ), "The grid weights are not stored."

    # Compute the spectra the second time which will reuse the weights
    random_part_stars.clear_all_emissions()
    second_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        first_spec._lnu,
        second_spec._lnu,
    ), "The first and second spectra are not the same."


def test_masked_spectra(particle_stars_B, reprocessed_emission_model):
    """Test the effect of masking during the generation of spectra."""
    # Generate spectra without masking
    unmasked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )
    particle_stars_B.clear_all_emissions()

    # Generate spectra with masking
    reprocessed_emission_model.add_mask(
        attr="ages",
        op=">",
        thresh=5.5 * Myr,
        set_all=True,
    )
    mask = particle_stars_B.get_mask(
        attr="ages",
        thresh=5.5 * Myr,
        op=">",
    )

    assert np.any(mask), "The mask is empty."

    masked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the masked spectra are different
    assert not np.allclose(
        unmasked_spec._lnu,
        masked_spec._lnu,
    ), "The masked and unmasked spectra are the same."

    # Ensure the masked spectra are less than the unmasked spectra
    assert np.sum(masked_spec._lnu) < np.sum(
        unmasked_spec._lnu
    ), "The masked spectra is not less than the unmasked spectra."
