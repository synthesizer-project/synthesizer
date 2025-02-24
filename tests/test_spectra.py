"""Tests for generating spectra."""

import numpy as np


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
