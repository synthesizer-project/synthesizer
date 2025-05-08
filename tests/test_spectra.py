"""Tests for generating spectra."""

import numpy as np
from unyt import Myr, dimensionless, yr


def test_integrated_generation_ngp(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the per particle machinery
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    integrated_spec = random_part_stars.spectra["nebular"]
    numpy_integrated_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra and per particle spectra are the same
    assert np.allclose(integrated_spec._lnu, numpy_integrated_spec._lnu), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )

    # Check the explicit integrated spectra agrees with the integrated spectra
    # calculated at the same time as the per particle spectra
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    assert np.allclose(
        integrated_spec._lnu,
        numpy_integrated_spec._lnu,
    ), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )


def test_masked_integrated_generation_ngp(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of integrated with a mask spectra."""
    # Add a mask to the emission model
    nebular_emission_model.add_mask(
        attr="ages",
        op=">=",
        thresh=np.median(random_part_stars.ages),
        set_all=True,
    )

    # Compute the spectra using the per particle machinery
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    integrated_spec = random_part_stars.spectra["nebular"]
    numpy_integrated_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra and per particle spectra are the same
    assert np.allclose(integrated_spec._lnu, numpy_integrated_spec._lnu), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )

    # Check the explicit integrated spectra agrees with the integrated spectra
    # calculated at the same time as the per particle spectra
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    assert np.allclose(
        integrated_spec._lnu,
        numpy_integrated_spec._lnu,
    ), (
        "The (truly) integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )


def test_integrated_generation_cic(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the per particle machinery
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    integrated_spec = random_part_stars.spectra["nebular"]
    numpy_integrated_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra and per particle spectra are the same
    assert np.allclose(integrated_spec._lnu, numpy_integrated_spec._lnu), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )

    # Check the explicit integrated spectra agrees with the integrated spectra
    # calculated at the same time as the per particle spectra
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    assert np.allclose(
        integrated_spec._lnu,
        numpy_integrated_spec._lnu,
    ), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )


def test_masked_integrated_generation_cic(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of integrated with a mask spectra."""
    # Add a mask to the emission model
    nebular_emission_model.add_mask(
        attr="ages",
        op=">=",
        thresh=np.median(random_part_stars.ages),
        set_all=True,
    )

    # Check the explicit integrated spectra agrees with the integrated spectra
    # calculated at the same time as the per particle spectra
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    # Compute the spectra using both the per particle machinery
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    _integrated_spec = random_part_stars.spectra["nebular"]
    numpy_integrated_spec = per_particle_spec.sum()

    assert np.allclose(
        integrated_spec._lnu,
        numpy_integrated_spec._lnu,
    ), (
        "The (truly) integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )

    # Ensure that the integrated spectra and per particle spectra are the same
    assert np.allclose(_integrated_spec._lnu, numpy_integrated_spec._lnu), (
        "The integrated and numpy summed per particle spectra are not"
        f" the same (integrated={np.sum(_integrated_spec._lnu)} vs "
        f"explicit={np.sum(numpy_integrated_spec._lnu)})."
    )


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
    assert np.allclose(serial_spec._lnu, threaded_spec._lnu), (
        "The serial and threaded spectra are not the same."
    )


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
    assert np.allclose(serial_spec._lnu, threaded_spec._lnu), (
        "The serial and threaded spectra are not the same."
    )


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
    assert np.allclose(serial_spec._lnu, threaded_spec._lnu), (
        "The serial and threaded spectra are not the same."
    )


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
    assert np.allclose(serial_spec._lnu, threaded_spec._lnu), (
        "The serial and threaded spectra are not the same."
    )


def test_reusing_weights_ngp(nebular_emission_model, random_part_stars):
    """Test reusing weights to calculate another spectra for the same grid."""
    # Compute the spectra the first time
    first_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure we have the weights
    assert hasattr(random_part_stars, "_grid_weights"), (
        "The grid weights are not stored."
    )
    assert "test_grid" in random_part_stars._grid_weights["ngp"], (
        "The grid weights are not stored."
    )

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
    assert hasattr(random_part_stars, "_grid_weights"), (
        "The grid weights are not stored."
    )
    assert "test_grid" in random_part_stars._grid_weights["cic"], (
        "The grid weights are not stored."
    )

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


def test_masked_int_spectra(particle_stars_B, reprocessed_emission_model):
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

    assert np.any(mask), f"The mask is empty (defined in yr): {mask}"
    assert False in mask, f"The mask is all True: {mask}"

    masked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the masked spectra are different
    assert not np.allclose(
        unmasked_spec._lnu,
        masked_spec._lnu,
    ), "The masked and unmasked integrated spectra are the same."

    # Ensure the masked spectra are less than the unmasked spectra
    assert np.sum(masked_spec._lnu) < np.sum(unmasked_spec._lnu), (
        "The masked integrated spectra is not less than "
        "the unmasked integrated spectra."
    )


def test_masked_int_spectra_diff_units(
    particle_stars_B,
    reprocessed_emission_model,
):
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
        thresh=10**6 * 5.5 * yr,
        set_all=True,
    )
    mask = particle_stars_B.get_mask(
        attr="ages",
        thresh=10**6 * 5.5 * yr,
        op=">",
    )

    assert np.any(mask), f"The mask is empty (defined in yr): {mask}"
    assert False in mask, f"The mask is all True: {mask}"

    masked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the masked spectra are different
    assert not np.allclose(
        unmasked_spec._lnu,
        masked_spec._lnu,
    ), "The masked and unmasked integrated spectra are the same."

    # Ensure the masked spectra are less than the unmasked spectra
    assert np.sum(masked_spec._lnu) < np.sum(unmasked_spec._lnu), (
        "The masked integrated spectra is not less than "
        "the unmasked integrated spectra."
    )

    # Clear the masks define a new mask in different units
    reprocessed_emission_model.clear_masks()
    reprocessed_emission_model.add_mask(
        attr="ages",
        op=">",
        thresh=5.5 * Myr,
        set_all=True,
    )
    new_mask = particle_stars_B.get_mask(
        attr="ages",
        thresh=5.5 * Myr,
        op=">",
    )

    assert np.any(new_mask), f"The mask is empty (defined in yr): {new_mask}"
    assert False in new_mask, f"The mask is all True: {new_mask}"
    assert np.all(mask == new_mask), (
        "The masks with different units are not the same."
    )

    dif_units_masked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure the masks with diffferent units have produced the same result
    assert np.allclose(
        masked_spec._lnu,
        dif_units_masked_spec._lnu,
    ), "The masked spectra with different units are not the same."


def test_masked_part_spectra(particle_stars_B, reprocessed_emission_model):
    """Test the effect of masking during the generation of spectra."""
    # Make the model per particle
    reprocessed_emission_model.set_per_particle(True)

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

    assert np.any(mask), f"The mask is empty (defined in yr): {mask}"
    assert False in mask, f"The mask is all True: {mask}"

    masked_spec = particle_stars_B.get_spectra(
        reprocessed_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure the masked particles have zeroed spectra
    assert np.allclose(
        masked_spec._lnu[~mask],
        0.0,
    ), "The masked particles have non-zero spectra."

    # Ensure that the masked spectra are different
    assert not np.allclose(
        unmasked_spec._lnu,
        masked_spec._lnu,
    ), "The masked and unmasked integrated spectra are the same."

    # Ensure the masked spectra are less than the unmasked spectra
    assert np.sum(masked_spec._lnu) < np.sum(unmasked_spec._lnu), (
        "The masked integrated spectra is not less than "
        "the unmasked integrated spectra."
    )


def test_pacman_spectra(
    particle_stars_B,
    pacman_emission_model,
    bimodal_pacman_emission_model,
):
    """Test the generation of PACMAN spectra."""
    # Set the fixed tau_vs
    particle_stars_B.tau_v = 0.1
    particle_stars_B.tau_v_birth = 0.3
    particle_stars_B.tau_v_ism = 0.1

    # Set the age pivot for the B stars
    bimodal_pacman_emission_model.age_pivot = 6.7 * dimensionless

    # Generate the PACMAN spectra
    pacman_spec = particle_stars_B.get_spectra(
        pacman_emission_model,
        grid_assignment_method="ngp",
    )
    particle_stars_B.clear_all_emissions()

    # Generate the bimodal PACMAN spectra
    bimodal_pacman_spec = particle_stars_B.get_spectra(
        bimodal_pacman_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the two PACMAN spectra are different to validate that the
    # age pivot is working in the bimodal PACMAN model
    assert not np.allclose(
        pacman_spec._lnu,
        bimodal_pacman_spec._lnu,
    ), "The PACMAN and bimodal PACMAN spectra are the same."


def test_masked_combination(
    particle_stars_B,
    transmitted_emission_model,
    test_grid,
):
    """Test that the combination of two masked spectra is correct."""
    # Get the spectra without any masks
    unmasked_spec = particle_stars_B.get_spectra(
        transmitted_emission_model,
        grid_assignment_method="ngp",
    )
    particle_stars_B.clear_all_emissions()

    # Add a mask to the emission model
    transmitted_emission_model.add_mask(
        attr="ages",
        op=">=",
        thresh=5.5 * Myr,
        set_all=True,
    )

    # Generate the old masked spectra
    old_spec = particle_stars_B.get_spectra(
        transmitted_emission_model,
        grid_assignment_method="ngp",
    )
    particle_stars_B.clear_all_emissions()

    # Ensure there is some emission
    assert np.sum(old_spec._lnu) > 0, "The old spectra has no emission."

    # And now swap the mask to a young mask
    transmitted_emission_model.clear_masks(True)
    transmitted_emission_model.add_mask(
        attr="ages",
        op="<",
        thresh=5.5 * Myr,
        set_all=True,
    )

    # Generate the yound masked spectra
    young_spec = particle_stars_B.get_spectra(
        transmitted_emission_model,
        grid_assignment_method="ngp",
    )
    particle_stars_B.clear_all_emissions()

    # Ensure there is some emission
    assert np.sum(young_spec._lnu) > 0, "The young spectra has no emission."

    # Combine the two masked spectra
    combined_spec = old_spec + young_spec
    combined_array = old_spec._lnu + young_spec._lnu

    # Ensure that the masked spectra are different
    assert not np.allclose(
        young_spec._lnu,
        old_spec._lnu,
    ), "The old and young integrated spectra are the same."

    # Ensure the arrays are the same
    assert np.allclose(
        combined_spec._lnu,
        combined_array,
    ), "The combined and summed arrays are not the same."

    # Ensure the combined spectra are the same as the unmasked spectra
    assert np.allclose(
        combined_spec._lnu,
        unmasked_spec._lnu,
    ), (
        "The combined and unmasked integrated spectra are not the same "
        f"(combined={np.sum(combined_spec._lnu)} vs"
        f" unmasked={np.sum(unmasked_spec._lnu)})."
    )
