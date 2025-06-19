"""Tests for generating spectra."""

import numpy as np
import pytest
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


class TestPacmanSpectra:
    """Test the generation of PACMAN spectra."""

    def test_pacman_spectra(self, particle_stars_B, pacman_emission_model):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1

        # Generate the PACMAN spectra
        pacman_spec = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Ensure that the PACMAN spectra is not empty
        assert np.sum(pacman_spec._lnu) > 0, (
            "The PACMAN spectra has no emission."
        )

    def test_emergent_is_combo(self, particle_stars_B, pacman_emission_model):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure emergent exists
        assert "emergent" in particle_stars_B.spectra, (
            "The emergent spectra is not in the particle stars."
        )

        # Ensure emergent is nonzero
        assert np.sum(particle_stars_B.spectra["emergent"]._lnu) > 0, (
            "The emergent spectra has no emission."
        )

        # Ensure escape is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure emergent is equal to attenuated + escaped
        explicit_emergent = (
            particle_stars_B.spectra["attenuated"]
            + particle_stars_B.spectra["escaped"]
        )
        assert np.allclose(
            particle_stars_B.spectra["emergent"]._lnu,
            explicit_emergent._lnu,
        ), (
            "The emergent spectra is not equal to the sum of the "
            "attenuated and escaped spectra."
        )

    def test_intrinsic_is_combo(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure intrinsic exists
        assert "intrinsic" in particle_stars_B.spectra, (
            "The intrinsic spectra is not in the particle stars."
        )

        # Ensure intrinsic is nonzero
        assert np.sum(particle_stars_B.spectra["intrinsic"]._lnu) > 0, (
            "The intrinsic spectra has no emission."
        )

        # Ensure escape is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure intrinsic is equal to reprocessed + escaped
        explicit_intrinsic = (
            particle_stars_B.spectra["reprocessed"]
            + particle_stars_B.spectra["escaped"]
        )
        assert np.allclose(
            particle_stars_B.spectra["intrinsic"]._lnu,
            explicit_intrinsic._lnu,
        ), (
            "The intrinsic spectra is not equal to the sum of the "
            "reprocessed and escaped spectra."
        )

    def test_transmitted_escaped(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure transmitted exists
        assert "transmitted" in particle_stars_B.spectra, (
            "The transmitted spectra is not in the particle stars."
        )

        # Ensure transmitted is nonzero
        assert np.sum(particle_stars_B.spectra["transmitted"]._lnu) > 0, (
            "The transmitted spectra has no emission."
        )

        # Ensure escaped exists
        assert "escaped" in particle_stars_B.spectra, (
            "The escaped spectra is not in the particle stars."
        )

        # Ensure escaped is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure incident is nonzero
        assert np.sum(particle_stars_B.spectra["incident"]._lnu) > 0, (
            "The incident spectra has no emission."
        )

        # Ensure escaped equals 0.5 * incident
        assert np.allclose(
            particle_stars_B.spectra["escaped"]._lnu,
            0.5 * particle_stars_B.spectra["incident"]._lnu,
        ), (
            "The escaped spectra is not equal to 0.5 times the incident "
            "spectra."
        )

    def test_reprocessed_is_combo(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure reprocessed exists
        assert "reprocessed" in particle_stars_B.spectra, (
            "The reprocessed spectra is not in the particle stars."
        )

        # Ensure reprocessed is nonzero
        assert np.sum(particle_stars_B.spectra["reprocessed"]._lnu) > 0, (
            "The reprocessed spectra has no emission."
        )

        # Ensure nebular exists
        assert "nebular" in particle_stars_B.spectra, (
            "The nebular spectra is not in the particle stars."
        )

        # Ensure nebular is nonzero
        assert np.sum(particle_stars_B.spectra["nebular"]._lnu) > 0, (
            "The nebular spectra has no emission."
        )

        # Ensure reprocessed is equal to nebular + transmitted
        explicit_reprocessed = (
            particle_stars_B.spectra["nebular"]
            + particle_stars_B.spectra["transmitted"]
        )
        assert np.allclose(
            particle_stars_B.spectra["reprocessed"]._lnu,
            explicit_reprocessed._lnu,
        ), (
            "The reprocessed spectra is not equal to the sum of the "
            "nebular and transmitted spectra."
        )


class TestBimodalPacmanEmission:
    """Test the generation of Bimodal PACMAN spectra."""

    @pytest.fixture(autouse=True)
    def setup(self, particle_stars_B, bimodal_pacman_emission_model):
        """Setup for all tests in this class."""
        # common setup for all tests
        self.ps = particle_stars_B
        self.model = bimodal_pacman_emission_model

        # fix optical depths & escape fractions
        self.ps.tau_v = 0.1
        self.ps.tau_v_birth = 0.3
        self.ps.tau_v_ism = 0.1
        self.ps.fesc = 0.5
        self.ps.fesc_ly_alpha = 0.2

        # generate spectra
        self.spec = self.ps.get_spectra(
            self.model,
            grid_assignment_method="ngp",
        )

    def test_incident_components(self):
        """Test the incident components of the spectra."""
        # incident spectra should include young, old, combined
        for key in ("young_incident", "old_incident", "incident"):
            assert key in self.ps.spectra, f"{key} missing"
            arr = self.ps.spectra[key]._lnu
            assert np.sum(arr) > 0, f"{key} has no emission"

    def test_transmitted_and_escaped_components(self):
        """Test the transmitted and escaped components of the spectra."""
        # transmitted + escaped for young, old, combined
        for kind in ("transmitted", "escaped"):
            # combined
            assert kind in self.ps.spectra, f"{kind} missing"
            assert np.sum(self.ps.spectra[kind]._lnu) > 0

            # young & old
            for pop in ("young", "old"):
                key = f"{pop}_{kind}"
                assert key in self.ps.spectra, f"{key} missing"
                assert np.sum(self.ps.spectra[key]._lnu) > 0

        # check escaped = fesc * incident for each pop & combined
        for pop in ("young", "old", ""):
            inc = self.ps.spectra[f"{pop + '_' if pop else ''}incident"]._lnu
            esc = self.ps.spectra[f"{pop + '_' if pop else ''}escaped"]._lnu
            assert np.allclose(esc, 0.5 * inc), (
                f"{pop or 'combined'} escaped != fesc * incident"
            )

    def test_nebular_and_reprocessed_components(self):
        """Test the nebular and reprocessed components of the spectra."""
        # nebular
        for key in ("young_nebular", "old_nebular", "nebular"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        # reprocessed = nebular + transmitted
        combined_reproc = (
            self.ps.spectra["nebular"] + self.ps.spectra["transmitted"]
        )
        assert np.allclose(
            self.ps.spectra["reprocessed"]._lnu,
            combined_reproc._lnu,
        ), "reprocessed != nebular + transmitted"

        # per-population
        for pop in ("young", "old"):
            neb = self.ps.spectra[f"{pop}_nebular"]
            trn = self.ps.spectra[f"{pop}_transmitted"]
            rep = self.ps.spectra[f"{pop}_reprocessed"]
            assert np.allclose(
                rep._lnu,
                (neb + trn)._lnu,
            ), f"{pop}_reprocessed != {pop}_nebular + {pop}_transmitted"

    def test_intrinsic_components(self):
        """Test the intrinsic components of the spectra."""
        # intrinsic exists and nonzero
        for key in ("young_intrinsic", "old_intrinsic", "intrinsic"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        # intrinsic = reprocessed + escaped
        combined_intr = (
            self.ps.spectra["reprocessed"] + self.ps.spectra["escaped"]
        )
        assert np.allclose(
            self.ps.spectra["intrinsic"]._lnu,
            combined_intr._lnu,
        ), "intrinsic != reprocessed + escaped"

    def test_attenuated_components(self):
        """Test the attenuated components of the spectra."""
        # attenuated combined
        assert "attenuated" in self.ps.spectra
        assert np.sum(self.ps.spectra["attenuated"]._lnu) > 0

        # young & old attenuated nebular/ism totals exist
        for pop in ("young",):
            for sub in ("attenuated_nebular", "attenuated_ism", "attenuated"):
                key = f"{pop}_{sub}"
                assert key in self.ps.spectra, f"{key} missing"
                assert np.sum(self.ps.spectra[key]._lnu) > 0

        # old_attenuated
        assert "old_attenuated" in self.ps.spectra
        assert np.sum(self.ps.spectra["old_attenuated"]._lnu) > 0

    def test_emergent_components(self):
        """Test the emergent components of the spectra."""
        # emergent = attenuated + escaped
        for key in ("young_emergent", "old_emergent", "emergent"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        combined_emergent = (
            self.ps.spectra["attenuated"] + self.ps.spectra["escaped"]
        )
        assert np.allclose(
            self.ps.spectra["emergent"]._lnu,
            combined_emergent._lnu,
        ), "emergent != attenuated + escaped"
