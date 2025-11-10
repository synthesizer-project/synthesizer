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
        test_grid.metallicity,
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


class TestCombinationMasking:
    """Test suite for combination operations with masking."""

    class TestIntegratedSpectraMasking:
        """Tests for integrated spectra combination with masks."""

        def test_wavelength_mask_basic(
            self,
            random_part_stars,
            test_grid,
            incident_emission_model,
            transmitted_emission_model,
        ):
            """Test combining integrated spectra with wavelength mask."""
            # Create a wavelength mask (first half unmasked)
            lam_mask = np.zeros(test_grid.lam.size, dtype=bool)
            lam_mask[: test_grid.lam.size // 2] = True

            # Get the base spectra
            incident_spec = random_part_stars.get_spectra(
                incident_emission_model
            )
            transmitted_spec = random_part_stars.get_spectra(
                transmitted_emission_model
            )

            # Create combination model with wavelength mask
            masked_combined_model = StellarEmissionModel(
                label="masked_combined",
                combine=(
                    incident_emission_model,
                    transmitted_emission_model,
                ),
                lam_mask=lam_mask,
                emitter="stellar",
            )

            # Get masked combined spectra
            masked_combined_spec = random_part_stars.get_spectra(
                masked_combined_model
            )

            # Unmasked region: should be sum of both
            expected_unmasked = (
                incident_spec.lnu[lam_mask] + transmitted_spec.lnu[lam_mask]
            )
            assert np.allclose(
                masked_combined_spec.lnu[lam_mask],
                expected_unmasked,
            ), "Unmasked region should be sum of both spectra"

            # Masked region: should be first spectrum only
            assert np.allclose(
                masked_combined_spec.lnu[~lam_mask],
                incident_spec.lnu[~lam_mask],
            ), "Masked region should equal first spectrum"

        def test_multiple_spectra_with_wavelength_mask(
            self,
            random_part_stars,
            test_grid,
            incident_emission_model,
            transmitted_emission_model,
            nebular_emission_model,
        ):
            """Test combining three spectra with wavelength mask."""
            # Create mask (middle third unmasked)
            lam_mask = np.zeros(test_grid.lam.size, dtype=bool)
            lam_mask[test_grid.lam.size // 3 : 2 * test_grid.lam.size // 3] = (
                True
            )

            # Get base spectra
            incident_spec = random_part_stars.get_spectra(
                incident_emission_model
            )
            transmitted_spec = random_part_stars.get_spectra(
                transmitted_emission_model
            )
            nebular_spec = random_part_stars.get_spectra(
                nebular_emission_model
            )

            # Create combination model
            combined_model = StellarEmissionModel(
                label="three_combined",
                combine=(
                    incident_emission_model,
                    transmitted_emission_model,
                    nebular_emission_model,
                ),
                lam_mask=lam_mask,
                emitter="stellar",
            )

            combined_spec = random_part_stars.get_spectra(combined_model)

            # Unmasked: sum of all three
            expected_unmasked = (
                incident_spec.lnu[lam_mask]
                + transmitted_spec.lnu[lam_mask]
                + nebular_spec.lnu[lam_mask]
            )
            assert np.allclose(
                combined_spec.lnu[lam_mask],
                expected_unmasked,
            ), "Unmasked region should be sum of all three spectra"

            # Masked: first spectrum only
            assert np.allclose(
                combined_spec.lnu[~lam_mask],
                incident_spec.lnu[~lam_mask],
            ), "Masked region should equal first spectrum"

    class TestPerParticleSpectraMasking:
        """Tests for per-particle spectra combination with masks."""

        def test_particle_mask_only(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test combining per-particle spectra with particle mask only."""
            # Create two simple extraction models
            model_a = StellarEmissionModel(
                label="per_particle_a",
                grid=test_grid,
                extract="incident",
                per_particle=True,
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="per_particle_b",
                grid=test_grid,
                extract="transmitted",
                per_particle=True,
                emitter="stellar",
            )

            # Get per-particle spectra
            spec_a = random_part_stars.get_spectra(model_a)
            spec_b = random_part_stars.get_spectra(model_b)

            # Create particle mask based on mass (first half)
            mass_threshold = np.median(random_part_stars.initial_masses)
            part_mask = random_part_stars.initial_masses > mass_threshold

            # Create combination with particle mask (no wavelength mask)
            combined_model = StellarEmissionModel(
                label="particle_masked_combined",
                combine=(model_a, model_b),
                mask_attr="initial_masses",
                mask_op=">",
                mask_thresh=mass_threshold,
                per_particle=True,
                emitter="stellar",
            )

            combined_spec = random_part_stars.get_spectra(combined_model)

            # Check shape is correct
            assert combined_spec.lnu.shape == spec_a.lnu.shape
            # Per-particle spectra have 2D shape (nparticles, nlam)
            assert combined_spec.lnu.ndim == 2

            # CRITICAL: Verify masking behavior
            # For unmasked particles (high mass): should be sum
            # For masked particles (low mass): should be first spectrum only
            expected_unmasked = (
                spec_a.lnu[part_mask, :] + spec_b.lnu[part_mask, :]
            )
            assert np.allclose(
                combined_spec.lnu[part_mask, :],
                expected_unmasked,
            ), "Unmasked particles should have sum of both spectra"

            # Masked particles: should equal first spectrum
            assert np.allclose(
                combined_spec.lnu[~part_mask, :],
                spec_a.lnu[~part_mask, :],
            ), "Masked particles should equal first spectrum only"

        def test_wavelength_mask_per_particle(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test per-particle spectra with wavelength mask."""
            # Create models
            model_a = StellarEmissionModel(
                label="per_particle_wl_a",
                grid=test_grid,
                extract="incident",
                per_particle=True,
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="per_particle_wl_b",
                grid=test_grid,
                extract="transmitted",
                per_particle=True,
                emitter="stellar",
            )

            # Get spectra
            spec_a = random_part_stars.get_spectra(model_a)
            spec_b = random_part_stars.get_spectra(model_b)

            # Create wavelength mask
            lam_mask = np.zeros(test_grid.lam.size, dtype=bool)
            lam_mask[: test_grid.lam.size // 2] = True

            # Combine with wavelength mask
            combined_model = StellarEmissionModel(
                label="per_particle_wl_combined",
                combine=(model_a, model_b),
                lam_mask=lam_mask,
                per_particle=True,
                emitter="stellar",
            )

            combined_spec = random_part_stars.get_spectra(combined_model)

            # Unmasked wavelengths: should be sum
            expected_unmasked = (
                spec_a.lnu[:, lam_mask] + spec_b.lnu[:, lam_mask]
            )
            assert np.allclose(
                combined_spec.lnu[:, lam_mask],
                expected_unmasked,
            ), "Unmasked wavelengths should be sum"

            # Masked wavelengths: should be first spectrum
            assert np.allclose(
                combined_spec.lnu[:, ~lam_mask],
                spec_a.lnu[:, ~lam_mask],
            ), "Masked wavelengths should equal first spectrum"

        def test_both_masks_per_particle(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test per-particle spectra with particle+wavelength masks."""
            # Create models
            model_a = StellarEmissionModel(
                label="per_particle_both_a",
                grid=test_grid,
                extract="incident",
                per_particle=True,
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="per_particle_both_b",
                grid=test_grid,
                extract="transmitted",
                per_particle=True,
                emitter="stellar",
            )

            # Get spectra
            spec_a = random_part_stars.get_spectra(model_a)
            random_part_stars.get_spectra(model_b)

            # Create wavelength mask
            lam_mask = np.zeros(test_grid.lam.size, dtype=bool)
            lam_mask[test_grid.lam.size // 4 : 3 * test_grid.lam.size // 4] = (
                True
            )

            # Combine with both masks (particle mask via mask_attr)
            combined_model = StellarEmissionModel(
                label="per_particle_both_combined",
                combine=(model_a, model_b),
                mask_attr="initial_masses",
                mask_op=">",
                mask_thresh=np.median(random_part_stars.initial_masses),
                lam_mask=lam_mask,
                per_particle=True,
                emitter="stellar",
            )

            combined_spec = random_part_stars.get_spectra(combined_model)

            # Verify shape
            assert combined_spec.lnu.shape == spec_a.lnu.shape
            # Per-particle spectra have 2D shape (nparticles, nlam)
            assert combined_spec.lnu.ndim == 2

    class TestLineCombinationMasking:
        """Tests for line combination with masks."""

        @pytest.mark.skip(reason="Requires nebular grid with lines")
        def test_integrated_lines_with_wavelength_mask(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test combining integrated lines with wavelength mask."""
            # Create models that extract lines
            model_a = StellarEmissionModel(
                label="lines_a",
                grid=test_grid,
                extract="incident",
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="lines_b",
                grid=test_grid,
                extract="transmitted",
                emitter="stellar",
            )

            # Get lines
            lines_a = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=model_a,
            )
            lines_b = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=model_b,
            )

            # Create wavelength mask (mask second line)
            lam_mask = np.array([True, False])

            # Combine with mask
            combined_model = StellarEmissionModel(
                label="lines_combined",
                combine=(model_a, model_b),
                lam_mask=lam_mask,
                emitter="stellar",
            )

            combined_lines = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=combined_model,
            )

            # First line (unmasked): should be sum
            assert np.isclose(
                combined_lines.luminosity[0],
                lines_a.luminosity[0] + lines_b.luminosity[0],
            ), "Unmasked line should be sum"

            # Second line (masked): should be first model only
            assert np.isclose(
                combined_lines.luminosity[1],
                lines_a.luminosity[1],
            ), "Masked line should equal first model"

        @pytest.mark.skip(reason="Requires nebular grid with lines")
        def test_per_particle_lines_with_mask(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test combining per-particle lines with mask."""
            # Create per-particle models
            model_a = StellarEmissionModel(
                label="per_particle_lines_a",
                grid=test_grid,
                extract="incident",
                per_particle=True,
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="per_particle_lines_b",
                grid=test_grid,
                extract="transmitted",
                per_particle=True,
                emitter="stellar",
            )

            # Get per-particle lines
            lines_a = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=model_a,
            )
            lines_b = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=model_b,
            )

            # Create line mask
            lam_mask = np.array([True, False])

            # Combine with mask
            combined_model = StellarEmissionModel(
                label="per_particle_lines_combined",
                combine=(model_a, model_b),
                lam_mask=lam_mask,
                per_particle=True,
                emitter="stellar",
            )

            combined_lines = random_part_stars.get_lines(
                line_ids=["H 1 4862.69A", "O 3 5006.84A"],
                emission_model=combined_model,
            )

            # Check shapes
            assert combined_lines.luminosity.shape == lines_a.luminosity.shape

            # First line (unmasked): should be sum
            expected_first = (
                lines_a.luminosity[:, 0] + lines_b.luminosity[:, 0]
            )
            assert np.allclose(
                combined_lines.luminosity[:, 0],
                expected_first,
            ), "Unmasked per-particle lines should be sum"

            # Second line (masked): should be first model
            assert np.allclose(
                combined_lines.luminosity[:, 1],
                lines_a.luminosity[:, 1],
            ), "Masked per-particle lines should equal first model"

    class TestEdgeCases:
        """Tests for edge cases and error handling in combination masking."""

        def test_single_model_combination_with_mask(
            self,
            random_part_stars,
            test_grid,
            incident_emission_model,
        ):
            """Test combining single model with mask (masking hack)."""
            # Get base spectrum
            incident_spec = random_part_stars.get_spectra(
                incident_emission_model
            )

            # Create mask
            lam_mask = np.zeros(test_grid.lam.size, dtype=bool)
            lam_mask[: test_grid.lam.size // 2] = True

            # Combine single model with mask
            masked_model = StellarEmissionModel(
                label="single_masked",
                combine=(incident_emission_model,),
                lam_mask=lam_mask,
                emitter="stellar",
            )

            masked_spec = random_part_stars.get_spectra(masked_model)

            # Unmasked region: should equal original
            assert np.allclose(
                masked_spec.lnu[lam_mask],
                incident_spec.lnu[lam_mask],
            ), "Unmasked region should equal original"

            # Masked region: should also equal original (only one model)
            assert np.allclose(
                masked_spec.lnu[~lam_mask],
                incident_spec.lnu[~lam_mask],
            ), "Masked region should equal original for single model"

        def test_particle_mask_raises_error_for_integrated(
            self,
            random_part_stars,
            test_grid,
            incident_emission_model,
            transmitted_emission_model,
        ):
            """Test that particle masks raise error for integrated spectra."""
            # When combining integrated spectra with mask_attr set,
            # this should raise an error during combination
            from synthesizer import exceptions

            combined_model = StellarEmissionModel(
                label="extraction_mask_test",
                combine=(
                    incident_emission_model,
                    transmitted_emission_model,
                ),
                mask_attr="initial_masses",
                mask_op=">",
                mask_thresh=np.median(random_part_stars.initial_masses),
                emitter="stellar",
            )

            # This should raise InvalidCombination error
            with pytest.raises(exceptions.InvalidCombination):
                random_part_stars.get_spectra(combined_model)

        def test_combination_with_nans_in_spectra(
            self,
            random_part_stars,
            test_grid,
        ):
            """Test that NaNs in spectra are handled correctly."""
            # Create models
            model_a = StellarEmissionModel(
                label="nan_test_a",
                grid=test_grid,
                extract="incident",
                emitter="stellar",
            )

            model_b = StellarEmissionModel(
                label="nan_test_b",
                grid=test_grid,
                extract="transmitted",
                emitter="stellar",
            )

            # Get spectra
            spec_a = random_part_stars.get_spectra(model_a)
            spec_b = random_part_stars.get_spectra(model_b)

            # Manually add NaNs to first spectrum
            spec_a._lnu[10:20] = np.nan

            # Combine
            combined_model = StellarEmissionModel(
                label="nan_combined",
                combine=(model_a, model_b),
                emitter="stellar",
            )

            combined_spec = random_part_stars.get_spectra(combined_model)

            # Where spec_a has NaNs, result should just be spec_b
            assert np.allclose(
                combined_spec.lnu[10:20],
                spec_b.lnu[10:20],
            ), "NaN regions should only have non-NaN spectrum"

            # Where both valid, should be sum
            valid_region = slice(30, 40)
            assert np.allclose(
                combined_spec.lnu[valid_region],
                spec_a.lnu[valid_region] + spec_b.lnu[valid_region],
            ), "Valid regions should be sum"
