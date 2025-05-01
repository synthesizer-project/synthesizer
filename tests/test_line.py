"""Tests for the LineCollection class in the emissions.line module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo
from unyt import (
    Hz,
    Myr,
    angstrom,
    c,
    cm,
    dimensionless,
    erg,
    eV,
    h,
    pc,
    s,
    yr,
)

from synthesizer.conversions import standard_to_vacuum
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emissions import LineCollection
from synthesizer.emissions.line_ratios import ratios
from synthesizer.emissions.utils import O2, O3, Hb, O3b, O3r


class TestLineCollectionInitialization:
    """Test LineCollection initialization and basic properties."""

    def test_initialization(self, simple_line_collection):
        """Test that the LineCollection initializes correctly."""
        lines = simple_line_collection

        assert lines.nlines == 2
        assert np.array_equal(
            lines.line_ids, np.array(["O III 5007 A", "H 1 6563 A"])
        )
        assert lines.lam.units == angstrom
        assert lines.luminosity.units == erg / s
        assert lines.continuum.units == erg / s / Hz

    def test_basic_properties(self, simple_line_collection):
        """Test basic properties of the LineCollection."""
        lines = simple_line_collection

        # Test elements property
        assert "O" in lines.elements
        assert "H" in lines.elements

        # Test vacuum wavelengths
        vacuum_wavelengths = standard_to_vacuum(lines.lam)
        assert np.allclose(lines.vacuum_wavelengths, vacuum_wavelengths)

        # Test frequency
        nu = (c / lines.lam).to(Hz)
        assert np.allclose(lines.nu, nu)

        # Test energy
        energy = (h * c / lines.lam).to(eV)
        assert np.allclose(lines.energy, energy)

        # Test continuum_llam and equivalent_width
        assert hasattr(lines, "continuum_llam")
        assert hasattr(lines, "equivalent_width")


class TestLineCollectionOperations:
    """Test LineCollection operations."""

    def test_indexing(self, simple_line_collection):
        """Test accessing individual lines by index."""
        lines = simple_line_collection

        # Test accessing by line ID
        oiii_line = lines["O III 5007 A"]
        assert oiii_line.nlines == 1
        assert oiii_line.line_ids[0] == "O III 5007 A"
        assert oiii_line.lam.value == 5007

        # Test accessing by list of line IDs
        subset = lines[["O III 5007 A", "H 1 6563 A"]]
        assert subset.nlines == 2

        # Test accessing non-existent line
        with pytest.raises(Exception):
            lines["non_existent_line"]

    def test_scaling(self, simple_line_collection):
        """Test scaling the line collection."""
        # Scale by unitless scalar
        scaled_lines = simple_line_collection.scale(2.0)
        assert np.allclose(
            scaled_lines.luminosity.value,
            simple_line_collection.luminosity.value * 2.0,
        ), (
            f"{scaled_lines.luminosity.value} !="
            f" {simple_line_collection.luminosity.value * 2.0}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            simple_line_collection.continuum.value * 2.0,
        ), (
            f"{scaled_lines.continuum.value} !="
            f" {simple_line_collection.continuum.value * 2.0}"
        )

        # Scale with scalar with units
        scaled_lines = simple_line_collection.scale(2.0 * erg / s)
        assert np.allclose(
            scaled_lines.luminosity.value,
            simple_line_collection.luminosity.value * 2.0,
        ), (
            f"{scaled_lines.luminosity.value} !="
            f" {simple_line_collection.luminosity.value * 2.0}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            simple_line_collection.continuum.value
            * 2.0
            / simple_line_collection.nu.value,
        ), (
            f"{scaled_lines.continuum.value} !="
            f" {simple_line_collection.continuum.value * 2.0}"
        )
        assert scaled_lines.luminosity.units == erg / s, (
            f"{scaled_lines.luminosity.units} != erg/s"
        )

        # Scale with scalar continuum units
        scaled_lines = simple_line_collection.scale(2.0 * erg / s / Hz)
        expected_lum = (
            simple_line_collection.luminosity.value
            * 2.0
            * simple_line_collection.nu.value
        )
        expected_cont = simple_line_collection.continuum.value * 2.0
        assert np.allclose(
            scaled_lines.luminosity.value,
            expected_lum,
        ), (
            f"Scaled luminosity doesn't match "
            f"{scaled_lines.luminosity.value} !="
            f" {expected_lum}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            expected_cont,
        ), (
            f"Scaled continuum doesn't match "
            f"{scaled_lines.continuum.value} !="
            f" {expected_cont}"
        )

        # Test inplace scaling
        original_lum = simple_line_collection.luminosity.copy()
        simple_line_collection.scale(3.0, inplace=True)
        assert np.allclose(
            simple_line_collection.luminosity.value,
            original_lum.value * 3.0,
        ), (
            f"{simple_line_collection.luminosity.value} !="
            f" {original_lum.value * 3.0}"
        )

    def test_scaling_multidim(self, multi_dimension_line_collection):
        """Test scaling the line collection with multiple dimensions."""
        lines = multi_dimension_line_collection

        # Scale by unitless scalar
        scaled_lines = lines.scale(2.0)
        assert np.allclose(
            scaled_lines.luminosity.value,
            lines.luminosity.value * 2.0,
        ), (
            "Scaled luminosity doesn't match "
            f"{scaled_lines.luminosity.value} !="
            f" {lines.luminosity.value * 2.0}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            lines.continuum.value * 2.0,
        ), (
            "Scaled continuum doesn't match "
            f"{scaled_lines.continuum.value} !="
            f" {lines.continuum.value * 2.0}"
        )

        # Scale with scalar with units
        scaled_lines = lines.scale(2.0 * erg / s)
        assert np.allclose(
            scaled_lines.luminosity.value,
            lines.luminosity.value * 2.0,
        ), (
            "Scaled luminosity doesn't match "
            f"{scaled_lines.luminosity.value} !="
            f" {lines.luminosity.value * 2.0}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            lines.continuum.value * 2.0 / lines.nu.value,
        ), (
            "Scaled continuum doesn't match "
            f"{scaled_lines.continuum.value} !="
            f" {lines.continuum.value * 2.0 / lines.nu.value}"
        )
        assert scaled_lines.luminosity.units == erg / s, (
            f"{scaled_lines.luminosity.units} != erg/s"
        )

        # Scale with scalar continuum units
        scaled_lines = lines.scale(2.0 * erg / s / Hz)
        expected_lum = lines.luminosity.value * 2.0 * lines.nu.value
        expected_cont = lines.continuum.value * 2.0
        assert np.allclose(
            scaled_lines.luminosity.value,
            expected_lum,
        ), (
            "Scaled luminosity doesn't match "
            f"{scaled_lines.luminosity.value} !="
            f" {expected_lum}"
        )
        assert np.allclose(
            scaled_lines.continuum.value,
            expected_cont,
        ), (
            "Scaled continuum doesn't match "
            f"{scaled_lines.continuum.value} !="
            f" {expected_cont}"
        )

    def test_addition(self, simple_line_collection):
        """Test adding line collections."""
        lines = simple_line_collection

        # Add the same collection to itself
        sum_lines = lines + lines
        assert sum_lines.nlines == lines.nlines
        assert np.allclose(
            sum_lines.luminosity.value, lines.luminosity.value * 2
        )
        assert np.allclose(
            sum_lines.continuum.value, lines.continuum.value * 2
        )

        # Test adding incompatible collections
        other_lines = LineCollection(
            line_ids=["N II 6584 A", "S II 6716 A"],
            lam=np.array([6584, 6716]) * angstrom,
            lum=np.array([1e40, 1e39]) * erg / s,
            cont=np.array([1e38, 1e37]) * erg / s / Hz,
        )

        with pytest.raises(Exception):
            lines + other_lines

    def test_multiplication(self, simple_line_collection):
        """Test multiplying line collections."""
        lines = simple_line_collection

        # Test left multiplication
        scaled_lines = 2.0 * lines
        assert np.allclose(
            scaled_lines.luminosity.value, lines.luminosity.value * 2.0
        )

        # Test right multiplication
        scaled_lines = lines * 3.0
        assert np.allclose(
            scaled_lines.luminosity.value, lines.luminosity.value * 3.0
        )

    def test_iteration(self, simple_line_collection):
        """Test iteration over line collection."""
        lines = simple_line_collection

        # Iterate and collect individual lines
        collected_lines = []
        for line in lines:
            collected_lines.append(line)

        assert len(collected_lines) == lines.nlines
        assert all(
            isinstance(line, LineCollection) for line in collected_lines
        )


class TestLineCollectionFlux:
    """Test flux calculation methods."""

    def test_get_flux0(self, simple_line_collection):
        """Test get_flux0 method that calculates rest frame fluxes."""
        lines = simple_line_collection

        flux = lines.get_flux0()
        assert flux.units == erg / s / cm**2
        assert lines.obslam is not None
        assert lines.flux is not None
        assert lines.continuum_flux is not None

        # Check the flux calculation
        expected_flux = lines.luminosity / (4 * np.pi * (10 * pc) ** 2)
        assert np.allclose(flux, expected_flux)

    @pytest.mark.parametrize("z", [0.1, 1.0, 2.0])
    def test_get_flux(self, simple_line_collection, z, test_grid):
        """Test get_flux method with different redshifts."""
        lines = simple_line_collection

        flux = lines.get_flux(cosmo, z)
        assert flux.units == erg / s / cm**2

        # Check observed wavelength is correctly redshifted
        assert np.allclose(lines.obslam, lines.lam * (1 + z))

    def test_get_flux_with_igm(self, simple_line_collection, test_grid):
        """Test get_flux method with IGM attenuation."""
        # Skip if IGM module is not available
        pytest.importorskip("synthesizer.emission_models.transformers.igm")

        from synthesizer.emission_models.transformers.igm import Inoue14

        lines = simple_line_collection
        z = 3.0  # High enough redshift to see IGM effects

        # Get flux with and without IGM
        flux_no_igm = lines.get_flux(cosmo, z)
        flux_with_igm = lines.get_flux(cosmo, z, igm=Inoue14)

        # IGM should reduce the flux
        assert np.all(flux_with_igm <= flux_no_igm)


class TestLineRatiosAndDiagrams:
    """Test line ratio and diagram functionality."""

    def test_available_ratios(self, line_ratio_collection):
        """Test that available ratios are correctly identified."""
        lines = line_ratio_collection

        # The collection should have some available ratios
        assert len(lines.available_ratios) > 0

        # Check specific ratios
        if "R23" in ratios:
            assert "R23" in lines.available_ratios
        if "O3Hb" in ratios:
            assert "O3Hb" in lines.available_ratios

    def test_get_ratio(self, line_ratio_collection):
        """Test getting line ratios."""
        lines = line_ratio_collection

        # Get a ratio if one is available
        if len(lines.available_ratios) > 0:
            ratio_id = lines.available_ratios[0]
            ratio = lines.get_ratio(ratio_id)
            assert isinstance(ratio, float)

        # Test custom ratio
        if "O III 5007 A" in lines.line_ids and "H 1 4861 A" in lines.line_ids:
            ratio = lines.get_ratio(["O III 5007 A", "H 1 4861 A"])
            expected = float(
                lines["O III 5007 A"].luminosity
                / lines["H 1 4861 A"].luminosity
            )
            assert ratio == expected

    def test_available_diagrams(self, line_ratio_collection):
        """Test that available diagrams are correctly identified."""
        lines = line_ratio_collection

        # The collection should have some available diagrams
        assert hasattr(lines, "available_diagrams")

    def test_get_diagram(self, line_ratio_collection):
        """Test getting line diagrams."""
        lines = line_ratio_collection

        # Get a diagram if one is available
        if len(lines.available_diagrams) > 0:
            diagram_id = lines.available_diagrams[0]
            diagram = lines.get_diagram(diagram_id)
            assert isinstance(diagram, tuple)
            assert len(diagram) == 2
            assert all(isinstance(value, float) for value in diagram)


class TestLineCollectionManipulation:
    """Test advanced line collection manipulation methods."""

    def test_sum(self, multi_dimension_line_collection):
        """Test summing lines across dimensions."""
        lines = multi_dimension_line_collection

        # Sum all dimensions except the last
        summed_lines = lines.sum()
        assert summed_lines.nlines == lines.nlines
        assert summed_lines.luminosity.shape[-1] == lines.luminosity.shape[-1]
        assert len(summed_lines.luminosity.shape) == 1

        # Sum specific axis
        if lines.ndim > 1:
            axis_summed = lines.sum(axis=0)
            assert axis_summed.nlines == lines.nlines
            assert axis_summed.luminosity.shape != lines.luminosity.shape

    def test_concat(self, simple_line_collection):
        """Test concatenating line collections."""
        lines = simple_line_collection

        # Create a similar collection
        other_lines = LineCollection(
            line_ids=lines.line_ids,
            lam=lines.lam,
            lum=lines.luminosity * 2,
            cont=lines.continuum * 2,
        )

        # Concatenate them
        concat_lines = lines.concat(other_lines)

        assert concat_lines.nlines == lines.nlines
        assert concat_lines.nlines == concat_lines.luminosity.shape[-1]
        assert concat_lines.luminosity.shape[0] == 2

    def test_extend(self, simple_line_collection):
        """Test extending line collections."""
        lines1 = simple_line_collection["O III 5007 A"]
        lines2 = simple_line_collection["H 1 6563 A"]

        # Extend the collection
        extended_lines = lines1.extend(lines2)

        assert extended_lines.nlines == lines1.nlines + lines2.nlines
        assert (
            extended_lines.luminosity.shape[-1]
            == lines1.luminosity.shape[-1] + lines2.luminosity.shape[-1]
        )
        assert np.allclose(
            extended_lines.luminosity,
            simple_line_collection.luminosity,
        )
        assert np.all(
            extended_lines.line_ids == simple_line_collection.line_ids
        )

    def test_apply_attenuation(self, simple_line_collection):
        """Test applying dust attenuation."""
        lines = simple_line_collection

        # Create a simple power law dust curve
        dust_curve = PowerLaw(slope=-0.7)

        # Apply attenuation
        tau_v = 1.0
        attenuated_lines = lines.apply_attenuation(tau_v, dust_curve)

        # Attenuated lines should have lower luminosity
        assert np.all(attenuated_lines.luminosity < lines.luminosity)

    def test_get_blended_lines(self, line_ratio_collection):
        """Test blending lines based on wavelength bins."""
        lines = line_ratio_collection

        # Create wavelength bins
        bins = np.array([4000, 5000, 7000]) * angstrom

        # Blend lines
        blended_lines = lines.get_blended_lines(bins)

        # Should have fewer lines than the original
        assert blended_lines.nlines <= lines.nlines

        # Test with invalid bins
        with pytest.raises(Exception):
            lines.get_blended_lines(np.array([5000]) * angstrom)


class TestLineCollectionVisualization:
    """Test visualization methods."""

    def test_plot_lines(self, simple_line_collection):
        """Test plot_lines method."""
        lines = simple_line_collection

        # Create a plot
        fig, ax = lines.plot_lines(show=False)

        # Check that the plot was created
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Test with subset
        fig, ax = lines.plot_lines(
            subset=["O III 5007 A"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

        # Test with limits
        fig, ax = lines.plot_lines(
            xlimits=(4000, 7000),
            ylimits=(1e38, 1e41),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)


class TestLineCollectionGenerationMethods:
    """Test line generation with different methods."""

    def test_integrated_generation_ngp(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of intergrated line emission."""
        # Compute the lines using both integrated and per particle methods
        nebular_emission_model.set_per_particle(False)
        intregrated_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        random_part_stars.clear_all_emissions()
        nebular_emission_model.set_per_particle(True)
        per_particle_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        per_particle_lines = per_particle_lines.sum()

        # Ensure the luminosities match
        assert np.allclose(
            intregrated_lines.luminosity.value,
            per_particle_lines.luminosity.value,
        ), (
            "The integrated and summed per particles "
            "line luminosities do not match."
            f"{intregrated_lines.luminosity.value} != "
            f"{per_particle_lines.luminosity.value}"
        )

        # Ensure the continuum luminosities match
        assert np.allclose(
            intregrated_lines.continuum.value,
            per_particle_lines.continuum.value,
        ), (
            "The integrated and summed per particles "
            "continuum luminosities do not match."
            f"{intregrated_lines.continuum.value} != "
            f"{per_particle_lines.continuum.value}"
        )

    def test_masked_integrated_generation_ngp(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of intergrated line emission with a mask."""
        # Add a mask to the emission model
        nebular_emission_model.add_mask(
            attr="ages",
            op=">=",
            thresh=5 * Myr,
            set_all=True,
        )

        # Compute the lines using both the integrated and per
        # particle machinery
        nebular_emission_model.set_per_particle(False)
        integrated_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        random_part_stars.clear_all_emissions()
        nebular_emission_model.set_per_particle(True)
        per_particle_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        per_particle_lines = per_particle_lines.sum()

        # Ensure the luminosities match
        assert np.allclose(
            integrated_lines.luminosity.value,
            per_particle_lines.luminosity.value,
        ), (
            "The integrated and summed per particle "
            "line luminosities do not match."
            f"{integrated_lines.luminosity.value} != "
            f"{per_particle_lines.luminosity.value}"
        )

        # Ensure the continuum luminosities match
        assert np.allclose(
            integrated_lines.continuum.value,
            per_particle_lines.continuum.value,
        ), (
            "The integrated and summed per particle "
            "continuum luminosities do not match."
            f"{integrated_lines.continuum.value} != "
            f"{per_particle_lines.continuum.value}"
        )

    def test_integrated_generation_cic(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of intergrated line emission."""
        # Compute the lines using both integrated and per particle methods
        nebular_emission_model.set_per_particle(False)
        intregrated_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
        )
        random_part_stars.clear_all_emissions()
        nebular_emission_model.set_per_particle(True)
        per_particle_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
        )
        per_particle_lines = per_particle_lines.sum()

        # Ensure the luminosities match
        assert np.allclose(
            intregrated_lines.luminosity.value,
            per_particle_lines.luminosity.value,
        ), (
            "The integrated and summed per particles "
            "line luminosities do not match."
            f"{intregrated_lines.luminosity.value} != "
            f"{per_particle_lines.luminosity.value}"
        )

        # Ensure the continuum luminosities match
        assert np.allclose(
            intregrated_lines.continuum.value,
            per_particle_lines.continuum.value,
        ), (
            "The integrated and summed per particles "
            "continuum luminosities do not match."
            f"{intregrated_lines.continuum.value} != "
            f"{per_particle_lines.continuum.value}"
        )

    def test_masked_integrated_generation_cic(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of intergrated line emission with a mask."""
        # Add a mask to the emission model
        nebular_emission_model.add_mask(
            attr="ages",
            op=">=",
            thresh=5 * Myr,
            set_all=True,
        )

        # Compute the lines using both the integrated and per
        # particle machinery
        nebular_emission_model.set_per_particle(False)
        integrated_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
        )
        random_part_stars.clear_all_emissions()
        nebular_emission_model.set_per_particle(True)
        per_particle_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
        )
        per_particle_lines = per_particle_lines.sum()

        # Ensure the luminosities match
        assert np.allclose(
            integrated_lines.luminosity.value,
            per_particle_lines.luminosity.value,
        ), (
            "The integrated and summed per particle "
            "line luminosities do not match."
            f"{integrated_lines.luminosity.value} != "
            f"{per_particle_lines.luminosity.value}"
        )

        # Ensure the continuum luminosities match
        assert np.allclose(
            integrated_lines.continuum.value,
            per_particle_lines.continuum.value,
        ), (
            "The integrated and summed per particle "
            "continuum luminosities do not match."
            f"{integrated_lines.continuum.value} != "
            f"{per_particle_lines.continuum.value}"
        )

    def test_pacman_lines(
        self,
        test_grid,
        particle_stars_B,
        pacman_emission_model,
        bimodal_pacman_emission_model,
    ):
        """Test the generation of PACMAN lines."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1

        # Set the age pivot for the B stars
        bimodal_pacman_emission_model.age_pivot = 6.7 * dimensionless

        # Generate the PACMAN lines
        pacman_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            pacman_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Generate the bimodal PACMAN lines
        bimodal_pacman_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            bimodal_pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure that the two PACMAN lines are different to validate that the
        # age pivot is working in the bimodal PACMAN model
        assert not np.allclose(
            pacman_lines._luminosity,
            bimodal_pacman_lines._luminosity,
        ), "The PACMAN and bimodal PACMAN lines are the same."
        assert np.allclose(
            pacman_lines._luminosity.shape,
            bimodal_pacman_lines._luminosity.shape,
        ), "The PACMAN and bimodal PACMAN lines have different shapes."
        assert not np.allclose(
            pacman_lines._continuum,
            bimodal_pacman_lines._continuum,
        ), "The PACMAN and bimodal PACMAN continuum are the same."


class TestLineCollectionGenerationThreaded:
    """Test line generation with and without threading."""

    def test_threaded_generation_ngp_per_particle(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of lines with and without threading."""
        nebular_emission_model.set_per_particle(True)

        # Compute the lines using both the integrated and
        # per particle machinery
        serial_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
            nthreads=1,
        )
        random_part_stars.clear_all_emissions()
        threaded_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
            nthreads=4,
        )

        # Ensure that the integrated lines are the same
        assert np.allclose(
            serial_lines._luminosity, threaded_lines._luminosity
        ), "The serial and threaded lines are not the same."
        assert np.allclose(
            serial_lines._continuum, threaded_lines._continuum
        ), "The serial and threaded continuum are not the same."

    def test_threaded_generation_ngp_integrated(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of lines with and without threading."""
        nebular_emission_model.set_per_particle(False)

        # Compute the lines using both the integrated and
        # per particle machinery
        serial_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
            nthreads=1,
        )
        random_part_stars.clear_all_emissions()
        threaded_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
            nthreads=4,
        )

        # Ensure that the integrated lines are the same
        assert np.allclose(
            serial_lines._luminosity, threaded_lines._luminosity
        ), "The serial and threaded lines are not the same."
        assert np.allclose(
            serial_lines._continuum, threaded_lines._continuum
        ), "The serial and threaded continuum are not the same."

    def test_threaded_generation_cic_per_particle(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of lines with and without threading."""
        nebular_emission_model.set_per_particle(True)

        # Compute the lines using both the integrated and
        # per particle machinery
        serial_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
            nthreads=1,
        )
        random_part_stars.clear_all_emissions()
        threaded_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
            nthreads=4,
        )

        # Ensure that the integrated lines are the same
        assert np.allclose(
            serial_lines._luminosity, threaded_lines._luminosity
        ), "The serial and threaded lines are not the same."
        assert np.allclose(
            serial_lines._continuum, threaded_lines._continuum
        ), "The serial and threaded continuum are not the same."

    def test_threaded_generation_cic_integrated(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test the generation of lines with and without threading."""
        nebular_emission_model.set_per_particle(False)

        # Compute the lines using both the integrated and
        # per particle machinery
        serial_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
            nthreads=1,
        )
        random_part_stars.clear_all_emissions()
        threaded_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
            nthreads=4,
        )

        # Ensure that the integrated lines are the same
        assert np.allclose(
            serial_lines._luminosity, threaded_lines._luminosity
        ), "The serial and threaded lines are not the same."
        assert np.allclose(
            serial_lines._continuum, threaded_lines._continuum
        ), "The serial and threaded continuum are not the same."


class TestLineCollectionWeights:
    """Test the storage and reuse of grid weights."""

    def test_reusing_weights_ngp(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test reusing weights to calculate lines for the same grid."""
        # Compute the lines the first time
        first_lines = random_part_stars.get_lines(
            test_grid.available_lines,
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

        # Compute the lines the second time which will reuse the weights
        random_part_stars.clear_all_emissions()
        second_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure that the integrated lines are different
        assert np.allclose(
            first_lines._luminosity,
            second_lines._luminosity,
        ), "The first and second lines are not the same."
        assert np.allclose(
            first_lines._continuum,
            second_lines._continuum,
        ), "The first and second continuum are not the same."

    def test_reusing_weights_cic(
        self,
        test_grid,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test reusing weights to calculate lines for the same grid."""
        # Compute the lines the first time
        first_lines = random_part_stars.get_lines(
            test_grid.available_lines,
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

        # Compute the lines the second time which will reuse the weights
        random_part_stars.clear_all_emissions()
        second_lines = random_part_stars.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="cic",
        )

        # Ensure that the integrated lines are different
        assert np.allclose(
            first_lines._luminosity,
            second_lines._luminosity,
        ), "The first and second lines are not the same."
        assert np.allclose(
            first_lines._continuum,
            second_lines._continuum,
        ), "The first and second continuum are not the same."


class TestLineCollectionGenerationMasked:
    """Test line generation with masks."""

    def test_masked_int_lines(
        self,
        test_grid,
        particle_stars_B,
        reprocessed_emission_model,
    ):
        """Test the effect of masking during integrated line generation."""
        # Generate lines without masking
        unmasked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Generate lines with masking
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

        masked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure that the masked lines are different
        assert not np.allclose(
            unmasked_lines._luminosity,
            masked_lines._luminosity,
        ), "The masked and unmasked integrated lines are the same."
        assert not np.allclose(
            unmasked_lines._continuum,
            masked_lines._continuum,
        ), "The masked and unmasked continuum are the same."

        # Ensure the masked lines are less than the unmasked lines
        assert np.sum(masked_lines._luminosity) < np.sum(
            unmasked_lines._luminosity
        ), (
            "The masked integrated lines is not less than "
            "the unmasked integrated lines."
        )
        assert np.sum(masked_lines._continuum) < np.sum(
            unmasked_lines._continuum
        ), "The masked continuum is not less than the unmasked continuum"

    def test_masked_int_lines_diff_units(
        self,
        test_grid,
        particle_stars_B,
        reprocessed_emission_model,
    ):
        """Test applying masks with different units."""
        # Generate lines without masking
        unmasked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Generate lines with masking
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

        masked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure that the masked lines are different
        assert not np.allclose(
            unmasked_lines._luminosity,
            masked_lines._luminosity,
        ), "The masked and unmasked integrated lines are the same."
        assert not np.allclose(
            unmasked_lines._continuum,
            masked_lines._continuum,
        ), "The masked and unmasked continuum are the same."

        # Ensure the masked lines are less than the unmasked lines
        assert np.sum(masked_lines._luminosity) < np.sum(
            unmasked_lines._luminosity
        ), (
            "The masked integrated lines is not less than "
            "the unmasked integrated lines."
        )
        assert np.sum(masked_lines._continuum) < np.sum(
            unmasked_lines._continuum
        ), "The masked continuum is not less than the unmasked continuum"

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

        assert np.any(new_mask), (
            f"The mask is empty (defined in yr): {new_mask}"
        )
        assert False in new_mask, f"The mask is all True: {new_mask}"
        assert np.all(mask == new_mask), (
            "The masks with different units are not the same."
        )

        dif_units_masked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the masks with diffferent units have produced the same result
        assert np.allclose(
            masked_lines._luminosity,
            dif_units_masked_lines._luminosity,
        ), "The masked lines with different units are not the same."
        assert np.allclose(
            masked_lines._continuum,
            dif_units_masked_lines._continuum,
        ), "The masked continuum with different units are not the same."

    def test_masked_part_lines(
        self,
        test_grid,
        particle_stars_B,
        reprocessed_emission_model,
    ):
        """Test the effect of masking during per particle line generation."""
        reprocessed_emission_model.set_per_particle(True)

        # Generate lines without masking
        unmasked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Generate lines with masking
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

        assert np.any(mask), f"The mask is empty (defined in Myr): {mask}"
        assert False in mask, f"The mask is all True: {mask}"

        masked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            reprocessed_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure that the masked lines are different
        assert not np.allclose(
            unmasked_lines._luminosity,
            masked_lines._luminosity,
        ), "The masked and unmasked integrated lines are the same."
        assert not np.allclose(
            unmasked_lines._continuum,
            masked_lines._continuum,
        ), "The masked and unmasked continuum are the same."

        # Ensure the masked lines are less than the unmasked lines
        assert np.sum(masked_lines._luminosity) < np.sum(
            unmasked_lines._luminosity
        ), (
            "The masked per particle lines are not less than "
            "the unmasked per particle lines."
        )
        assert np.sum(masked_lines._continuum) < np.sum(
            unmasked_lines._continuum
        ), "The masked continuum is not less than the unmasked continuum"

    def test_masked_combination(
        self,
        test_grid,
        particle_stars_B,
        nebular_emission_model,
    ):
        """Test that the combination of two masked lines is correct."""
        # Get the lines without any masks
        unmasked_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Add a mask to the emission model
        nebular_emission_model.add_mask(
            attr="ages",
            op=">=",
            thresh=5.5 * Myr,
            set_all=True,
        )
        mask = particle_stars_B.get_mask(
            attr="ages",
            thresh=5.5 * Myr,
            op=">=",
        )
        assert np.any(mask), f"The mask is empty (defined in Myr): {mask}"
        assert False in mask, f"The mask is all True: {mask}"

        # Generate the old masked lines
        old_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Ensure there is some emission
        assert np.sum(old_lines._luminosity) > 0, (
            "The old lines has no emission."
        )

        # And now swap the mask to a young mask
        nebular_emission_model.clear_masks(True)
        nebular_emission_model.add_mask(
            attr="ages",
            op="<",
            thresh=5.5 * Myr,
            set_all=True,
        )
        assert np.any(mask), f"The mask is empty (defined in Myr): {mask}"
        assert False in mask, f"The mask is all True: {mask}"

        # Generate the yound masked lines
        young_lines = particle_stars_B.get_lines(
            test_grid.available_lines,
            nebular_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Ensure there is some emission
        assert np.sum(young_lines._luminosity) > 0, (
            "The young lines has no emission."
        )

        # Combine the two masked lines
        combined_lines = old_lines + young_lines
        combined_array = old_lines._luminosity + young_lines._luminosity

        # Ensure that the masked lines are different
        assert not np.allclose(
            young_lines._luminosity,
            old_lines._luminosity,
        ), "The old and young integrated lines are the same."
        assert not np.allclose(
            young_lines._continuum,
            old_lines._continuum,
        ), "The old and young continuum are the same."

        # Ensure the arrays are the same
        assert np.allclose(
            combined_lines._luminosity,
            combined_array,
        ), "The combined and summed arrays are not the same."
        assert np.allclose(
            combined_lines._continuum,
            old_lines._continuum + young_lines._continuum,
        ), "The combined and summed continuum are not the same."

        # Ensure the combined lines are the same as the unmasked lines
        assert np.allclose(
            combined_lines._luminosity,
            unmasked_lines._luminosity,
        ), (
            "The combined and unmasked integrated lines are not the same "
            f"(combined={np.sum(combined_lines._luminosity)} vs"
            f" unmasked={np.sum(unmasked_lines._luminosity)})."
        )
        assert np.allclose(
            combined_lines._continuum,
            unmasked_lines._continuum,
        ), (
            "The combined and unmasked continuum are not the same "
            f"(combined={np.sum(combined_lines._continuum)} vs"
            f" unmasked={np.sum(unmasked_lines._continuum)})."
        )


class TestLineCollectionGenerationSubsets:
    """Test generating subsets of lines."""

    def test_singular_int_lines(
        self,
        random_part_stars,
        nebular_emission_model,
    ):
        """Test generating a singular line."""
        # Get the subset of lines
        subset_lines = random_part_stars.get_lines(
            [
                "H 1 6562.80A",
            ],
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the subset has the correct number of lines
        assert subset_lines.nlines == 1, (
            "The subset has the wrong number of lines."
        )

        # Ensure the luminosity and continuum is an array of length 1
        assert subset_lines.luminosity.shape[-1] == 1, (
            "The luminosity is not an array of length 1."
        )
        assert subset_lines.continuum.shape[-1] == 1, (
            "The continuum is not an array of length 1."
        )

    def test_subset_int_lines(
        self,
        random_part_stars,
        nebular_emission_model,
    ):
        """Test generating a subset of lines."""
        # Get the subset of lines
        subset_lines = random_part_stars.get_lines(
            [
                "H 1 6562.80A",
                "O 3 4363.21A",
            ],
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the subset has the correct number of lines
        assert subset_lines.nlines == 2, (
            "The subset has the wrong number of lines."
        )

        # Ensure the luminosity is an array of length 2
        assert subset_lines.luminosity.shape[-1] == 2, (
            "The luminosity is not an array of length 2."
        )

        # Ensure the continuum is an array of length 2
        assert subset_lines.continuum.shape[-1] == 2, (
            "The continuum is not an array of length 2."
        )

        # Ensure the lines are 1D (i.e. intrgrated)
        assert len(subset_lines.luminosity.shape) == 1, (
            "The luminosity is not 1D."
        )
        assert len(subset_lines.continuum.shape) == 1, (
            "The continuum is not 1D."
        )

    def test_singular_part_lines(
        self,
        random_part_stars,
        nebular_emission_model,
    ):
        """Test generating a singular line."""
        nebular_emission_model.set_per_particle(True)

        # Get the subset of lines
        subset_lines = random_part_stars.get_lines(
            [
                "H 1 6562.80A",
            ],
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the subset has the correct number of lines
        assert subset_lines.nlines == 1, (
            "The subset has the wrong number of lines."
        )

        # Ensure the luminosity and continuum is an array of length 1
        assert subset_lines.luminosity.shape[-1] == 1, (
            "The luminosity is not an array of length 1."
        )
        assert subset_lines.continuum.shape[-1] == 1, (
            "The continuum is not an array of length 1."
        )

        # Ensure the lines are 2D (i.e. per particle)
        assert len(subset_lines.luminosity.shape) == 2, (
            "The luminosity is not 2D."
        )
        assert len(subset_lines.continuum.shape) == 2, (
            "The continuum is not 2D."
        )

    def test_subset_part_lines(
        self,
        random_part_stars,
        nebular_emission_model,
    ):
        """Test generating a subset of lines."""
        nebular_emission_model.set_per_particle(True)

        # Get the subset of lines
        subset_lines = random_part_stars.get_lines(
            [
                "H 1 6562.80A",
                "O 3 4363.21A",
            ],
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the subset has the correct number of lines
        assert subset_lines.nlines == 2, (
            "The subset has the wrong number of lines."
        )

        # Ensure the luminosity is an array of length 2
        assert subset_lines.luminosity.shape[-1] == 2, (
            "The luminosity is not an array of length 2."
        )

        # Ensure the continuum is an array of length 2
        assert subset_lines.continuum.shape[-1] == 2, (
            "The continuum is not an array of length 2."
        )

        # Ensure the lines are 2D (i.e. per particle)
        assert len(subset_lines.luminosity.shape) == 2, (
            "The luminosity is not 2D."
        )
        assert len(subset_lines.continuum.shape) == 2, (
            "The continuum is not 2D."
        )

    def test_subset_with_composites(
        self,
        random_part_stars,
        nebular_emission_model,
    ):
        """Test generating a subset of lines with a composite line."""
        # Get the subset of lines with a mixture of composite and normal lines
        subset_lines = random_part_stars.get_lines(
            [O2, O3, Hb, O3b, O3r],
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure the subset has the correct number of lines
        assert subset_lines.nlines == 5, (
            "The subset has the wrong number of lines."
        )

        # Ensure the luminosity is an array of length 5
        assert subset_lines.luminosity.shape[-1] == 5, (
            "The luminosity is not an array of length 5."
        )

        # Ensure the continuum is an array of length 5
        assert subset_lines.continuum.shape[-1] == 5, (
            "The continuum is not an array of length 5."
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
