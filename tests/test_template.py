"""Tests of the Template generator."""

import numpy as np
import pytest
from unyt import unyt_array, unyt_quantity

from synthesizer.emissions import Sed
from synthesizer.grid import Template


class TestTemplateInit:
    """Test suite for initialising a Template object."""

    def test_template_init(self):
        """Test Template initialization."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Initialize a Template object
        template = Template(lam, lnu)

        # Check that the wavelength and luminosity arrays are stored correctly
        assert np.allclose(template.lam.value, lam.value), (
            f"Wavelengths differ {template.lam.value} != {lam.value}"
        )
        assert template.lam.units == lam.units, (
            f"Units differ {template.lam.units} != {lam.units}"
        )

        # Check that normalization happened
        assert isinstance(template.normalisation, unyt_quantity), (
            f"Normalisation is not a unyt_quantity {template.normalisation}"
        )
        assert template.normalisation > 0, (
            f"Normalisation is not positive {template.normalisation}"
        )

    def test_template_normalization(self):
        """Test that the template luminosity is normalized correctly."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Initialize a Template object
        template = Template(lam, lnu)

        # Check that the internal SED has been normalized
        assert np.isclose(template._sed._bolometric_luminosity, 1.0), (
            "Template is not normalized "
            f"{template._sed._bolometric_luminosity}"
        )

    def test_template_unify_with_grid(self, test_grid):
        """Test unifying the template with a grid."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Initialize a Template object with the mock grid
        template = Template(lam, lnu, unify_with_grid=test_grid)

        # Check that the template wavelength matches the grid wavelength
        assert len(template.lam) == len(test_grid.lam)
        assert np.allclose(template.lam.value, test_grid.lam.value)
        assert template.lam.units == test_grid.lam.units

    def test_template_scaling_missing_units_error(self):
        """Test that an error is raised when the scaling has no units."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Initialize a Template object
        template = Template(lam, lnu)

        # Test with a bolometric luminosity that has no units
        with pytest.raises(Exception) as excinfo:
            # This should raise an exception because 1e45 has no units
            template.get_spectra(1e45)

        # Check that the error message mentions units
        assert "units" in str(excinfo.value).lower()

    def test_template_init_missing_units_error(self):
        """Test that an error is raised when the input has no units."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Test with a bolometric luminosity that has no units
        with pytest.raises(Exception) as excinfo:
            # This should raise an exception because the input has no units
            Template(lam, lnu.value)

        # Check that the error message mentions units
        assert "units" in str(excinfo.value).lower()


class TestTemplateGeneration:
    """Test suite for generating SEDs from templates."""

    def test_template_get_spectra(self):
        """Test the get_spectra method."""
        # Create sample data
        lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
        lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")

        # Initialize a Template object
        template = Template(lam, lnu)

        # Test with a specific bolometric luminosity
        bol_lum = unyt_quantity(1.0e45, "erg/s")
        scaled_sed = template.get_spectra(bol_lum)

        # Check that the returned object is an Sed
        assert isinstance(scaled_sed, Sed)

        # Check that scaling was applied correctly
        assert np.isclose(scaled_sed._bolometric_luminosity, bol_lum.value)

    def test_bh_template_model(
        self,
        particle_black_hole,
        template_emission_model_bh,
    ):
        """Test the template model with a black hole particle."""
        sed = particle_black_hole.get_spectra(template_emission_model_bh)

        # Check that the returned object is an Sed
        assert isinstance(sed, Sed)

        # Ensure the Sed has the correct shape (i.e. ndim = 2 and (nbh, nlam))
        assert sed.shape[0] == particle_black_hole.nbh, (
            f"Sed has the wrong shape {sed.shape}"
        )
        assert sed.shape[1] == len(template_emission_model_bh.lam), (
            f"Sed has the wrong shape {sed.shape}"
        )

        # Check that the bolometric luminosity is correct
        assert np.all(
            np.isclose(
                sed.bolometric_luminosity,
                particle_black_hole.bolometric_luminosity,
            )
        ), (
            "Bolometric luminosity differs "
            f"{sed.bolometric_luminosity} != "
            f"{particle_black_hole.bolometric_luminosity}"
        )

    def test_single_array_bh_template_model(
        self,
        single_particle_black_hole,
        template_emission_model_bh,
    ):
        """Test the template model with a single black hole."""
        sed = single_particle_black_hole.get_spectra(
            template_emission_model_bh
        )

        # Check that the returned object is an Sed
        assert isinstance(sed, Sed)

        # Ensure the Sed has the correct shape (i.e. ndim = 2 and (nbh, nlam))
        assert sed.shape[0] == single_particle_black_hole.nbh, (
            f"Sed has the wrong shape {sed.shape}"
        )
        assert sed.shape[1] == len(template_emission_model_bh.lam), (
            f"Sed has the wrong shape {sed.shape}"
        )

        # Check that the bolometric luminosity is correct
        assert np.isclose(
            sed.bolometric_luminosity,
            single_particle_black_hole.bolometric_luminosity,
        ), (
            "Bolometric luminosity differs "
            f"{sed.bolometric_luminosity} != "
            f"{single_particle_black_hole.bolometric_luminosity}"
        )

    def test_single_scalar_bh_template_model(
        self,
        single_particle_black_hole_scalars,
        template_emission_model_bh,
    ):
        """Test the template model with a single black hole."""
        sed = single_particle_black_hole_scalars.get_spectra(
            template_emission_model_bh
        )

        # Check that the returned object is an Sed
        assert isinstance(sed, Sed)

        # Ensure the Sed has the correct shape (i.e. ndim = 2 and (nbh, nlam))
        assert sed.shape[0] == single_particle_black_hole_scalars.nbh, (
            f"Sed has the wrong shape {sed.shape}"
        )
        assert sed.shape[1] == len(template_emission_model_bh.lam), (
            f"Sed has the wrong shape {sed.shape}"
        )

        # Check that the bolometric luminosity is correct
        assert np.isclose(
            sed.bolometric_luminosity,
            single_particle_black_hole_scalars.bolometric_luminosity,
        ), (
            "Bolometric luminosity differs "
            f"{sed.bolometric_luminosity} != "
            f"{single_particle_black_hole_scalars.bolometric_luminosity}"
        )
