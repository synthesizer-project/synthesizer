"""A test suite for particle based Stars."""

import numpy as np
import pytest
from unyt import Myr, kpc

from synthesizer import exceptions
from synthesizer.exceptions import InconsistentAddition
from synthesizer.parametric.stars import Stars as ParaStars
from synthesizer.particle.stars import Stars


def test_cant_add_different_types(particle_stars_A, particle_gas_A):
    """Test we can't add different types of particles together."""
    with pytest.raises(InconsistentAddition):
        particle_stars_A + particle_gas_A


def test_add_stars(particle_stars_A, particle_stars_B):
    """Test we can add two Stars objects together."""
    assert isinstance(particle_stars_A + particle_stars_B, Stars)


def test_cant_add_stars_different_redshifts(
    particle_stars_A, particle_stars_B
):
    """Test we can't add two Stars objects with different redshifts."""
    particle_stars_B.redshift = 2.0

    with pytest.raises(InconsistentAddition):
        particle_stars_A + particle_stars_B


def test_add_stars_with_different_attributes(
    particle_stars_A, particle_stars_B
):
    """Test we can add two Stars objects with different attributes."""
    particle_stars_B.dummy_attr = None

    assert isinstance(particle_stars_A + particle_stars_B, Stars)


def test_parametric_young_stars(particle_stars_A, test_grid):
    """Test parametric_young_stars replacing young star particles."""
    particle_stars_A.parametric_young_stars(
        age=10 * Myr,
        parametric_sfh="constant",
        grid=test_grid,
    )

    assert isinstance(particle_stars_A, Stars)
    assert isinstance(particle_stars_A._parametric_young_stars, ParaStars)
    assert isinstance(particle_stars_A._old_stars, Stars)
    assert isinstance(particle_stars_A.young_stars_parametrisation, dict)
    assert particle_stars_A.young_stars_parametrisation["age"] == 10 * Myr
    assert (
        particle_stars_A.young_stars_parametrisation["parametrisation"]
        == "constant"
    )


def test_calculate_radii(particle_stars_A):
    """Test that we can calculate the radii of stars."""
    with pytest.raises(
        exceptions.InconsistentArguments,
        match="Can't calculate radii without a centre.",
    ):
        particle_stars_A.get_radii()

    particle_stars_A.centre = np.array([1.0, 0.0, 2.0]) * kpc

    assert isinstance(particle_stars_A.get_radii(), np.ndarray)
    assert (particle_stars_A.radii <= 6 * kpc).all()


class TestWeightedAttributes:
    """Tests for checking the weighted attributes work correctly."""

    def test_mass_weighted_age(self, unit_mass_stars):
        """Test that mass weighted age is calculated correctly."""
        result = unit_mass_stars.get_mass_weighted_age()
        expected = 2.0 * Myr
        assert np.isclose(result, expected), f"Age: {result} != {expected}"

    def test_mass_weighted_metallicity(self, unit_mass_stars):
        """Test that mass weighted metallicity is calculated correctly."""
        result = unit_mass_stars.get_mass_weighted_metallicity()
        expected = 0.02
        assert np.isclose(result, expected), f"Z: {result} != {expected}"

    def test_mass_weighted_optical_depth(self, unit_mass_stars):
        """Test that mass weighted optical depth is calculated correctly."""
        result = unit_mass_stars.get_mass_weighted_optical_depth()
        expected = 0.2
        assert np.isclose(result, expected), f"tau_V: {result} != {expected}"

    def test_weighted_attr_coordinates_initial_masses(self, unit_mass_stars):
        """Test weighted coordinates using 'initial_masses'."""
        result = unit_mass_stars.get_weighted_attr(
            "coordinates",
            "initial_masses",
            axis=0,
        )
        expected = np.array([1.0, 1.0, 1.0]) * kpc
        assert np.allclose(result, expected)

    def test_weighted_attr_coordinates_current_masses(self, unit_mass_stars):
        """Test weighted coordinates using current_masses."""
        result = unit_mass_stars.get_weighted_attr(
            "coordinates",
            unit_mass_stars.current_masses,
            axis=0,
        )
        expected = np.array([1.0, 1.0, 1.0]) * kpc
        assert np.allclose(result, expected)

    def test_weighted_attr_coordinates_invalid_axis(self, unit_mass_stars):
        """Test that using an invalid axis raises an exception."""
        with pytest.raises(
            TypeError,
            match="Axis must be specified when "
            "shapes of a and weights differ.",
        ):
            unit_mass_stars.get_weighted_attr(
                "coordinates",
                "initial_masses",
            )
        with pytest.raises(
            np.exceptions.AxisError,
            match="axis: axis 3 is out of bounds for array of dimension 2",
        ):
            unit_mass_stars.get_weighted_attr(
                "coordinates",
                "initial_masses",
                axis=3,
            )

    def test_invalid_attr(self, unit_mass_stars):
        """Test that using a non-existent attribute raises an exception."""
        with pytest.raises(
            AttributeError,
            match="'Stars' object has no attribute 'not_an_attribute'",
        ):
            unit_mass_stars.get_weighted_attr(
                "not_an_attribute", "initial_masses"
            )

    def test_invalid_mass_weight_attr(self, unit_mass_stars):
        """Test that using a non-existent weight raises an exception."""
        with pytest.raises(
            AttributeError,
            match="'Stars' object has no attribute 'not_an_attribute'",
        ):
            unit_mass_stars.get_weighted_attr(
                "coordinates", "not_an_attribute"
            )

    def test_lum_weighted_age(self, unit_emission_stars):
        """Test that luminosity weighted age is calculated correctly."""
        result = unit_emission_stars.get_lum_weighted_age("FAKE", "fake")
        expected = 2.0 * Myr
        assert np.isclose(result, expected), f"Age: {result} != {expected}"

    def test_flux_weighted_age(self, unit_emission_stars):
        """Test that flux weighted age is calculated correctly."""
        result = unit_emission_stars.get_flux_weighted_age("FAKE", "fake")
        expected = 2.0 * Myr
        assert np.isclose(result, expected), f"Age: {result} != {expected}"

    def test_lum_weighted_metallicity(self, unit_emission_stars):
        """Test that luminosity weighted metallicity is  correct."""
        result = unit_emission_stars.get_lum_weighted_metallicity(
            "FAKE", "fake"
        )
        expected = 0.02
        assert np.isclose(result, expected), f"Z: {result} != {expected}"

    def test_flux_weighted_metallicity(self, unit_emission_stars):
        """Test that flux weighted metallicity is calculated correctly."""
        result = unit_emission_stars.get_flux_weighted_metallicity(
            "FAKE", "fake"
        )
        expected = 0.02
        assert np.isclose(result, expected), f"Z: {result} != {expected}"

    def test_lum_weighted_optical_depth(self, unit_emission_stars):
        """Test that luminosity weighted optical depth is  correct."""
        result = unit_emission_stars.get_lum_weighted_optical_depth(
            "FAKE", "fake"
        )
        expected = 0.2
        assert np.isclose(result, expected), f"tau_V: {result} != {expected}"

    def test_flux_weighted_optical_depth(self, unit_emission_stars):
        """Test that flux weighted optical depth is calculated correctly."""
        result = unit_emission_stars.get_flux_weighted_optical_depth(
            "FAKE", "fake"
        )
        expected = 0.2
        assert np.isclose(result, expected), f"tau_V: {result} != {expected}"

    def test_weighted_attr_coordinates_luminosity(self, unit_emission_stars):
        """Test weighted coordinates using luminosity."""
        result = unit_emission_stars.get_lum_weighted_attr(
            "coordinates",
            "FAKE",
            "fake",
            axis=0,
        )
        expected = np.array([1.0, 1.0, 1.0]) * kpc
        assert np.allclose(result, expected)

    def test_weighted_attr_coordinates_flux(self, unit_emission_stars):
        """Test weighted coordinates using flux."""
        result = unit_emission_stars.get_flux_weighted_attr(
            "coordinates",
            "FAKE",
            "fake",
            axis=0,
        )
        expected = np.array([1.0, 1.0, 1.0]) * kpc
        assert np.allclose(result, expected)

    def test_invalid_attr_lum_weighted(self, unit_emission_stars):
        """Test that using a non-existent attribute raises an exception."""
        with pytest.raises(
            AttributeError,
            match="'Stars' object has no attribute 'not_an_attribute'",
        ):
            unit_emission_stars.get_lum_weighted_attr(
                "not_an_attribute",
                "FAKE",
                "fake",
            )

    def test_invalid_attr_flux_weighted(self, unit_emission_stars):
        """Test that using a non-existent attribute raises an exception."""
        with pytest.raises(
            AttributeError,
            match="'Stars' object has no attribute 'not_an_attribute'",
        ):
            unit_emission_stars.get_flux_weighted_attr(
                "not_an_attribute",
                "FAKE",
                "fake",
            )

    def test_invalid_spec_key(self, unit_emission_stars):
        """Test that using a non-existent key raises an exception."""
        with pytest.raises(
            KeyError,
            match="'not_a_key'",
        ):
            unit_emission_stars.get_lum_weighted_attr(
                "ages",
                "not_a_key",
                "fake",
            )

    def test_invalid_phot_key(self, unit_emission_stars):
        """Test that using a non-existent key raises an exception."""
        with pytest.raises(
            KeyError,
            match="'Filter code not_a_key not found"
            " in photometry collection.'",
        ):
            unit_emission_stars.get_lum_weighted_attr(
                "ages",
                "FAKE",
                "not_a_key",
            )
