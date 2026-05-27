"""A test suite for parametric Stars methods introduced in PR #1153.

Tests cover:
- calculate_surviving_sfzh
- calculate_surviving_sfh
- calculate_surviving_mass
- calculate_initial_mass_at_age
- calculate_surviving_mass_at_age
"""

import numpy as np
import pytest
from unyt import Msun, Myr, yr

from synthesizer.parametric.stars import Stars
from synthesizer.units import Units


@pytest.fixture
def instantaneous_stars(test_grid):
    """Return a parametric Stars with an instantaneous burst at 10 Myr."""
    return Stars(
        test_grid.log10ages,
        test_grid.metallicities,
        sf_hist=1e7 * yr,
        metal_dist=0.01,
        initial_mass=1e10 * Msun,
    )


@pytest.fixture
def constant_sfh_stars(test_grid):
    """Return a parametric Stars with a uniform SFH across all age bins."""
    n_ages = len(test_grid.log10ages)
    n_metals = len(test_grid.metallicities)
    # Uniform SFH distributed equally across age bins
    sf_hist = np.ones(n_ages)
    sf_hist = sf_hist / sf_hist.sum()  # normalise
    metal_dist = np.ones(n_metals) / n_metals
    return Stars(
        test_grid.log10ages,
        test_grid.metallicities,
        sf_hist=sf_hist,
        metal_dist=metal_dist,
        initial_mass=1e10 * Msun,
    )


class TestCalculateSurvivingSFZH:
    """Tests for Stars.calculate_surviving_sfzh."""

    def test_returns_array(self, instantaneous_stars, test_grid):
        """Test that calculate_surviving_sfzh returns a numpy array."""
        result = instantaneous_stars.calculate_surviving_sfzh(test_grid)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_sfzh(self, instantaneous_stars, test_grid):
        """Test that the surviving SFZH has the same shape as sfzh."""
        result = instantaneous_stars.calculate_surviving_sfzh(test_grid)
        assert result.shape == instantaneous_stars.sfzh.shape

    def test_values_le_sfzh(self, instantaneous_stars, test_grid):
        """Test that surviving SFZH <= the SFZH values."""
        result = instantaneous_stars.calculate_surviving_sfzh(test_grid)
        assert np.all(result <= instantaneous_stars.sfzh + 1e-30)

    def test_values_non_negative(self, instantaneous_stars, test_grid):
        """Test that surviving SFZH values are non-negative."""
        result = instantaneous_stars.calculate_surviving_sfzh(test_grid)
        assert np.all(result >= 0)

    def test_sum_matches_surviving_mass(self, instantaneous_stars, test_grid):
        """Test that the sum of surviving SFZH equals the surviving mass."""
        surviving_sfzh = instantaneous_stars.calculate_surviving_sfzh(
            test_grid
        )
        surviving_mass = instantaneous_stars.calculate_surviving_mass(
            test_grid
        )
        assert np.isclose(
            np.sum(surviving_sfzh) * Msun, surviving_mass, rtol=1e-10
        )

    def test_uniform_stellar_fraction_scales_correctly(
        self, constant_sfh_stars, test_grid
    ):
        """Test that surviving SFZH is sfzh * stellar_fraction."""
        result = constant_sfh_stars.calculate_surviving_sfzh(test_grid)
        expected = constant_sfh_stars.sfzh * test_grid.stellar_fraction
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestCalculateSurvivingSFH:
    """Tests for Stars.calculate_surviving_sfh."""

    def test_returns_array(self, instantaneous_stars, test_grid):
        """Test that calculate_surviving_sfh returns a numpy array."""
        result = instantaneous_stars.calculate_surviving_sfh(test_grid)
        assert isinstance(result, np.ndarray)

    def test_shape_is_1d_with_n_ages(self, instantaneous_stars, test_grid):
        """Test that surviving SFH is 1D with length = number of age bins."""
        result = instantaneous_stars.calculate_surviving_sfh(test_grid)
        assert result.ndim == 1
        assert len(result) == len(test_grid.log10ages)

    def test_values_non_negative(self, instantaneous_stars, test_grid):
        """Test that surviving SFH values are non-negative."""
        result = instantaneous_stars.calculate_surviving_sfh(test_grid)
        assert np.all(result >= 0)

    def test_sum_matches_surviving_sfzh_sum(
        self, constant_sfh_stars, test_grid
    ):
        """Test that the sum of surviving SFH = the sum of surviving SFZH."""
        surviving_sfh = constant_sfh_stars.calculate_surviving_sfh(test_grid)
        surviving_sfzh = constant_sfh_stars.calculate_surviving_sfzh(test_grid)
        assert np.isclose(
            np.sum(surviving_sfh), np.sum(surviving_sfzh), rtol=1e-10
        )

    def test_is_metallicity_marginalisation_of_sfzh(
        self, constant_sfh_stars, test_grid
    ):
        """Test that surviving SFH is the sum of surviving SFZH."""
        surviving_sfh = constant_sfh_stars.calculate_surviving_sfh(test_grid)
        surviving_sfzh = constant_sfh_stars.calculate_surviving_sfzh(test_grid)
        expected = np.sum(surviving_sfzh, axis=1)
        np.testing.assert_allclose(surviving_sfh, expected, rtol=1e-10)


class TestCalculateSurvivingMass:
    """Tests for Stars.calculate_surviving_mass."""

    def test_returns_unyt_quantity(self, instantaneous_stars, test_grid):
        """Test that calculate_surviving_mass returns a unyt quantity."""
        from unyt import unyt_quantity

        result = instantaneous_stars.calculate_surviving_mass(test_grid)
        assert isinstance(result, unyt_quantity)

    def test_units_are_solar_masses(self, instantaneous_stars, test_grid):
        """Test that the returned quantity has solar mass units."""
        result = instantaneous_stars.calculate_surviving_mass(test_grid)
        # Should be convertible to Msun without error
        result_msun = result.to("Msun")
        assert result_msun.units == Units().mass

    def test_surviving_mass_le_initial_mass(
        self,
        instantaneous_stars,
        test_grid,
    ):
        """Test that surviving mass is <= to the initial mass."""
        surviving = instantaneous_stars.calculate_surviving_mass(test_grid)
        initial = instantaneous_stars.initial_mass
        assert surviving <= initial + 1e-30 * Msun

    def test_surviving_mass_positive(self, instantaneous_stars, test_grid):
        """Test that surviving mass is positive."""
        result = instantaneous_stars.calculate_surviving_mass(test_grid)
        assert result > 0 * Msun

    def test_uses_surviving_sfzh(self, constant_sfh_stars, test_grid):
        """Test that surviving mass equals sum of surviving SFZH * Msun."""
        surviving_mass = constant_sfh_stars.calculate_surviving_mass(test_grid)
        surviving_sfzh = constant_sfh_stars.calculate_surviving_sfzh(test_grid)
        expected = np.sum(surviving_sfzh) * Msun
        assert np.isclose(surviving_mass, expected, rtol=1e-10)


class TestCalculateInitialMassAtAge:
    """Tests for Stars.calculate_initial_mass_at_age."""

    def test_returns_unyt_quantity(self, instantaneous_stars):
        """Test that calculate_initial_mass_at_age returns a unyt quantity."""
        from unyt import unyt_quantity

        result = instantaneous_stars.calculate_initial_mass_at_age(50 * Myr)
        assert isinstance(result, unyt_quantity)

    def test_units_are_solar_masses(self, instantaneous_stars):
        """Test that the returned quantity has solar mass units."""
        result = instantaneous_stars.calculate_initial_mass_at_age(50 * Myr)
        result_msun = result.to("Msun")
        assert result_msun.units == Units().mass

    def test_age_less_than_burst_returns_initial_mass(
        self, instantaneous_stars
    ):
        """Test that querying before the burst age returns the initial mass.

        The instantaneous_stars fixture has a burst at 10 Myr (1e7 yr).
        Querying at age=5 Myr (older than 5 Myr in lookback time) should
        include the 10 Myr burst, returning the full initial mass.
        """
        result = instantaneous_stars.calculate_initial_mass_at_age(5 * Myr)
        initial = instantaneous_stars.initial_mass
        # The burst at 10 Myr is older than 5 Myr, so it should be included
        assert np.isclose(
            result.to("Msun").value, initial.to("Msun").value, rtol=0.01
        )

    def test_age_greater_than_burst_returns_zero(self, instantaneous_stars):
        """Test that querying after the burst age returns near-zero mass.

        The instantaneous_stars fixture has a burst at 10 Myr.
        Querying at age=50 Myr should exclude the 10 Myr burst (which is
        more recent than 50 Myr lookback time), returning ~0.
        """
        result = instantaneous_stars.calculate_initial_mass_at_age(50 * Myr)
        assert result.to("Msun").value == pytest.approx(0.0, abs=1.0)

    def test_very_small_age_returns_initial_mass(self, constant_sfh_stars):
        """Test that querying at a very small age returns the initial mass.

        With a tiny lookback time, all stellar populations are older than the
        query age, so the returned mass should equal the total initial mass.
        """
        # Use a very small age (smaller than the smallest age bin lower edge)
        result = constant_sfh_stars.calculate_initial_mass_at_age(1e4 * yr)
        initial = constant_sfh_stars.initial_mass
        assert np.isclose(
            result.to("Msun").value, initial.to("Msun").value, rtol=0.01
        )

    def test_very_large_age_returns_zero(self, constant_sfh_stars):
        """Test that querying at a very large age returns ~0 mass.

        With a lookback time larger than the oldest age bin, no stellar
        populations are older than the query age, so the result should be ~0.
        """
        result = constant_sfh_stars.calculate_initial_mass_at_age(1e12 * yr)
        assert result.to("Msun").value == pytest.approx(0.0, abs=1.0)

    def test_mass_decreases_with_increasing_age(self, constant_sfh_stars):
        """Test that returned mass is monotonically non-increasing with age.

        As the lookback age increases, fewer and fewer stellar populations
        are older than the query age, so the returned mass should decrease.
        """
        ages = [1 * Myr, 10 * Myr, 100 * Myr, 1000 * Myr]
        masses = [
            constant_sfh_stars.calculate_initial_mass_at_age(a)
            .to("Msun")
            .value
            for a in ages
        ]
        for i in range(len(masses) - 1):
            assert masses[i] >= masses[i + 1] - 1e-10

    def test_accepts_float_in_years(self, instantaneous_stars):
        """Test that calculate_initial_mass_at_age accepts float in years."""
        # @accepts(age=yr) should allow passing a float treated as years
        result = instantaneous_stars.calculate_initial_mass_at_age(5e6 * yr)
        assert result > 0 * Msun

    def test_result_bounded_by_initial_mass(self, constant_sfh_stars):
        """Test that the result is always <= initial_mass."""
        for age in [1 * Myr, 10 * Myr, 100 * Myr, 500 * Myr]:
            result = constant_sfh_stars.calculate_initial_mass_at_age(age)
            assert result <= constant_sfh_stars.initial_mass + 1e-30 * Msun


class TestCalculateSurvivingMassAtAge:
    """Tests for Stars.calculate_surviving_mass_at_age."""

    def test_returns_unyt_quantity(self, instantaneous_stars, test_grid):
        """Test that calculate_surviving_mass_at_age returns unyt quantity."""
        from unyt import unyt_quantity

        result = instantaneous_stars.calculate_surviving_mass_at_age(
            50 * Myr, test_grid
        )
        assert isinstance(result, unyt_quantity)

    def test_units_are_solar_masses(self, instantaneous_stars, test_grid):
        """Test that the returned quantity has solar mass units."""
        result = instantaneous_stars.calculate_surviving_mass_at_age(
            50 * Myr, test_grid
        )
        result_msun = result.to("Msun")
        assert result_msun.units == Units().mass

    def test_surviving_le_initial_at_same_age(
        self, constant_sfh_stars, test_grid
    ):
        """Test that surviving mass at age <= initial mass at the same age."""
        age = 10 * Myr
        surviving = constant_sfh_stars.calculate_surviving_mass_at_age(
            age, test_grid
        )
        initial = constant_sfh_stars.calculate_initial_mass_at_age(age)
        assert surviving <= initial + 1e-30 * Msun

    def test_very_small_age_returns_surviving_mass(
        self, constant_sfh_stars, test_grid
    ):
        """Test at small lookback age, result approaches total surviving mass.

        With a tiny lookback time, all populations are included, so the
        result should approach calculate_surviving_mass(grid).
        """
        result = constant_sfh_stars.calculate_surviving_mass_at_age(
            1e4 * yr, test_grid
        )
        total_surviving = constant_sfh_stars.calculate_surviving_mass(
            test_grid
        )
        assert np.isclose(
            result.to("Msun").value,
            total_surviving.to("Msun").value,
            rtol=0.01,
        )

    def test_very_large_age_returns_zero(self, constant_sfh_stars, test_grid):
        """Test that at a very large lookback age, the result is ~0."""
        result = constant_sfh_stars.calculate_surviving_mass_at_age(
            1e12 * yr, test_grid
        )
        assert result.to("Msun").value == pytest.approx(0.0, abs=1.0)

    def test_non_negative(self, constant_sfh_stars, test_grid):
        """Test that surviving mass at age is non-negative."""
        for age in [1 * Myr, 10 * Myr, 100 * Myr]:
            result = constant_sfh_stars.calculate_surviving_mass_at_age(
                age, test_grid
            )
            assert result >= 0 * Msun

    def test_mass_decreases_with_increasing_age(
        self, constant_sfh_stars, test_grid
    ):
        """Test surviving mass is monotonically non-increasing with age."""
        ages = [1 * Myr, 10 * Myr, 100 * Myr, 1000 * Myr]
        masses = [
            constant_sfh_stars.calculate_surviving_mass_at_age(a, test_grid)
            .to("Msun")
            .value
            for a in ages
        ]
        for i in range(len(masses) - 1):
            assert masses[i] >= masses[i + 1] - 1e-10

    def test_age_greater_than_burst_returns_zero(
        self, instantaneous_stars, test_grid
    ):
        """Test querying after the burst age returns near-zero surviving mass.

        The instantaneous_stars fixture has a burst at 10 Myr.
        Querying at 50 Myr should give ~0 since the burst is more recent.
        """
        result = instantaneous_stars.calculate_surviving_mass_at_age(
            50 * Myr, test_grid
        )
        assert result.to("Msun").value == pytest.approx(0.0, abs=1.0)

    def test_age_less_than_burst_returns_positive(
        self, instantaneous_stars, test_grid
    ):
        """Test querying before the burst age returns positive surviving mass.

        The instantaneous_stars fixture has a burst at 10 Myr.
        Querying at 5 Myr should give positive surviving mass since the 10 Myr
        burst is older than 5 Myr lookback time.
        """
        result = instantaneous_stars.calculate_surviving_mass_at_age(
            5 * Myr, test_grid
        )
        assert result.to("Msun").value > 0
