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
from scipy.integrate import quad
from unyt import Msun, Myr, kpc, yr

from synthesizer import exceptions
from synthesizer.emission_models import IncidentEmission
from synthesizer.parametric import SFH, Galaxy, PointSource, ZDist
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
        grid=test_grid,
    )


@pytest.fixture
def surviving_mass_stars(test_grid):
    """Constant SFH constructed with surviving_mass."""
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
        surviving_mass=1e10 * Msun,
        grid=test_grid,
    )


@pytest.fixture
def sfzh_stars(test_grid):
    """Return a parametric Stars constructed from an explicit SFZH."""
    n_ages = len(test_grid.log10ages)
    n_metals = len(test_grid.metallicities)

    sfzh = np.ones((n_ages, n_metals))

    return Stars(
        test_grid.log10ages,
        test_grid.metallicities,
        sfzh=sfzh,
        initial_mass=1e10 * Msun,
    )


class TestSurvivingMassNormalisation:
    """Tests surviving_mass SFZHs have the correct shape."""

    def test_sfzh_normalisation_is_scalar(self, surviving_mass_stars):
        """sfzh_normalisation must be a scalar."""
        assert np.shape(surviving_mass_stars.sfzh_normalisation) == ()

    def test_sfzh_shape_preserved(
        self, constant_sfh_stars, surviving_mass_stars
    ):
        """The SFZH shape is preserved.

        The SFZH shape should be the same regardless of whether the
        initial_mass or surviving_mass is used.
        """
        mask = constant_sfh_stars.sfzh > 0
        ratios = (
            surviving_mass_stars.sfzh[mask] / constant_sfh_stars.sfzh[mask]
        )
        np.testing.assert_allclose(
            ratios, np.full_like(ratios, ratios[0]), rtol=1e-6
        )

    def test_constant_sfh_remains_constant(self, surviving_mass_stars):
        """Check the SFH remains constant.

        The constant_sfh_stars forms the same mass in each age bin, and
        the same should be true after rescaling.
        """
        sf_hist = np.sum(surviving_mass_stars.sfzh, axis=1)
        np.testing.assert_allclose(
            sf_hist, np.full_like(sf_hist, sf_hist[0]), rtol=1e-6
        )

    def test_initial_mass_matches_shape_based_normalisation(
        self, constant_sfh_stars, surviving_mass_stars, test_grid
    ):
        """Check initial masses are consistent.

        The initial_mass computed from surviving_mass must match a
        single scalar normalisation of the SFZH shape.
        """
        surviving_mass = surviving_mass_stars.surviving_mass

        # Use constant_sfh_stars' SFZH as the shape reference.
        shape_sfzh = constant_sfh_stars.sfzh
        expected_norm = surviving_mass.to("Msun").value / np.sum(
            shape_sfzh * test_grid.stellar_fraction
        )
        expected_initial_mass = expected_norm * np.sum(shape_sfzh)

        assert surviving_mass_stars.initial_mass.to(
            "Msun"
        ).value == pytest.approx(expected_initial_mass, rel=1e-6)

    def test_surviving_mass_matches_requested(self, surviving_mass_stars):
        """Check the surviving mass matches the requested value.

        Sum of sfzh * stellar_fraction should equal the requested
        surviving_mass.
        """
        recovered = np.sum(
            surviving_mass_stars.sfzh * surviving_mass_stars.stellar_fraction
        )
        assert recovered == pytest.approx(
            surviving_mass_stars.surviving_mass.to("Msun").value, rel=1e-8
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

    def test_nonzero_age_with_sfzh(self, sfzh_stars):
        """Test that an explicit SFZH works at a non-zero age."""
        result = sfzh_stars.calculate_initial_mass_at_age(100 * Myr)
        assert result >= 0 * Msun
        assert result <= sfzh_stars.initial_mass + 1e-30 * Msun

    def test_nonzero_age_with_array_sfh(self, constant_sfh_stars):
        """Test that array-based SFH and ZH work at a non-zero age."""
        result = constant_sfh_stars.calculate_initial_mass_at_age(100 * Myr)
        assert result >= 0 * Msun
        assert result <= constant_sfh_stars.initial_mass + 1e-30 * Msun

    def test_sfzh_is_not_modified(self, constant_sfh_stars):
        """Test that SFZH is unchanged by calculate_initial_mass_at_age."""
        sfzh = constant_sfh_stars.sfzh.copy()

        constant_sfh_stars.calculate_initial_mass_at_age(100 * Myr)

        np.testing.assert_array_equal(constant_sfh_stars.sfzh, sfzh)


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

    def test_nonzero_age_with_sfzh(self, sfzh_stars, test_grid):
        """Test that an explicit SFZH works at a non-zero age."""
        result = sfzh_stars.calculate_surviving_mass_at_age(
            100 * Myr, test_grid
        )
        initial = sfzh_stars.calculate_initial_mass_at_age(100 * Myr)
        assert result >= 0 * Msun
        assert result <= initial + 1e-30 * Msun

    def test_nonzero_age_with_array_sfh(self, constant_sfh_stars, test_grid):
        """Test that array-based SFH and ZH work at a non-zero age."""
        result = constant_sfh_stars.calculate_surviving_mass_at_age(
            100 * Myr, test_grid
        )
        initial = constant_sfh_stars.calculate_initial_mass_at_age(100 * Myr)
        assert result >= 0 * Msun
        assert result <= initial + 1e-30 * Msun


class TestGetAtEarlierTime:
    """Tests for getting a parametric Stars at an earlier time."""

    def test_age_offset_is_required(self, instantaneous_stars):
        """Test that omitting the age offset raises a clear TypeError."""
        with pytest.raises(TypeError):
            instantaneous_stars.get_at_earlier_time()

    def test_earlier_time_removes_young_mass(self, instantaneous_stars):
        """Test that shifting past a 10 Myr burst removes all its mass."""
        earlier = instantaneous_stars.get_at_earlier_time(20 * Myr)
        assert np.isclose(earlier.sfzh.sum(), 0.0)


class TestAddition:
    """Tests for adding parametric Stars and Galaxies."""

    @staticmethod
    def _make_stars(test_grid, **kwargs):
        """Return a parametric Stars with a constant SFH."""
        return Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=100 * Myr),
            metal_dist=0.01,
            initial_mass=1e9 * Msun,
            **kwargs,
        )

    def test_shared_attributes_survive_addition(self, test_grid):
        """Test that attributes shared by both Stars are kept."""
        morph = PointSource(offset=[0, 0] * kpc)
        stars1 = self._make_stars(test_grid, fesc=0.1, morphology=morph)
        stars2 = self._make_stars(test_grid, fesc=0.1, morphology=morph)
        combined = stars1 + stars2
        assert combined.fesc == 0.1
        assert combined.morphology is morph
        assert np.allclose(combined.sfzh, stars1.sfzh + stars2.sfzh)

    def test_differing_attributes_are_dropped(self, test_grid):
        """Test that attributes the Stars disagree on are dropped."""
        stars1 = self._make_stars(test_grid, fesc=0.1)
        stars2 = self._make_stars(test_grid, fesc=0.2)
        with pytest.warns(RuntimeWarning):
            combined = stars1 + stars2
        assert combined.fesc == self._make_stars(test_grid).fesc

    def test_galaxy_addition_with_lines(self, test_grid):
        """Test that galaxies with lines can be added together."""
        model = IncidentEmission(test_grid)
        galaxies = []
        for _ in range(2):
            gal = Galaxy(self._make_stars(test_grid))
            gal.stars.get_spectra(model)
            gal.stars.get_lines(test_grid.available_lines[:2], model)
            galaxies.append(gal)
        combined = galaxies[0] + galaxies[1]
        label = model.label
        assert np.allclose(
            combined.stars.lines[label].luminosity,
            galaxies[0].stars.lines[label].luminosity
            + galaxies[1].stars.lines[label].luminosity,
        )


class TestFunctionSFZHEdgeBins:
    """Tests that function based SFZHs populate the outermost grid bins."""

    def test_oldest_age_bin_populated(self, test_grid):
        """Test a constant SFH beyond the oldest grid age fills the last bin.

        With a unit SFR the mass in the oldest bin is its width, which runs
        from the midpoint of the last two grid ages to the oldest grid age.
        """
        ages = 10**test_grid.log10ages
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=2 * ages[-1] * yr),
            metal_dist=0.01,
        )
        expected = ages[-1] - 0.5 * (ages[-1] + ages[-2])
        assert np.isclose(stars.sf_hist[-1], expected, rtol=1e-6)

    def test_most_metal_rich_bin_populated(self, test_grid):
        """Test a ZDist peaked at the highest grid Z fills the last bin."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=ZDist.Normal(
                mean=test_grid.metallicities[-1],
                sigma=0.002,
            ),
        )
        assert stars.metal_dist[-1] > 0


class TestCalculateAverageSFR:
    """Tests for the average SFR of a parametric Stars."""

    def test_constant_sfh_over_full_range(self, test_grid):
        """Test a unit SFR covering the whole grid averages to 1 Msun/yr.

        The oldest bin must end at the oldest grid age, as it does when the
        SFZH is integrated, otherwise its mass is spread past the range.
        """
        ages = 10**test_grid.log10ages
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=2 * ages[-1] * yr),
            metal_dist=0.01,
        )
        sfr = stars.calculate_average_sfr(t_range=(0, ages[-1]))
        assert np.isclose(sfr.to("Msun/yr").value, 1.0, rtol=1e-6)


class TestGetSFZHRemap:
    """Tests for remapping a parametric SFZH onto new axes."""

    def test_remap_to_different_length_axes(self, test_grid):
        """Test remapping onto axes of a different length conserves mass."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=100 * Myr),
            metal_dist=0.01,
            initial_mass=1e9 * Msun,
        )
        new_log10ages = np.linspace(6, 10, 20)
        remapped = stars.get_sfzh(new_log10ages, test_grid.metallicities)
        assert remapped.sfzh.shape == (20, len(test_grid.metallicities))
        assert np.isclose(remapped.sfzh.sum(), stars.sfzh.sum())


class TestSFHCDF:
    """Tests for binning SFH functions with their CDFs."""

    def test_narrow_gaussian_is_resolved(self, test_grid):
        """A burst far narrower than the age axis spacing keeps its mass."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Gaussian(peak_age=3e9 * yr, sigma=1e6 * yr),
            metal_dist=0.01,
        )
        np.testing.assert_allclose(
            stars.sfzh.sum(), np.sqrt(np.pi) * 1e6, rtol=1e-6
        )

    @pytest.mark.parametrize(
        "sfh",
        [
            SFH.Constant(max_age=100 * Myr, min_age=10 * Myr),
            SFH.Gaussian(
                peak_age=1e9 * yr,
                sigma=3e8 * yr,
                max_age=3e9 * yr,
                min_age=1e8 * yr,
            ),
            SFH.Exponential(tau=1e9 * yr, max_age=5e9 * yr, min_age=2e8 * yr),
            SFH.Exponential(tau=-2e9 * yr, max_age=5e9 * yr),
            SFH.DecliningExponential(tau=12 * Myr, max_age=2.9e9 * yr),
            SFH.DelayedExponential(
                tau=1e9 * yr, max_age=5e9 * yr, min_age=1e8 * yr
            ),
            SFH.LogNormal(tau=0.5, peak_age=1e9 * yr, max_age=5e9 * yr),
            SFH.LogNormal(tau=0.1, peak_age=9.8e9 * yr, max_age=1e10 * yr),
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr, alpha=-5, beta=5, max_age=5e9 * yr
            ),
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr, alpha=0.9, beta=3, max_age=1.3e10 * yr
            ),
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr, alpha=0.5, beta=0.5, max_age=5e9 * yr
            ),
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr, alpha=0.1, beta=1e3, max_age=1.3e10 * yr
            ),
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr,
                alpha=2,
                beta=3,
                min_age=1e6 * yr,
                max_age=5e9 * yr,
            ),
            SFH.Continuity(logsfr_ratios=np.array([0.3, -0.2])),
            # tau much larger than the ages, where closed forms can cancel
            SFH.Exponential(tau=1e20 * yr, max_age=1e10 * yr),
            SFH.DelayedExponential(tau=1e15 * yr, max_age=1e10 * yr),
            # A steep SFR down to a small min_age
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr,
                alpha=2,
                beta=3,
                min_age=1 * yr,
                max_age=1e10 * yr,
            ),
            # beta close to 1
            SFH.DoublePowerLaw(
                peak_age=1e9 * yr,
                alpha=0.5,
                beta=1 + 1e-10,
                max_age=1e10 * yr,
            ),
        ],
    )
    def test_cdf_matches_quad(self, sfh):
        """Binned masses must match direct integration of the SFR."""
        edges = np.array([0, 5e6, 3e7, 2e8, 9e8, 1.5e9, 4e9, 6e9, 1e10])
        expected = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Split each bin at its breakpoints and into log spaced pieces
            # from the youngest of them, so quad can follow steep SFRs (e.g.
            # a power law down to a small min_age)
            points = [p for p in sfh._get_breakpoints() if lo < p < hi]
            start = min([p for p in [lo, *points] if p > 0], default=hi * 1e-8)
            pieces = np.unique([lo, hi, *points, *np.geomspace(start, hi, 50)])
            expected.append(
                sum(
                    quad(
                        sfh.get_sfr, a, b, limit=2000, epsabs=0, epsrel=1e-12
                    )[0]
                    for a, b in zip(pieces[:-1], pieces[1:])
                )
            )
        expected = np.array(expected)
        np.testing.assert_allclose(
            sfh.get_bin_masses(edges),
            expected,
            rtol=0,
            atol=1e-8 * np.abs(expected).max(),
        )

    def test_inverted_constant_forms_no_mass(self):
        """A Constant SFH with min_age > max_age forms no mass."""
        sfh = SFH.Constant(max_age=10 * yr, min_age=20 * yr)
        np.testing.assert_array_equal(
            sfh.get_cdf(np.array([0.0, 15.0, 30.0])), 0.0
        )

    def test_empty_combined_sfh_forms_no_mass(self):
        """A CombinedSFH without components forms no mass."""
        edges = np.array([0, 1e7, 1e9, 1e10])
        np.testing.assert_array_equal(
            SFH.CombinedSFH([]).get_bin_masses(edges), 0.0
        )

    @pytest.mark.parametrize(
        ("sf_hist", "metal_dist"),
        [
            # A Gaussian SFH peaking far beyond the oldest age
            (
                SFH.Gaussian(
                    peak_age=1e13 * yr, sigma=1e8 * yr, max_age=1e14 * yr
                ),
                0.01,
            ),
            # A Normal ZDist centred far above the highest metallicity
            (
                SFH.Constant(max_age=100 * Myr),
                ZDist.Normal(mean=0.09, sigma=0.005),
            ),
        ],
    )
    def test_no_mass_on_the_axes_raises(self, test_grid, sf_hist, metal_dist):
        """Normalising an SFZH with no mass must raise, not give NaN."""
        with pytest.raises(exceptions.InconsistentArguments):
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=sf_hist,
                metal_dist=metal_dist,
                initial_mass=1e10 * Msun,
            )

    def test_divergent_double_power_law_raises(self):
        """A DoublePowerLaw with infinite mass at age zero must raise."""
        with pytest.raises(exceptions.InconsistentArguments):
            SFH.DoublePowerLaw(peak_age=1e9 * yr, alpha=1.5, beta=3)

    def test_normal_zdist_cdf_matches_quad(self):
        """Binned metallicity weights must match direct integration."""
        zdist = ZDist.Normal(0.01, 0.005)
        edges = np.array([0, 1e-3, 5e-3, 0.01, 0.02, 0.04])
        expected = [
            quad(zdist.get_dist_weight, lo, hi, epsabs=0, epsrel=1e-12)[0]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        np.testing.assert_allclose(
            zdist.get_bin_weights(edges), expected, rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize(
        "zdist",
        [ZDist.Normal(0.01, 0.005), ZDist.Normal(1e-4, 2e-5)],
    )
    def test_numerical_zdist_cdf_matches_closed_form(self, zdist):
        """The numerical ZDist CDF must match a closed form."""
        edges = np.array([0, 1e-3, 5e-3, 0.01, 0.02, 0.04])
        expected = zdist.get_cdf(edges)
        np.testing.assert_allclose(
            ZDist.Common.get_cdf(zdist, edges),
            expected,
            rtol=0,
            atol=1e-12 * expected.max(),
        )
