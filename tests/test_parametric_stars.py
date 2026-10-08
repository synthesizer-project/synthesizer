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
from unyt import Msun, Myr, angstrom, dimensionless, kpc, yr

import synthesizer.parametric.stars as stars_module
from synthesizer import exceptions
from synthesizer.emission_models import (
    BimodalPacmanEmission,
    IncidentEmission,
    PacmanEmission,
)
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import (
    SFH,
    Annuli,
    Galaxy,
    PerPopulation,
    PointSource,
    Sersic2D,
    ZDist,
)
from synthesizer.parametric import Galaxy as ParametricGalaxy
from synthesizer.parametric.bin_mask import rebin_axis, union_edges
from synthesizer.parametric.stars import Stars
from synthesizer.particle import Stars as ParticleStars
from synthesizer.pipeline import Pipeline
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

    return Stars.from_sfzh(
        test_grid.log10ages,
        test_grid.metallicities,
        sfzh,
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


class TestFunctionSFZHEdgeBins:
    """Tests that function based SFZHs populate the outermost grid bins."""

    def test_oldest_age_bin_populated(self, test_grid):
        """Test a constant SFH beyond the oldest grid age fills the last bin.

        With a unit SFR the mass formed is the oldest grid age (the SFH is
        binned up to the oldest grid age) and the oldest point receives some
        of it.
        """
        ages = 10**test_grid.log10ages
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=2 * ages[-1] * yr),
            metal_dist=0.01,
        )
        assert stars.sf_hist[-1] > 0
        assert np.isclose(stars.sf_hist.sum(), ages[-1], rtol=1e-10)

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

    def test_differing_attributes_become_per_population(self, test_grid):
        """Escape fractions the Stars disagree on become one per population."""
        stars1 = self._make_stars(test_grid, fesc=0.1)
        stars2 = self._make_stars(test_grid, fesc=0.2)
        combined = stars1 + stars2
        np.testing.assert_allclose(combined.fesc, [0.1, 0.2])

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


def _set_age_bin(stars, lo, hi, mass, metallicity_index=0):
    """Give a Stars a single finite age bin at one metallicity point."""
    metals = np.asarray(stars.metallicities)
    masses = np.zeros((1, 1, 1))
    masses[0, 0, 0] = mass
    stars._set_bins(
        {
            "ages": np.array([lo, hi]),
            "metallicities": np.repeat(metals[metallicity_index], 2),
        },
        masses,
    )


class TestBinStorage:
    """Tests for the binned representation of a parametric Stars."""

    def test_sfzh_view_matches_input(self, test_grid):
        """An SFZH on the axes must be returned unchanged by the view."""
        sfzh = np.random.default_rng(0).random(
            (test_grid.log10ages.size, test_grid.metallicities.size)
        )
        stars = Stars.from_sfzh(
            test_grid.log10ages, test_grid.metallicities, sfzh
        )
        np.testing.assert_allclose(stars.sfzh, sfzh, rtol=1e-12)

    def test_passing_sfzh_is_deprecated(self, test_grid):
        """Passing sfzh warns and matches from_sfzh."""
        sfzh = np.ones(
            (test_grid.log10ages.size, test_grid.metallicities.size)
        )
        with pytest.warns(FutureWarning, match="from_sfzh"):
            stars = Stars(
                test_grid.log10ages, test_grid.metallicities, sfzh=sfzh
            )
        expected = Stars.from_sfzh(
            test_grid.log10ages, test_grid.metallicities, sfzh
        )
        np.testing.assert_array_equal(stars.bin_masses, expected.bin_masses)

    def test_sfzh_view_is_read_only(self, instantaneous_stars):
        """The SFZH view can't be modified in place."""
        with pytest.raises(ValueError):
            instantaneous_stars.sfzh[0, 0] = 1.0

    def test_input_sfzh_is_not_modified(self, test_grid):
        """Normalising to an initial mass must not touch the input array."""
        sfzh = np.ones(
            (test_grid.log10ages.size, test_grid.metallicities.size)
        )
        Stars.from_sfzh(
            test_grid.log10ages,
            test_grid.metallicities,
            sfzh,
            initial_mass=1e9 * Msun,
        )
        assert np.all(sfzh == 1.0)

    def test_setting_bins_clears_derived_state(self, test_grid):
        """New bins must invalidate the SFZH view and stored weights."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=0.01,
            initial_mass=1e9 * Msun,
        )
        model = IncidentEmission(test_grid)
        lnu = stars.get_spectra(model).lnu.copy()
        stars._set_bins(stars._bin_edges, stars.bin_masses * 2)
        np.testing.assert_allclose(stars.sfzh.sum(), 2e9)
        np.testing.assert_allclose(stars.get_spectra(model).lnu, 2 * lnu)


class TestFractionOutsideGrid:
    """Tests for the fraction of the mass outside a grid's axes."""

    @staticmethod
    def _stars(test_grid):
        """Return a Stars on the grid's axes."""
        return Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=0.01,
            initial_mass=1 * Msun,
        )

    def test_inside(self, test_grid):
        """A bin inside the grid has no mass outside."""
        stars = self._stars(test_grid)
        age = 10 ** test_grid.log10ages[3]
        _set_age_bin(stars, age, 2 * age, 1.0)
        assert stars.get_fraction_outside_grid(test_grid) == 0.0

    def test_straddling_bin(self, test_grid):
        """A bin straddling the youngest grid age is partly outside."""
        stars = self._stars(test_grid)
        _set_age_bin(stars, 0.0, 2 * 10 ** test_grid.log10ages[0], 1.0)
        assert np.isclose(stars.get_fraction_outside_grid(test_grid), 0.5)

    def test_point_outside_metallicities(self, test_grid):
        """A metallicity below the grid puts all of a bin outside."""
        age = 10 ** test_grid.log10ages[3]
        stars = Stars.from_binned(
            test_grid.log10ages,
            test_grid.metallicities,
            [age, 2 * age],
            [0.0, 0.0],
            np.ones((1, 1)),
        )
        assert stars.get_fraction_outside_grid(test_grid) == 1.0


class TestBinMask:
    """Tests for fractional masks on binned parametric populations."""

    @pytest.fixture
    def stars(self, test_grid):
        """Return a Stars with a single 5-20 Myr bin."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=0.01,
        )
        _set_age_bin(stars, 5e6, 2e7, 1.0, metallicity_index=3)
        return stars

    def test_straddling_bin_fraction(self, stars):
        """A threshold through a bin passes the fraction below it."""
        mask = stars.get_mask("log10ages", 7, "<")
        np.testing.assert_allclose(mask.get_fractions(stars), 5.0 / 15.0)
        mask = stars.get_mask("ages", 10 * Myr, ">=")
        np.testing.assert_allclose(mask.get_fractions(stars), 10.0 / 15.0)

    def test_same_axis_conditions_intersect(self, stars):
        """Two conditions on one axis keep the intersection of intervals."""
        mask = stars.get_mask("ages", 8 * Myr, ">")
        mask = stars.get_mask("ages", 12 * Myr, "<", mask=mask)
        np.testing.assert_allclose(mask.get_fractions(stars), 4.0 / 15.0)

    def test_conditions_on_different_axes_multiply(self, test_grid):
        """Conditions on different axes multiply their fractions."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=0.01,
        )
        stars._set_bins(
            {
                "ages": np.array([5e6, 2e7]),
                "metallicities": np.array([0.001, 0.005]),
            },
            np.ones((1, 1, 1)),
        )
        mask = stars.get_mask("log10ages", 7, "<")
        mask = stars.get_mask("metallicities", 0.002, "<", mask=mask)
        np.testing.assert_allclose(
            mask.get_fractions(stars), (5.0 / 15.0) * (1.0 / 4.0)
        )

    def test_points_are_exact(self, instantaneous_stars):
        """Zero width bins are evaluated exactly, including equality."""
        stars = instantaneous_stars
        mask = stars.get_mask("log10ages", 7, "<")
        fracs = mask.get_fractions(stars)[0, :, 0]

        # The point bins (even indices) pass exactly when below the
        # threshold, the empty bins between them get the clipped fraction
        edges = stars.get_bin_edges("ages", values_only=True)
        lo, hi = edges[:-1], edges[1:]
        expected = np.where(
            lo == hi,
            (lo < 1e7).astype(float),
            np.clip(np.minimum(hi, 1e7) - lo, 0, None)
            / np.where(lo == hi, 1.0, hi - lo),
        )
        np.testing.assert_allclose(fracs, expected)

        # Equality is allowed on points
        mask = stars.get_mask("metallicities", stars.metallicities[2], "==")
        fracs = mask.get_fractions(stars)[0, 0, ::2]
        np.testing.assert_allclose(fracs, np.arange(fracs.size) == 2)

    def test_equality_on_finite_bins(self, stars):
        """No mass in a finite bin sits exactly at a single value."""
        mask = stars.get_mask("ages", 10 * Myr, "==")
        np.testing.assert_allclose(mask.get_fractions(stars), 0.0)
        mask = stars.get_mask("ages", 10 * Myr, "!=")
        np.testing.assert_allclose(mask.get_fractions(stars), 1.0)

    def test_masked_extraction_matches_clipped_bin(self, test_grid, stars):
        """A masked straddling bin must equal the clipped bin on its own.

        Masking a 5-20 Myr bin at 10 Myr must give exactly the emission of
        a 5-10 Myr bin holding the passing fraction of the mass.
        """
        model = IncidentEmission(test_grid)
        model.add_mask("log10ages", "<", 7 * dimensionless)
        masked = stars.get_spectra(model).lnu

        clipped = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=1e7 * yr,
            metal_dist=0.01,
        )
        _set_age_bin(clipped, 5e6, 1e7, 5.0 / 15.0, metallicity_index=3)
        expected = clipped.get_spectra(IncidentEmission(test_grid)).lnu
        np.testing.assert_allclose(masked, expected, rtol=1e-10)

    def test_masked_weights_are_reused(self, test_grid, stars):
        """Extractions sharing a mask share one set of stored weights."""
        young = IncidentEmission(test_grid, label="young")
        young.add_mask("log10ages", "<", 7 * dimensionless)
        also_young = IncidentEmission(test_grid, label="also_young")
        also_young.add_mask("log10ages", "<", 7 * dimensionless)
        old = IncidentEmission(test_grid, label="old")
        old.add_mask("log10ages", ">=", 7 * dimensionless)
        for model in (young, also_young, old):
            stars.get_spectra(model)
        assert len(stars._grid_weights) == 2
        np.testing.assert_allclose(
            stars.spectra["young"].lnu, stars.spectra["also_young"].lnu
        )

    def test_fixed_parameter_masks_whole_population(self, test_grid, stars):
        """A mask on a fixed model parameter includes or excludes it all."""
        model = IncidentEmission(test_grid, fesc=0.3)
        mask = stars.get_mask("fesc", 0.5, "<", attr_override_obj=model)
        np.testing.assert_allclose(mask.get_fractions(stars), 1.0)
        mask = stars.get_mask("fesc", 0.1, "<", attr_override_obj=model)
        np.testing.assert_allclose(mask.get_fractions(stars), 0.0)


class TestBinnedSFH:
    """Tests for binning SFH and metallicity distribution functions."""

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
            SFH.DelayedExponential(
                tau=1e9 * yr, max_age=5e9 * yr, min_age=1e8 * yr
            ),
            SFH.LogNormal(tau=0.5, peak_age=1e9 * yr, max_age=5e9 * yr),
            SFH.Continuity(logsfr_ratios=np.array([0.3, -0.2])),
        ],
    )
    def test_cdf_matches_quad(self, sfh):
        """Binned masses must match direct integration of the SFR."""
        edges = np.array([0, 5e6, 3e7, 2e8, 9e8, 1.5e9, 4e9, 6e9, 1e10])
        expected = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            points = [p for p in sfh._get_breakpoints() if lo < p < hi]
            expected.append(
                quad(sfh.get_sfr, lo, hi, limit=500, points=points or None)[0]
            )
        expected = np.array(expected)
        np.testing.assert_allclose(
            sfh.get_bin_masses(edges),
            expected,
            rtol=0,
            atol=1e-5 * np.abs(expected).max(),
        )

    def test_normal_zdist_cdf_matches_quad(self):
        """Binned metallicity weights must match direct integration."""
        zdist = ZDist.Normal(0.01, 0.005)
        edges = np.array([0, 1e-3, 5e-3, 0.01, 0.02, 0.04])
        expected = [
            quad(zdist.get_dist_weight, lo, hi)[0]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        np.testing.assert_allclose(
            zdist.get_bin_weights(edges), expected, atol=1e-12
        )

    def test_piecewise_constant_sfh_is_exact(self, test_grid, monkeypatch):
        """A piecewise constant SFH doesn't depend on the subdivisions.

        Its bin edges are breakpoints, so every fine bin has a constant SFR
        and the uniform mass within each bin is exact.
        """
        sfh = SFH.Continuity(logsfr_ratios=np.array([0.3, -0.2]))
        sfzhs = []
        for subdivisions in (1, 8):
            monkeypatch.setattr(
                stars_module, "SFZH_SUBDIVISIONS", subdivisions
            )
            sfzhs.append(
                Stars(
                    test_grid.log10ages,
                    test_grid.metallicities,
                    sf_hist=sfh,
                    metal_dist=0.01,
                ).sfzh
            )
        np.testing.assert_allclose(sfzhs[0], sfzhs[1], rtol=1e-12)

    def test_breakpoints_are_bin_edges(self, test_grid):
        """No age bin may straddle an SFH breakpoint."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=37 * Myr, min_age=3 * Myr),
            metal_dist=0.01,
        )
        edges = stars.get_bin_edges("ages", values_only=True)
        assert np.any(np.isclose(edges, 37e6, rtol=1e-12))
        assert np.any(np.isclose(edges, 3e6, rtol=1e-12))

    @pytest.mark.parametrize(
        ("sf_hist", "metal_dist"),
        [
            (1e7 * yr, ZDist.Normal(0.01, 0.005)),
            (SFH.Constant(max_age=50 * Myr), 0.004),
            (SFH.Constant(max_age=50 * Myr), ZDist.DeltaConstant(0.004)),
        ],
    )
    def test_mixed_routes(self, test_grid, sf_hist, metal_dist):
        """Points and bins can be mixed across the two axes."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=sf_hist,
            metal_dist=metal_dist,
            initial_mass=1e9 * Msun,
        )
        np.testing.assert_allclose(stars.sfzh.sum(), 1e9, rtol=1e-10)

    def test_constant_sfh_matches_particles(self, test_grid):
        """A constant SFH must match particles spread evenly over its ages.

        The particles are placed at evenly spaced ages (a deterministic
        quadrature rather than a random sample) so they converge to the
        exact answer, which the binned SFH must agree with.
        """
        model = IncidentEmission(test_grid)
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=50 * Myr),
            metal_dist=0.004,
            initial_mass=1e9 * Msun,
        )
        nparts = 50000
        particles = ParticleStars(
            initial_masses=np.full(nparts, 1e9 / nparts) * Msun,
            ages=(np.arange(nparts) + 0.5) / nparts * 5e7 * yr,
            metallicities=np.full(nparts, 0.004),
        )
        param = stars.get_spectra(model).lnu.value
        part = particles.get_spectra(model).lnu.value
        good = part > part.max() * 1e-6
        np.testing.assert_allclose(param[good], part[good], rtol=2e-3)


class TestBinConsumers:
    """Tests for the methods built on the population's bins."""

    def test_earlier_time_is_exact(self, test_grid):
        """A constant SFH 30 Myr earlier is the same SFH 30 Myr shorter."""
        model = IncidentEmission(test_grid)
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=100 * Myr),
            metal_dist=0.01,
            initial_mass=1e9 * Msun,
        )
        earlier = stars.get_at_earlier_time(30 * Myr)
        direct = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=70 * Myr),
            metal_dist=0.01,
            initial_mass=0.7e9 * Msun,
        )
        np.testing.assert_allclose(earlier.initial_mass, 0.7e9 * Msun)
        np.testing.assert_allclose(
            stars.calculate_initial_mass_at_age(30 * Myr), 0.7e9 * Msun
        )
        np.testing.assert_allclose(
            earlier.get_spectra(model).lnu,
            direct.get_spectra(model).lnu,
            rtol=1e-8,
        )

    def test_surviving_mass_off_grid_matches_particles(self, test_grid):
        """Surviving mass on axes other than the grid's matches particles."""
        log10ages = np.linspace(6.05, 9.95, 7)
        metallicities = np.array([0.0005, 0.003, 0.017])
        sfzh = np.random.default_rng(3).random((7, 3))
        stars = Stars.from_sfzh(log10ages, metallicities, sfzh)
        ages, metals = np.meshgrid(10**log10ages, metallicities, indexing="ij")
        particles = ParticleStars(
            initial_masses=sfzh.ravel() * Msun,
            ages=ages.ravel() * yr,
            metallicities=metals.ravel(),
        )
        np.testing.assert_allclose(
            stars.calculate_surviving_mass(test_grid),
            particles.calculate_surviving_mass(test_grid),
            rtol=1e-10,
        )

    def test_get_sfzh_is_lossless(self, test_grid):
        """Remapping keeps the bins so remapping back recovers the SFZH."""
        model = IncidentEmission(test_grid)
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.DelayedExponential(tau=1e9 * yr, max_age=5e9 * yr),
            metal_dist=ZDist.Normal(0.01, 0.005),
            initial_mass=1e9 * Msun,
        )
        coarse = stars.get_sfzh(np.linspace(6, 11, 6), np.array([1e-4, 0.02]))
        assert coarse.sfzh.shape == (6, 2)
        np.testing.assert_allclose(coarse.sfzh.sum(), 1e9, rtol=1e-10)
        back = coarse.get_sfzh(test_grid.log10ages, test_grid.metallicities)
        np.testing.assert_allclose(back.sfzh, stars.sfzh, rtol=1e-10)
        np.testing.assert_allclose(
            coarse.get_spectra(model).lnu,
            stars.get_spectra(model).lnu,
            rtol=1e-10,
        )

    def test_addition_with_different_edges(self, test_grid):
        """Adding populations binned differently sums their emission."""
        model = IncidentEmission(test_grid)
        sfzh = np.random.default_rng(4).random(
            (test_grid.log10ages.size, test_grid.metallicities.size)
        )
        components = [
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=SFH.Constant(max_age=50 * Myr),
                metal_dist=ZDist.Normal(0.01, 0.005),
                initial_mass=1e9 * Msun,
            ),
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=SFH.Gaussian(peak_age=2e9 * yr, sigma=3e8 * yr),
                metal_dist=0.004,
                initial_mass=1e9 * Msun,
            ),
            Stars.from_sfzh(
                test_grid.log10ages, test_grid.metallicities, sfzh
            ),
        ]
        combined = components[0] + components[1] + components[2]
        expected = sum(c.get_spectra(model).lnu for c in components)
        np.testing.assert_allclose(
            combined.get_spectra(model).lnu, expected, rtol=1e-10
        )
        np.testing.assert_allclose(
            combined.initial_mass, 2e9 * Msun + sfzh.sum() * Msun
        )

    def test_average_sfr_with_straddling_bins(self, test_grid):
        """A unit SFR averages to 1 over a range cutting through bins."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=SFH.Constant(max_age=100 * Myr),
            metal_dist=0.01,
        )
        sfr = stars.calculate_average_sfr(t_range=(25 * Myr, 60 * Myr))
        np.testing.assert_allclose(sfr.to("Msun/yr").value, 1.0, rtol=1e-10)


class TestRebinning:
    """Tests for combining bins with different edges."""

    def test_union_keeps_points(self):
        """Zero width bins survive the union of edges."""
        edges = union_edges(np.array([1.0, 1.0, 3.0]), np.array([0.0, 2.0]))
        np.testing.assert_array_equal(edges, [0.0, 1.0, 1.0, 2.0, 3.0])

    def test_rebin_conserves_and_splits(self):
        """Finite bins split by width and points move to points."""
        edges = np.array([1.0, 1.0, 3.0])
        new_edges = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        masses = np.array([[2.0, 4.0]])
        rebinned = rebin_axis(masses, 1, edges, new_edges)
        np.testing.assert_allclose(rebinned, [[0.0, 2.0, 2.0, 2.0]])


class TestPopulations:
    """Tests for Stars holding several populations."""

    @pytest.fixture
    def populations(self, test_grid):
        """Return three differently binned single population Stars."""
        return [
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=SFH.Constant(max_age=50 * Myr),
                metal_dist=ZDist.Normal(0.01, 0.005),
                initial_mass=1e9 * Msun,
            ),
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=SFH.DelayedExponential(tau=1e9 * yr, max_age=4e9 * yr),
                metal_dist=0.004,
                initial_mass=2e9 * Msun,
            ),
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=1e7 * yr,
                metal_dist=0.02,
                initial_mass=5e8 * Msun,
            ),
        ]

    def test_populations_are_kept(self, populations):
        """Combining keeps every population and its mass."""
        combined = Stars.from_populations(populations)
        assert combined.bin_masses.shape[0] == 3
        np.testing.assert_allclose(
            combined.bin_masses.sum(axis=tuple(range(1, 3))),
            [1e9, 2e9, 5e8],
            rtol=1e-10,
        )

    @pytest.mark.parametrize("masked", [False, True])
    def test_integrated_emission_is_the_sum(
        self, test_grid, populations, masked
    ):
        """One extraction of all populations equals the sum of each."""
        model = IncidentEmission(test_grid)
        if masked:
            model.add_mask("log10ages", "<", 7.3 * dimensionless)
        combined = Stars.from_populations(populations)
        expected = sum(p.get_spectra(model).lnu for p in populations)
        np.testing.assert_allclose(
            combined.get_spectra(model).lnu, expected, rtol=1e-10
        )


class TestPerPopulationEmission:
    """Tests for the emission of each population with its own parameters."""

    TAU_V = (0.1, 0.5, 1.2)
    FESC = (0.0, 0.2, 0.4)

    def _populations(self, test_grid):
        """Return three populations with their own tau_v and fesc."""
        sfhs = (
            SFH.Constant(max_age=50 * Myr),
            SFH.DelayedExponential(tau=1e9 * yr, max_age=4e9 * yr),
            SFH.Gaussian(peak_age=5e8 * yr, sigma=1e8 * yr),
        )
        return [
            Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=sfh,
                metal_dist=ZDist.Normal(0.01, 0.005),
                initial_mass=1e9 * Msun,
                tau_v=tau_v,
                fesc=fesc,
                fesc_ly_alpha=0.1,
            )
            for sfh, tau_v, fesc in zip(sfhs, self.TAU_V, self.FESC)
        ]

    @pytest.mark.parametrize("bimodal", [False, True])
    def test_matches_separate_populations(self, test_grid, bimodal):
        """Per population parameters match each population on its own.

        One per population extraction of the combined Stars must give, for
        every emission in the model tree, each population's emission and
        their sum exactly as extracting each population separately with its
        own scalar parameters does.
        """

        def make_model(per_particle):
            if bimodal:
                return BimodalPacmanEmission(
                    test_grid,
                    tau_v_ism="tau_v",
                    tau_v_birth="tau_v",
                    dust_curve_ism=Calzetti2000(),
                    dust_curve_birth=Calzetti2000(),
                    per_particle=per_particle,
                )
            return PacmanEmission(
                test_grid, dust_curve=Calzetti2000(), per_particle=per_particle
            )

        expected = {}
        for stars in self._populations(test_grid):
            stars.get_spectra(make_model(False))
            for label, sed in stars.spectra.items():
                expected.setdefault(label, []).append(sed.lnu.value)

        combined = Stars.from_populations(self._populations(test_grid))
        combined.tau_v = np.array(self.TAU_V)
        combined.get_spectra(make_model(True))
        for label, rows in expected.items():
            rows = np.array(rows)
            scale = np.abs(rows).max()
            np.testing.assert_allclose(
                combined.particle_spectra[label].lnu.value,
                rows,
                rtol=0,
                atol=1e-12 * scale,
            )
            np.testing.assert_allclose(
                combined.spectra[label].lnu.value,
                rows.sum(axis=0),
                rtol=0,
                atol=1e-12 * scale,
            )

    def test_lines_match_separate_populations(self, test_grid):
        """Per population lines match each population on its own."""
        lines = test_grid.available_lines[:6]
        expected = {}
        for stars in self._populations(test_grid):
            stars.get_lines(
                lines, PacmanEmission(test_grid, dust_curve=Calzetti2000())
            )
            for label, lc in stars.lines.items():
                expected.setdefault(label, []).append(lc.luminosity.value)

        combined = Stars.from_populations(self._populations(test_grid))
        combined.tau_v = np.array(self.TAU_V)
        combined.get_lines(
            lines,
            PacmanEmission(
                test_grid, dust_curve=Calzetti2000(), per_particle=True
            ),
        )
        for label, rows in expected.items():
            rows = np.array(rows)
            np.testing.assert_allclose(
                combined.particle_lines[label].luminosity.value,
                rows,
                rtol=0,
                atol=1e-12 * max(np.abs(rows).max(), 1.0),
            )


class TestPopulationAccessAndImaging:
    """Tests for named populations and their morphologies."""

    @staticmethod
    def _filters(test_grid):
        """Return two top hat filters on the grid's wavelengths."""
        return FilterCollection(
            tophat_dict={
                "f1": {"lam_eff": 2000 * angstrom, "lam_fwhm": 400 * angstrom},
                "f2": {"lam_eff": 6000 * angstrom, "lam_fwhm": 1e3 * angstrom},
            },
            new_lam=test_grid.lam,
        )

    @staticmethod
    def _population(test_grid, sfh, tau_v, morphology=None):
        """Return a single population with its own tau_v and morphology."""
        return Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=sfh,
            metal_dist=ZDist.Normal(0.01, 0.005),
            initial_mass=1e9 * Msun,
            tau_v=tau_v,
            fesc=0.1,
            fesc_ly_alpha=0.1,
            morphology=morphology,
        )

    def _bulge_and_disk(self, test_grid):
        """Return a bulge and a disk with different SFHs and morphologies."""
        return [
            self._population(
                test_grid,
                SFH.DelayedExponential(tau=5e8 * yr, max_age=8e9 * yr),
                0.2,
                Sersic2D(r_eff=1 * kpc, sersic_index=4),
            ),
            self._population(
                test_grid,
                SFH.Constant(max_age=1e9 * yr),
                0.8,
                Sersic2D(
                    r_eff=4 * kpc, sersic_index=1, ellipticity=0.5, theta=0.3
                ),
            ),
        ]

    def test_named_access(self, test_grid):
        """Populations can be looked up by name with their own parameters."""
        stars = Stars.from_populations(
            self._bulge_and_disk(test_grid), names=["bulge", "disk"]
        )
        np.testing.assert_allclose(stars.tau_v, [0.2, 0.8])
        assert isinstance(stars.morphology, PerPopulation)
        disk = stars["disk"]
        assert disk.npop == 1
        assert disk.tau_v == 0.8
        assert disk.morphology.r_eff == 4 * kpc
        np.testing.assert_allclose(disk.initial_mass, 1e9 * Msun)
        np.testing.assert_allclose(stars[0].tau_v, 0.2)

    def test_bulge_and_disk_images(self, test_grid):
        """Images, cubes and line maps are the sum over the populations."""
        filters = self._filters(test_grid)
        imager = Instrument("img", filters=filters, resolution=0.2 * kpc)
        ifu = Instrument(
            "ifu",
            resolution=0.5 * kpc,
            lam=np.linspace(1500, 7000, 20) * angstrom,
        )
        lines = test_grid.available_lines[:3]
        line_imager = Instrument("lines", line_ids=lines, resolution=0.2 * kpc)
        model = PacmanEmission(test_grid, dust_curve=Calzetti2000())

        expected = {"image": 0, "cube": 0, "map": 0}
        for stars in self._bulge_and_disk(test_grid):
            stars.get_spectra(model)
            stars.get_lines(lines, model)
            stars.get_photo_lnu(filters)
            expected["image"] += stars.get_images_luminosity(
                "emergent", fov=20 * kpc, instrument=imager
            )["f1"].arr
            expected["cube"] += stars.get_data_cube(
                "emergent", fov=20 * kpc, instrument=ifu
            ).arr
            expected["map"] += stars.get_line_maps_luminosity(
                "emergent",
                line_ids=lines,
                fov=20 * kpc,
                instrument=line_imager,
            )[lines[0]].arr

        per_pop = PacmanEmission(
            test_grid, dust_curve=Calzetti2000(), per_particle=True
        )
        stars = Stars.from_populations(self._bulge_and_disk(test_grid))
        stars.get_spectra(per_pop)
        stars.get_lines(lines, per_pop)
        stars.get_particle_photo_lnu(filters)
        image = stars.get_images_luminosity(
            "emergent", fov=20 * kpc, instrument=imager
        )["f1"].arr
        cube = stars.get_data_cube(
            "emergent", fov=20 * kpc, instrument=ifu
        ).arr
        line_map = stars.get_line_maps_luminosity(
            "emergent", line_ids=lines, fov=20 * kpc, instrument=line_imager
        )[lines[0]].arr
        for name, result in (
            ("image", image),
            ("cube", cube),
            ("map", line_map),
        ):
            np.testing.assert_allclose(
                result,
                expected[name],
                rtol=0,
                atol=1e-12 * expected[name].max(),
            )

    def test_annuli_images(self, test_grid):
        """An Annuli image is the sum of each annulus imaged on its own."""
        filters = self._filters(test_grid)
        imager = Instrument("img", filters=filters, resolution=0.2 * kpc)
        model = PacmanEmission(
            test_grid, dust_curve=Calzetti2000(), per_particle=True
        )
        annuli = Annuli(
            Sersic2D(r_eff=3 * kpc, sersic_index=1),
            np.array([0, 1, 2, 4, 7, np.inf]) * kpc,
        )
        populations = [
            self._population(
                test_grid,
                SFH.Constant(max_age=(2 + 4 * i) * 1e8 * yr),
                0.2 * i,
            )
            for i in range(5)
        ]
        stars = Stars.from_populations(populations)
        stars.morphology = annuli
        stars.get_spectra(model)
        stars.get_particle_photo_lnu(filters)
        image = stars.get_images_luminosity(
            "emergent", fov=30 * kpc, instrument=imager
        )["f1"].arr

        expected = 0
        for i, population in enumerate(populations):
            population.morphology = annuli.get_population_morphology(i)
            population.get_spectra(model)
            population.get_particle_photo_lnu(filters)
            expected += population.get_images_luminosity(
                "emergent", fov=30 * kpc, instrument=imager
            )["f1"].arr
        np.testing.assert_allclose(
            image, expected, rtol=0, atol=1e-12 * expected.max()
        )

        # Every annulus is normalised over its own pixels so the image holds
        # all of the light
        np.testing.assert_allclose(
            image.sum(),
            stars.particle_photo_lnu["emergent"]["f1"].value.sum(),
            rtol=1e-10,
        )

    def test_pipeline(self, test_grid):
        """Multi population parametric galaxies run through a Pipeline."""
        filters = self._filters(test_grid)
        imager = Instrument("img", filters=filters, resolution=0.5 * kpc)
        model = PacmanEmission(
            test_grid, dust_curve=Calzetti2000(), per_particle=True
        )
        galaxies = [
            ParametricGalaxy(
                Stars.from_populations(self._bulge_and_disk(test_grid))
            )
            for _ in range(3)
        ]
        pipeline = Pipeline(emission_model=model, nthreads=1, verbose=0)
        pipeline.add_galaxies(list(galaxies))
        pipeline.get_sfzh(test_grid.log10ages, test_grid.metallicities)
        pipeline.get_sfh(test_grid.log10ages)
        pipeline.get_spectra()
        pipeline.get_photometry_luminosities(imager)
        pipeline.get_lines(test_grid.available_lines[:3])
        pipeline.get_images_luminosity(imager, fov=20 * kpc)
        pipeline.run()

        # The Pipeline adapts its model to its instruments so compare with a
        # fresh one
        direct = Stars.from_populations(self._bulge_and_disk(test_grid))
        direct.get_spectra(
            PacmanEmission(
                test_grid, dust_curve=Calzetti2000(), per_particle=True
            )
        )
        np.testing.assert_allclose(
            np.asarray(pipeline.lnu_spectra["Stars"]["emergent"][0]),
            direct.spectra["emergent"].lnu.value,
            rtol=1e-10,
        )
        np.testing.assert_allclose(galaxies[0].sfzh.sum(), 2e9, rtol=1e-10)


class TestFromBinned:
    """Tests for creating Stars from binned masses."""

    EDGES = np.array([2e9, 5e8, 1e8, 1e7, 1e6])
    METALS = np.array([0.002, 0.006, 0.014, 0.02])

    def test_matches_constant_bins(self, test_grid):
        """Binned masses equal the same bins built from constant SFHs.

        Lookback (decreasing) age edges with one metallicity per age bin
        (zero width metallicity bins) must give exactly the emission of a
        constant SFH in each bin.
        """
        model = IncidentEmission(test_grid)
        bin_masses = np.array([3e8, 1e8, 5e7, 1e7])
        z_edges = np.repeat(self.METALS, 2)
        masses = np.zeros((4, z_edges.size - 1))
        masses[np.arange(4), 2 * np.arange(4)] = bin_masses
        binned = Stars.from_binned(
            test_grid.log10ages,
            test_grid.metallicities,
            self.EDGES,
            z_edges,
            masses,
        )

        constant = None
        for i in range(4):
            stars = Stars(
                test_grid.log10ages,
                test_grid.metallicities,
                sf_hist=SFH.Constant(
                    min_age=self.EDGES[i + 1] * yr, max_age=self.EDGES[i] * yr
                ),
                metal_dist=ZDist.DeltaConstant(metallicity=self.METALS[i]),
                initial_mass=bin_masses[i] * Msun,
            )
            constant = stars if constant is None else constant + stars
        np.testing.assert_allclose(
            binned.initial_mass, bin_masses.sum() * Msun
        )
        np.testing.assert_allclose(
            binned.get_spectra(model).lnu,
            constant.get_spectra(model).lnu,
            rtol=1e-10,
        )

    def test_populations_and_parameters(self, test_grid):
        """Several populations can be binned at once with their own names."""
        masses = np.random.default_rng(5).random((3, 4, 2))
        stars = Stars.from_binned(
            test_grid.log10ages,
            test_grid.metallicities,
            self.EDGES,
            np.array([0.001, 0.01, 0.03]),
            masses,
            names=["a", "b", "c"],
            tau_v=np.array([0.1, 0.2, 0.3]),
        )
        assert stars.npop == 3
        assert stars["b"].tau_v == 0.2
        np.testing.assert_allclose(
            stars["c"].initial_mass, masses[2].sum() * Msun
        )

    def test_bad_shapes_raise(self, test_grid):
        """Masses must match the edges."""
        with pytest.raises(exceptions.InconsistentArguments):
            Stars.from_binned(
                test_grid.log10ages,
                test_grid.metallicities,
                self.EDGES,
                np.array([0.001, 0.01]),
                np.ones((3, 1)),
            )
