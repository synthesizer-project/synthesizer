"""Tests for parametric Stars mass calculation methods."""

import numpy as np
import pytest
from unyt import Gyr, Msun, yr

from synthesizer.parametric.stars import Stars as ParametricStars


@pytest.fixture
def simple_parametric_stars():
    """Return a parametric Stars object built from explicit arrays.

    Uses a flat SFH and a single metallicity so expected values are
    easy to compute analytically.
    """
    log10ages = np.array([6.0, 7.0, 8.0, 9.0])
    metallicities = np.array([0.02])
    # Uniform SFZH: 1e8 Msun per age-metallicity bin
    sfzh = np.ones((len(log10ages), len(metallicities))) * 1e8
    return ParametricStars(
        log10ages,
        metallicities,
        sfzh=sfzh,
    )


class TestCalculateInitialMassAtAge:
    """Tests for Stars.calculate_initial_mass_at_age."""

    def test_returns_unyt_quantity(self, simple_parametric_stars):
        """Return value should be a unyt_quantity (carries units)."""
        result = simple_parametric_stars.calculate_initial_mass_at_age(1e8)
        assert hasattr(result, "units"), "Result must carry units"

    def test_age_beyond_all_bins_returns_total_mass(
        self, simple_parametric_stars
    ):
        """An age older than all bins should return the full initial mass."""
        total = simple_parametric_stars.calculate_initial_mass_at_age(1e12)
        expected = np.sum(simple_parametric_stars.sf_hist) * Msun
        np.testing.assert_allclose(
            total.to("Msun").value,
            expected.to("Msun").value,
            rtol=1e-6,
        )

    def test_age_before_all_bins_returns_zero(self, simple_parametric_stars):
        """An age younger than all bins should return zero mass."""
        result = simple_parametric_stars.calculate_initial_mass_at_age(1.0)
        assert result.to("Msun").value == pytest.approx(0.0)

    def test_accepts_unyt_quantity_age(self, simple_parametric_stars):
        """Method should accept a unyt_quantity age and give the same result."""
        age_float = 1e8
        age_unyt = 1e8 * yr
        result_float = simple_parametric_stars.calculate_initial_mass_at_age(
            age_float
        )
        result_unyt = simple_parametric_stars.calculate_initial_mass_at_age(
            age_unyt
        )
        np.testing.assert_allclose(
            result_float.to("Msun").value,
            result_unyt.to("Msun").value,
            rtol=1e-10,
        )

    def test_accepts_gyr_unyt_quantity(self, simple_parametric_stars):
        """Method should correctly convert non-yr unyt_quantity ages."""
        age_gyr = 1 * Gyr  # 1e9 yr
        age_yr = 1e9  # plain float in years
        result_gyr = simple_parametric_stars.calculate_initial_mass_at_age(
            age_gyr
        )
        result_yr = simple_parametric_stars.calculate_initial_mass_at_age(
            age_yr
        )
        np.testing.assert_allclose(
            result_gyr.to("Msun").value,
            result_yr.to("Msun").value,
            rtol=1e-10,
        )

    def test_mass_is_non_negative(self, simple_parametric_stars):
        """Returned mass should always be non-negative."""
        for age in [1e5, 1e7, 1e8, 5e9]:
            result = simple_parametric_stars.calculate_initial_mass_at_age(age)
            assert result.to("Msun").value >= 0.0

    def test_mass_is_monotonically_increasing(self, simple_parametric_stars):
        """Older lookback ages should accumulate more or equal mass."""
        ages = [1e6, 1e7, 1e8, 1e9, 1e10]
        masses = [
            simple_parametric_stars.calculate_initial_mass_at_age(a)
            .to("Msun")
            .value
            for a in ages
        ]
        for i in range(len(masses) - 1):
            assert masses[i] <= masses[i + 1], (
                f"Mass at age {ages[i]} ({masses[i]}) exceeds mass at "
                f"age {ages[i+1]} ({masses[i+1]})"
            )
