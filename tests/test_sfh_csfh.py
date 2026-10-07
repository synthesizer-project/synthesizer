"""Tests for the Madau & Dickinson (2014) cosmic star formation history.

Covers the ``CSFH.MadauDickinson`` model, which converts the cosmic star
formation rate density, parametrised as a function of redshift, into a
function of stellar age for a galaxy observed at a given redshift.
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck18
from unyt import Gyr, Msun

from synthesizer.parametric import CSFH
from synthesizer.parametric.stars import Stars
from synthesizer.utils.integrate import trapezoid

# The parameters fit by MD+14.
PARAMS = {"b": 2.7, "c": 2.9, "d": 5.6}


def md_function(z):
    """Evaluate the Madau & Dickinson function at redshift z."""
    b, c, d = PARAMS["b"], PARAMS["c"], PARAMS["d"]
    return (1.0 + z) ** b / (1.0 + ((1.0 + z) / c) ** d)


@pytest.fixture
def cosmo():
    """Return the cosmology assumed by MD+14."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def md_sfh(cosmo):
    """Return a Madau & Dickinson SFH observed at z=0."""
    return CSFH.MadauDickinson(redshift=0.0, cosmo=cosmo, **PARAMS)


class TestMadauDickinsonSFH:
    """Tests for the SFH.MadauDickinsonCSFH model."""

    def test_name(self, md_sfh):
        """The SFH should report its name."""
        assert md_sfh.name == "MadauDickinsonCSFH"

    def test_finegrid_ascending(self, md_sfh):
        """The age grid must be ascending for downstream interpolation."""
        assert np.all(np.diff(md_sfh.finegrid) > 0)

    def test_finegrid_spans_universe_age(self, cosmo):
        """The age grid should start at 0 and end within the universe age."""
        sfh = CSFH.MadauDickinson(redshift=1.0, cosmo=cosmo, **PARAMS)
        t_univ = cosmo.age(1.0).to("yr").value
        assert np.isclose(sfh.finegrid.min(), 0.0, atol=1.0)
        assert sfh.finegrid.max() < t_univ

    def test_sfh_finite_and_positive(self, md_sfh):
        """The reconstructed SFH should be finite and non-negative."""
        t, sfh = md_sfh.calculate_sfh()
        assert np.all(np.isfinite(sfh))
        assert np.all(sfh >= 0)
        assert trapezoid(sfh, x=t) > 0

    def test_float64(self, md_sfh):
        """The stored SFH must be float64 (precision requirement)."""
        assert md_sfh.finegrid.dtype == np.float64
        assert md_sfh.intsfh.dtype == np.float64

    def test_age_matches_formation_redshift(self):
        """Stars formed at z should have the SFR of the function at z.

        A star formed at redshift z in a galaxy observed at z=0 has an age
        of t_univ(0) - t(z).
        """
        cosmologies = [
            FlatLambdaCDM(H0=70, Om0=0.3),
            Planck18,
            LambdaCDM(H0=70, Om0=0.3, Ode0=0.6),
        ]
        for cosmo in cosmologies:
            sfh = CSFH.MadauDickinson(redshift=0.0, cosmo=cosmo, **PARAMS)
            for z in (0.5, 1.0, 2.0, 4.0, 6.0):
                age = sfh.t_univ - cosmo.age(z).to("yr").value
                assert np.isclose(sfh.get_sfr(age), md_function(z), rtol=1e-3)

    def test_observation_redshift_shift(self, cosmo, md_sfh):
        """Observing at z_obs should shift the SFH by the lookback time."""
        sfh_z = CSFH.MadauDickinson(redshift=1.0, cosmo=cosmo, **PARAMS)
        lookback = md_sfh.t_univ - sfh_z.t_univ

        ages = np.linspace(0.0, 0.9 * sfh_z.t_univ, 200)
        assert np.allclose(
            sfh_z.get_sfr(ages), md_sfh.get_sfr(ages + lookback), rtol=1e-3
        )

    def test_zero_outside_universe(self, md_sfh):
        """No stars should have negative age or be older than the universe."""
        assert md_sfh.get_sfr(-1e6) == 0.0
        assert md_sfh.get_sfr(md_sfh.t_univ * 1.01) == 0.0

    def test_age_window(self, cosmo, md_sfh):
        """The SFH should be zero outside min_age <= age < max_age."""
        sfh = CSFH.MadauDickinson(
            redshift=0.0,
            cosmo=cosmo,
            min_age=1 * Gyr,
            max_age=8 * Gyr,
            **PARAMS,
        )
        ages = np.array([0.0, 9.99e8, 1e9, 5e9, 7.99e9, 8e9, 1.2e10])
        inside = (ages >= 1e9) & (ages < 8e9)
        sfr = sfh.get_sfr(ages)

        # Zero outside, positive inside, and unchanged from the full SFH.
        assert np.all(sfr[~inside] == 0.0)
        assert np.all(sfr[inside] > 0.0)
        assert np.array_equal(sfr[inside], md_sfh.get_sfr(ages[inside]))
        assert sfh.get_sfr(1e9 - 1.0) == 0.0
        assert sfh.get_sfr(1e9) > 0.0

    def test_scalar_and_array_agree(self, md_sfh):
        """Floats, numpy scalars and arrays should give the same SFR."""
        for age in (0.0, 3e8, 5e9, 1.4e10):
            expected = md_sfh.get_sfr(np.array([age]))[0]
            assert np.isclose(md_sfh.get_sfr(float(age)), expected)
            assert np.isclose(md_sfh.get_sfr(np.float64(age)), expected)

    def test_init_from_prior(self, cosmo):
        """init_from_prior should draw parameters within the prior range."""
        np.random.seed(0)
        sfh = CSFH.MadauDickinson.init_from_prior(
            redshift=1.0,
            b=[2.0, 3.0],
            c=3.1,
            d=6.26,
            cosmo=cosmo,
            min_age=[0.0, 1.0] * Gyr,
        )
        assert 2.0 <= sfh.b <= 3.0
        assert 0.0 <= sfh.min_age <= 1e9

    def test_redshift_must_be_below_100(self, cosmo):
        """Observation redshifts of 100 or above should be rejected."""
        for redshift in (100.0, 150.0):
            with pytest.raises(ValueError, match="redshift must be less"):
                CSFH.MadauDickinson(redshift=redshift, cosmo=cosmo, **PARAMS)

        # Just below the limit should still build a valid SFH
        sfh = CSFH.MadauDickinson(redshift=99.0, cosmo=cosmo, **PARAMS)
        assert np.all(np.diff(sfh.finegrid) > 0)
        assert np.all(np.isfinite(sfh.intsfh))

    def test_n_grid_must_be_integer_of_at_least_two(self, cosmo):
        """n_grid should be rejected unless it is an integer >= 2."""
        for n_grid in (1, 0, -5, 10.0, 10.5, "10", None):
            with pytest.raises(ValueError, match="n_grid must be an integer"):
                CSFH.MadauDickinson(
                    redshift=0.0, cosmo=cosmo, n_grid=n_grid, **PARAMS
                )

        # The smallest grid and numpy integers should be accepted
        for n_grid in (2, np.int64(10)):
            sfh = CSFH.MadauDickinson(
                redshift=0.0, cosmo=cosmo, n_grid=n_grid, **PARAMS
            )
            assert sfh.finegrid.size == n_grid
            assert np.all(np.diff(sfh.finegrid) > 0)


class TestMadauDickinsonIntegration:
    """Tests that the SFH plugs into Stars and spectra generation."""

    def test_builds_valid_sfzh(self, test_grid, md_sfh):
        """A Stars object built with the SFH should have a valid SFZH."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=md_sfh,
            metal_dist=0.01,
            initial_mass=1e10 * Msun,
        )
        assert stars.sfzh.shape == (
            test_grid.log10ages.size,
            test_grid.metallicities.size,
        )
        assert np.all(np.isfinite(stars.sfzh))
        assert np.all(stars.sfzh >= 0)
        assert np.isclose(stars.sfzh.sum(), 1e10, rtol=1e-6)

    def test_get_spectra(self, test_grid, incident_emission_model, md_sfh):
        """The SFH should produce a finite, positive spectrum."""
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=md_sfh,
            metal_dist=0.01,
            initial_mass=1e10 * Msun,
        )
        spectra = stars.get_spectra(incident_emission_model)
        assert np.all(np.isfinite(spectra._lnu))
        assert np.sum(spectra._lnu) > 0
