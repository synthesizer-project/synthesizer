"""A test suite for the stochastic (GP + PSD) star formation history.

Covers the covariance kernels in ``synthesizer.parametric.sfh_kernels`` and the
``SFH.Stochastic`` model that draws a star formation history as a Gaussian
Process realisation (Iyer et al. 2024, arXiv:2208.05938).
"""

import numpy as np
import pytest
from unyt import Gyr, Msun

from synthesizer import exceptions
from synthesizer.parametric import SFH, Kernels
from synthesizer.parametric.stars import Stars


@pytest.fixture
def drw_kernel():
    """Return a damped random walk kernel."""
    return Kernels.DampedRandomWalk(sigma=0.3, tau=1 * Gyr)


@pytest.fixture
def stochastic_sfh(drw_kernel):
    """Return a stochastic SFH at z=1 with a fixed seed."""
    return SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=42)


class TestDampedRandomWalk:
    """Tests for the DampedRandomWalk covariance kernel."""

    def test_zero_lag_is_variance(self, drw_kernel):
        """The covariance at zero lag should equal sigma**2."""
        assert np.isclose(drw_kernel.covariance(0.0), 0.3**2)

    def test_one_efold(self, drw_kernel):
        """The covariance at a lag of tau should be sigma**2 / e."""
        tau_yr = (1 * Gyr).to("yr").value
        assert np.isclose(drw_kernel.covariance(tau_yr), 0.3**2 / np.e)

    def test_covariance_vectorised(self, drw_kernel):
        """Covariance should accept and return arrays."""
        lags = np.linspace(0, 1e10, 17)
        cov = drw_kernel.covariance(lags)
        assert cov.shape == lags.shape
        # Covariance decreases monotonically with lag
        assert np.all(np.diff(cov) <= 0)

    def test_covariance_matrix_symmetric_psd(self, drw_kernel):
        """The covariance matrix should be symmetric and positive definite."""
        tarr = np.linspace(0, 1e10, 100)
        cov = drw_kernel.build_covariance_matrix(tarr)
        assert cov.shape == (100, 100)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() >= -1e-12
        # The diagonal is the variance
        assert np.allclose(np.diag(cov), 0.3**2)

    def test_invalid_sigma(self):
        """A non-positive sigma should raise."""
        with pytest.raises(exceptions.InconsistentArguments):
            Kernels.DampedRandomWalk(sigma=0.0, tau=1 * Gyr)

    def test_invalid_tau(self):
        """A non-positive tau should raise."""
        with pytest.raises(exceptions.InconsistentArguments):
            Kernels.DampedRandomWalk(sigma=0.3, tau=0.0 * Gyr)

    def test_init_from_prior(self):
        """init_from_prior should draw parameters within the prior range."""
        np.random.seed(0)
        kernel = Kernels.DampedRandomWalk.init_from_prior(
            sigma=[0.1, 0.5], tau=[0.5, 3.0] * Gyr
        )
        assert 0.1 <= kernel.sigma <= 0.5
        # tau is stored in years, the prior was 0.5-3 Gyr
        assert 0.5e9 <= kernel.tau <= 3.0e9


class TestStochasticSFH:
    """Tests for the SFH.Stochastic model."""

    def test_name(self, stochastic_sfh):
        """The SFH should report its name."""
        assert stochastic_sfh.name == "Stochastic"

    def test_finegrid_ascending(self, stochastic_sfh):
        """The age grid must be ascending for downstream interpolation."""
        assert np.all(np.diff(stochastic_sfh.finegrid) >= 0)

    def test_finegrid_spans_universe_age(self, stochastic_sfh):
        """The age grid should span 0 to the age of the universe at z."""
        t_univ = stochastic_sfh.cosmo.age(1.0).to("yr").value
        assert np.isclose(stochastic_sfh.finegrid.min(), 0.0)
        assert np.isclose(stochastic_sfh.finegrid.max(), t_univ)

    def test_sfh_finite_and_positive(self, stochastic_sfh):
        """The reconstructed SFH should be finite and non-negative."""
        t, sfh = stochastic_sfh.calculate_sfh()
        assert np.all(np.isfinite(sfh))
        assert np.all(sfh >= 0)
        assert np.trapezoid(sfh, t) > 0

    def test_float64(self, stochastic_sfh):
        """The stored realisation must be float64 (precision requirement)."""
        assert stochastic_sfh.finegrid.dtype == np.float64
        assert stochastic_sfh.intsfh.dtype == np.float64

    def test_reproducible_with_seed(self, drw_kernel):
        """The same seed should give an identical realisation."""
        a = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=7)
        b = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=7)
        assert np.array_equal(a.intsfh, b.intsfh)

    def test_different_seeds_differ(self, drw_kernel):
        """Different seeds should give different realisations."""
        a = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=7)
        c = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=8)
        assert not np.array_equal(a.intsfh, c.intsfh)

    def test_small_sigma_recovers_constant(self):
        """A tiny sigma should recover the (flat) base SFH."""
        kernel = Kernels.DampedRandomWalk(sigma=1e-6, tau=1 * Gyr)
        sfh = SFH.Stochastic(redshift=1.0, kernel=kernel, random_seed=1)
        # With a constant base and negligible fluctuations the SFR is flat
        assert np.allclose(sfh.intsfh, sfh.intsfh[0], rtol=1e-4)

    def test_callable_base_sfh(self, drw_kernel):
        """A callable base SFH should be accepted."""
        sfh = SFH.Stochastic(
            redshift=1.0,
            kernel=drw_kernel,
            base_sfh=lambda t: np.exp(-t / 3e9),
            random_seed=3,
        )
        assert np.all(np.isfinite(sfh.intsfh))
        assert np.all(sfh.intsfh >= 0)

    def test_bad_base_sfh_array_length(self, drw_kernel):
        """A base SFH array of the wrong length should raise."""
        with pytest.raises(exceptions.InconsistentArguments):
            SFH.Stochastic(
                redshift=1.0,
                kernel=drw_kernel,
                base_sfh=np.ones(10),
                n_grid=1000,
            )

    def test_bad_base_sfh_string(self, drw_kernel):
        """An unknown base SFH string should raise."""
        with pytest.raises(exceptions.InconsistentArguments):
            SFH.Stochastic(
                redshift=1.0, kernel=drw_kernel, base_sfh="not_a_model"
            )


class TestStochasticIntegration:
    """Tests that Stochastic plugs into Stars and spectra generation."""

    def test_builds_valid_sfzh(self, test_grid, drw_kernel):
        """A Stars object built with Stochastic should have a valid SFZH."""
        sfh = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=42)
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=sfh,
            metal_dist=0.01,
            initial_mass=1e10 * Msun,
        )
        assert stars.sfzh.shape == (
            test_grid.log10ages.size,
            test_grid.metallicities.size,
        )
        assert np.all(np.isfinite(stars.sfzh))
        assert np.all(stars.sfzh >= 0)
        assert stars.sfzh.sum() > 0

    def test_get_spectra(self, test_grid, incident_emission_model, drw_kernel):
        """A Stochastic SFH should produce a finite, positive spectrum."""
        sfh = SFH.Stochastic(redshift=1.0, kernel=drw_kernel, random_seed=42)
        stars = Stars(
            test_grid.log10ages,
            test_grid.metallicities,
            sf_hist=sfh,
            metal_dist=0.01,
            initial_mass=1e10 * Msun,
        )
        spectra = stars.get_spectra(incident_emission_model)
        assert np.all(np.isfinite(spectra._lnu))
        assert np.sum(spectra._lnu) > 0
