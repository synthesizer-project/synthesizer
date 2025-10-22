"""A test suite for the cosmology module."""

import numpy as np
from astropy.cosmology import (
    WMAP9,
    FlatLambdaCDM,
    Flatw0waCDM,
    Flatw0wzCDM,
    FlatwCDM,
    FlatwpwaCDM,
    Planck18,
    w0waCDM,
    w0wzCDM,
    wCDM,
    wpwaCDM,
)
from unyt import Mpc

from synthesizer.cosmology import (
    get_angular_diameter_distance,
    get_luminosity_distance,
)


class TestCosmologyDistanceFunctions:
    """Tests for cosmology distance calculation functions."""

    def test_get_luminosity_distance_scalar(self):
        """Test luminosity distance calculation for scalar redshift."""
        redshift = 1.0
        result = get_luminosity_distance(Planck18, redshift)

        # Compare with direct astropy calculation
        expected = Planck18.luminosity_distance(redshift).to("Mpc").value * Mpc

        assert isinstance(result, type(expected)), (
            f"Expected unyt_quantity but got {type(result)}"
        )
        assert result.units == Mpc, (
            f"Expected Mpc units but got {result.units}"
        )
        assert np.isclose(result.value, expected.value), (
            f"Expected {expected.value} but got {result.value}"
        )

    def test_get_luminosity_distance_multiple_redshifts(self):
        """Test luminosity distance calculation for multiple redshifts."""
        redshifts = [0.1, 0.5, 1.0, 2.0]

        for z in redshifts:
            result = get_luminosity_distance(Planck18, z)
            expected = Planck18.luminosity_distance(z).to("Mpc").value * Mpc

            assert np.isclose(result.value, expected.value), (
                f"For z={z}, expected {expected.value} but got {result.value}"
            )

    def test_get_angular_diameter_distance_scalar(self):
        """Test angular diameter distance calculation for scalar redshift."""
        redshift = 1.0
        result = get_angular_diameter_distance(Planck18, redshift)

        # Compare with direct astropy calculation
        expected = (
            Planck18.angular_diameter_distance(redshift).to("Mpc").value * Mpc
        )

        assert isinstance(result, type(expected)), (
            f"Expected unyt_quantity but got {type(result)}"
        )
        assert result.units == Mpc, (
            f"Expected Mpc units but got {result.units}"
        )
        assert np.isclose(result.value, expected.value), (
            f"Expected {expected.value} but got {result.value}"
        )

    def test_get_angular_diameter_distance_multiple_redshifts(self):
        """Test angular diameter distance for multiple redshifts."""
        redshifts = [0.1, 0.5, 1.0, 2.0]

        for z in redshifts:
            result = get_angular_diameter_distance(Planck18, z)
            expected = (
                Planck18.angular_diameter_distance(z).to("Mpc").value * Mpc
            )

            assert np.isclose(result.value, expected.value), (
                f"For z={z}, expected {expected.value} but got {result.value}"
            )

    def test_distance_relationship(self):
        """Test lum and ang diameter distances have correct relationship."""
        redshift = 1.0

        lum_dist = get_luminosity_distance(Planck18, redshift)
        ang_dist = get_angular_diameter_distance(Planck18, redshift)

        # For a flat universe: D_L = D_A * (1 + z)^2
        expected_ratio = (1 + redshift) ** 2
        actual_ratio = lum_dist.value / ang_dist.value

        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10), (
            f"Expected D_L/D_A = {expected_ratio} but got {actual_ratio}"
        )

    def test_zero_redshift(self):
        """Test distance calculations at zero redshift."""
        redshift = 0.0

        lum_dist = get_luminosity_distance(Planck18, redshift)
        ang_dist = get_angular_diameter_distance(Planck18, redshift)

        # At z=0, both distances should be very small (effectively zero)
        assert lum_dist.value < 1e-10, (
            f"Expected ~0 luminosity distance at z=0 but got {lum_dist.value}"
        )
        assert ang_dist.value < 1e-10, (
            "Expected ~0 angular diameter distance at z=0 but got "
            f"{ang_dist.value}"
        )
        assert np.isclose(lum_dist.value, ang_dist.value), (
            "At z=0, luminosity and angular diameter distances should be equal"
        )

    def test_caching_behavior(self):
        """Test that caching works correctly for repeated calls."""
        redshift = 1.0

        # First call
        result1 = get_luminosity_distance(Planck18, redshift)
        # Second call with same parameters should return cached result
        result2 = get_luminosity_distance(Planck18, redshift)

        assert np.isclose(result1.value, result2.value), (
            "Cached result should match original calculation"
        )

        # Same test for angular diameter distance
        result3 = get_angular_diameter_distance(Planck18, redshift)
        result4 = get_angular_diameter_distance(Planck18, redshift)

        assert np.isclose(result3.value, result4.value), (
            "Cached result should match original calculation"
        )

    def test_different_cosmologies(self):
        """Test that functions work with different cosmology objects."""
        redshift = 1.0

        # Test with different cosmology
        result_planck = get_luminosity_distance(Planck18, redshift)
        result_wmap = get_luminosity_distance(WMAP9, redshift)

        # Results should be different for different cosmologies
        assert not np.isclose(
            result_planck.value, result_wmap.value, rtol=1e-3
        ), "Different cosmologies should give different distance results"

        # But both should have correct units
        assert result_planck.units == Mpc
        assert result_wmap.units == Mpc

    def test_comprehensive_cosmology_support(self):
        """Test cosmology caching supports all major cosmology classes."""
        redshift = 1.0

        # Test a comprehensive set of cosmology models
        cosmologies = [
            # Basic ΛCDM models
            FlatLambdaCDM(H0=70, Om0=0.3),
            # Constant w models
            wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1.1),
            FlatwCDM(H0=70, Om0=0.3, w0=-1.1),
            # Time-varying w models
            w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1.0, wa=0.1),
            Flatw0waCDM(H0=70, Om0=0.3, w0=-1.0, wa=0.1),
            # Pivot redshift models
            wpwaCDM(H0=70, Om0=0.3, Ode0=0.7, wp=-1.0, wa=0.1, zp=0.5),
            FlatwpwaCDM(H0=70, Om0=0.3, wp=-1.0, wa=0.1, zp=0.5),
            # Redshift derivative models
            w0wzCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1.0, wz=0.1),
            Flatw0wzCDM(H0=70, Om0=0.3, w0=-1.0, wz=0.1),
        ]

        # Test both distance functions with each cosmology
        for cosmo in cosmologies:
            lum_dist = get_luminosity_distance(cosmo, redshift)
            ang_dist = get_angular_diameter_distance(cosmo, redshift)

            # Basic sanity checks
            assert lum_dist.units == Mpc, (
                "Luminosity distance should have Mpc units for "
                f"{cosmo.__class__.__name__}"
            )
            assert ang_dist.units == Mpc, (
                "Angular diameter distance should have Mpc units for "
                f"{cosmo.__class__.__name__}"
            )
            assert lum_dist.value > 0, (
                "Luminosity distance should be positive for "
                f"{cosmo.__class__.__name__}"
            )
            assert ang_dist.value > 0, (
                "Angular diameter distance should be positive for "
                f"{cosmo.__class__.__name__}"
            )

            # Test caching by calling again
            lum_dist_cached = get_luminosity_distance(cosmo, redshift)
            ang_dist_cached = get_angular_diameter_distance(cosmo, redshift)

            assert np.isclose(lum_dist.value, lum_dist_cached.value), (
                f"Cached luminosity distance should match for "
                f"{cosmo.__class__.__name__}"
            )
            assert np.isclose(ang_dist.value, ang_dist_cached.value), (
                f"Cached angular diameter distance should match for "
                f"{cosmo.__class__.__name__}"
            )

    def test_high_redshift(self):
        """Test distance calculations at high redshift."""
        redshift = 10.0

        lum_dist = get_luminosity_distance(Planck18, redshift)
        ang_dist = get_angular_diameter_distance(Planck18, redshift)

        # At high redshift, luminosity distance should be much larger than
        # angular diameter distance
        assert lum_dist.value > ang_dist.value, (
            "At high z, luminosity distance should exceed angular diameter"
            " distance"
        )

        # Both should be positive and reasonable values
        assert lum_dist.value > 0, "Luminosity distance should be positive"
        assert ang_dist.value > 0, (
            "Angular diameter distance should be positive"
        )
        assert lum_dist.value < 1e6, (
            "Luminosity distance should be reasonable (< 1 Gpc)"
        )
