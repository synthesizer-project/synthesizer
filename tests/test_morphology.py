"""Test suite for synthesizer.parametric.morphology module.

This module provides end-to-end and unit tests for the following classes:

- MorphologyBase
- PointSource
- Gaussian2D
- Gaussian2DAnnuli
- Sersic2D
- Sersic2DAnnuli

Includes comprehensive tests comparing against astropy counterparts.
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling import models as astropy_models
from unyt import kpc, mas, unyt_array

from synthesizer import exceptions
from synthesizer.parametric.morphology import (
    Gaussian2D,
    Gaussian2DAnnuli,
    PointSource,
    Sersic2D,
    Sersic2DAnnuli,
)

RESOLUTION = 0.1 * kpc  # Resolution for grid sampling
NPIX = (50, 50)  # Number of pixels in x and y
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)  # Cosmology for point source conversions
REDSHIFT = 0.5  # Redshift for cosmology-dependent tests


@pytest.fixture
def grid_coords():
    """Generate coordinate grids for tests.

    Returns:
        tuple: Meshgrid arrays (xx, yy) with units matching RESOLUTION.
    """
    x = unyt_array(np.linspace(-5, 5, NPIX[0]), RESOLUTION.units)
    y = unyt_array(np.linspace(-5, 5, NPIX[1]), RESOLUTION.units)
    return np.meshgrid(x, y)


class TestMorphologyBase:
    """Tests for base functionality provided by MorphologyBase."""

    def test_str_contains_class_name(self):
        """__str__ should include the class name in its output."""
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        assert "Gaussian2D".upper() in str(gauss)

    def test_get_density_grid_normalization(self):
        """get_density_grid should return a grid normalized to sum=1."""
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        grid = gauss.get_density_grid(RESOLUTION, NPIX)
        assert pytest.approx(1.0, rel=1e-6) == grid.sum()


class TestPointSource:
    """Tests for the PointSource morphology class."""

    def test_center_pixel_and_normalization(self):
        """Ensure PointSource places density in center pixel and normalizes."""
        ps = PointSource(
            offset=unyt_array([0, 0], kpc), cosmo=COSMO, redshift=REDSHIFT
        )
        grid = ps.get_density_grid(RESOLUTION, NPIX)
        assert np.count_nonzero(grid) == 1
        assert pytest.approx(1.0, rel=1e-6) == grid.sum()

    def test_offset_without_units_raises(self):
        """Initializing PointSource without units raises TypeError."""
        with pytest.raises(exceptions.MissingUnits):
            PointSource(offset=[1, 1])

    def test_mismatched_grid_units_raises(self):
        """Using resolution in mas raises InconsistentArguments."""
        ps = PointSource(offset=unyt_array([1, 1], kpc))
        factor = COSMO.kpc_proper_per_arcmin(REDSHIFT).to("kpc/mas").value
        res_mas = (RESOLUTION.to("kpc").value * factor) * mas
        with pytest.raises(exceptions.InconsistentArguments):
            ps.get_density_grid(res_mas, NPIX)


class TestGaussian2D:
    """Tests for the Gaussian2D morphology class."""

    def test_unit_mismatch_in_compute(self, grid_coords):
        """Test incompatible units to compute_density_grid."""
        xx, yy = grid_coords
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        with pytest.raises(exceptions.InconsistentUnits):
            gauss.compute_density_grid(xx, yy.value * mas)

    def test_get_density_grid_shape(self):
        """get_density_grid should return an array of shape NPIX."""
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        grid = gauss.get_density_grid(RESOLUTION, NPIX)
        assert grid.shape == NPIX


class TestGaussian2DAnnuli:
    """Tests for the Gaussian2DAnnuli morphology class."""

    def test_annulus_masking_sums(self):
        """The sum of a single annulus mask should be < than the full grid."""
        radii = unyt_array([1.0, 2.0], kpc)
        ga_ann = Gaussian2DAnnuli(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, radii=radii, rho=0
        )
        ga = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        full = ga.get_density_grid(RESOLUTION, NPIX)
        shell0 = ga_ann.get_density_grid(RESOLUTION, NPIX, annulus=0)
        assert shell0.sum() < full.sum()

    def test_outside_annulus_zero(self):
        """Regions outside the specified annulus should be zero."""
        radii = unyt_array([0.5, 1.5], kpc)
        ga_ann = Gaussian2DAnnuli(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, radii=radii, rho=0
        )
        ga = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        full = ga.get_density_grid(RESOLUTION, NPIX)
        shell1 = ga_ann.get_density_grid(RESOLUTION, NPIX, annulus=1)
        diff = full - shell1
        assert np.all(shell1[diff > 0] == 0)

    def test_invalid_annulus_index_raises(self):
        """Requesting an out-of-range annulus index should raise ValueError."""
        radii = unyt_array([1.0, 2.0], kpc)
        ga_ann = Gaussian2DAnnuli(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, radii=radii, rho=0
        )
        with pytest.raises(exceptions.InconsistentArguments):
            ga_ann.get_density_grid(RESOLUTION, NPIX, annulus=5)


class TestSersic2D:
    """Tests for the Sersic2D morphology class."""

    def test_shape_and_normalization(self):
        """get_density_grid should return normalized grid of correct shape."""
        s2 = Sersic2D(
            r_eff=3 * kpc,
            amplitude=1,
            sersic_index=1,
            x_0=0 * kpc,
            y_0=0 * kpc,
            theta=0,
            ellipticity=0,
            cosmo=COSMO,
            redshift=REDSHIFT,
        )
        grid = s2.get_density_grid(RESOLUTION, NPIX)
        print(grid)
        assert grid.shape == NPIX
        assert pytest.approx(1.0, rel=1e-6) == grid.sum()

    def test_compute_density_grid_shape(self, grid_coords):
        """Ensure compute_density_grid returns correct shape."""
        xx, yy = grid_coords
        s2 = Sersic2D(
            r_eff=3 * kpc,
            amplitude=1,
            sersic_index=1,
            x_0=0 * kpc,
            y_0=0 * kpc,
            theta=0,
            ellipticity=0,
        )
        out, _ = s2.compute_density_grid(xx, yy)
        assert out.shape == xx.shape


class TestSersic2DAnnuli:
    """Tests for the Sersic2DAnnuli morphology class."""

    def test_annulus_masking_sums(self):
        """The sum of a single annulus mask should be < than the full grid."""
        radii = unyt_array([1.0, 2.0, 3.0], kpc)
        s_ann = Sersic2DAnnuli(
            r_eff=3 * kpc,
            radii=radii,
            amplitude=1,
            sersic_index=1,
            x_0=0 * kpc,
            y_0=0 * kpc,
            theta=0,
            ellipticity=0,
            cosmo=COSMO,
            redshift=REDSHIFT,
        )
        s = Sersic2D(
            r_eff=3 * kpc,
            amplitude=1,
            sersic_index=1,
            x_0=0 * kpc,
            y_0=0 * kpc,
            theta=0,
            ellipticity=0,
            cosmo=COSMO,
            redshift=REDSHIFT,
        )
        full = s.get_density_grid(RESOLUTION, NPIX)
        shell1 = s_ann.get_density_grid(RESOLUTION, NPIX, annulus=1)
        assert shell1.sum() < full.sum()

    def test_invalid_annulus_index_raises(self):
        """Negative annulus index should raise ValueError."""
        radii = unyt_array([1.0, 2.0], kpc)
        s_ann = Sersic2DAnnuli(
            r_eff=3 * kpc,
            radii=radii,
            amplitude=1,
            sersic_index=1,
            x_0=0 * kpc,
            y_0=0 * kpc,
            theta=0,
            ellipticity=0,
            cosmo=COSMO,
            redshift=REDSHIFT,
        )
        with pytest.raises(exceptions.InconsistentArguments):
            s_ann.get_density_grid(RESOLUTION, NPIX, annulus=-1)


class TestAstropyCompatibility:
    """Test compatibility with astropy.modeling counterparts."""

    def test_gaussian2d_vs_astropy(self, grid_coords):
        """Compare Gaussian2D results with astropy counterpart."""
        xx, yy = grid_coords

        # Test parameters (in value form for astropy)
        x_mean = 0.5
        y_mean = -0.3
        x_stddev = 1.2
        y_stddev = 0.8

        # Synthesizer Gaussian2D (with units)
        synth_gauss = Gaussian2D(
            x_mean * kpc, y_mean * kpc, x_stddev * kpc, y_stddev * kpc, rho=0
        )
        synth_density, synth_norm = synth_gauss.compute_density_grid(xx, yy)

        # Astropy Gaussian2D (without units)
        astropy_gauss = astropy_models.Gaussian2D(
            amplitude=1,
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
        )
        astropy_density = astropy_gauss(xx.value, yy.value)

        # Normalize astropy result to match synthesizer
        astropy_norm = np.sum(astropy_density)
        astropy_density_normalized = astropy_density / astropy_norm
        synth_density_normalized = synth_density / synth_norm

        # Compare normalized results (should be very close)
        np.testing.assert_allclose(
            synth_density_normalized,
            astropy_density_normalized,
            rtol=1e-10,
            err_msg="Synthesizer Gaussian2D differs from astropy counterpart",
        )

    def test_sersic2d_vs_astropy(self, grid_coords):
        """Compare Sersic2D results with astropy.modeling.models.Sersic2D."""
        xx, yy = grid_coords

        # Test parameters
        amplitude = 1.5
        r_eff = 2.0
        n = 2.0
        x_0 = 0.2
        y_0 = -0.1
        ellip = 0.3
        theta = np.pi / 6

        # Synthesizer Sersic2D
        synth_sersic = Sersic2D(
            r_eff=r_eff * kpc,
            amplitude=amplitude,
            sersic_index=n,
            x_0=x_0 * kpc,
            y_0=y_0 * kpc,
            theta=theta,
            ellipticity=ellip,
        )
        synth_density, synth_norm = synth_sersic.compute_density_grid(xx, yy)

        # Astropy Sersic2D
        astropy_sersic = astropy_models.Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=n,
            x_0=x_0,
            y_0=y_0,
            ellip=ellip,
            theta=theta,
        )
        astropy_density = astropy_sersic(xx.value, yy.value)

        # Normalize both for comparison
        astropy_norm = np.sum(astropy_density)
        astropy_density_normalized = astropy_density / astropy_norm
        synth_density_normalized = synth_density / synth_norm

        # Compare normalized results
        np.testing.assert_allclose(
            synth_density_normalized,
            astropy_density_normalized,
            rtol=1e-10,
            err_msg="Synthesizer Sersic2D differs from astropy counterpart",
        )

    def test_sersic2d_special_cases_vs_astropy(self, grid_coords):
        """Test special Sersic index cases against astropy."""
        xx, yy = grid_coords

        # Test n=1 (exponential profile)
        for n in [0.5, 1.0, 2.0, 4.0, 8.0]:
            synth_sersic = Sersic2D(
                r_eff=1.5 * kpc,
                amplitude=1.0,
                sersic_index=n,
                x_0=0 * kpc,
                y_0=0 * kpc,
                theta=0,
                ellipticity=0,
            )
            synth_density, synth_norm = synth_sersic.compute_density_grid(
                xx, yy
            )

            astropy_sersic = astropy_models.Sersic2D(
                amplitude=1.0,
                r_eff=1.5,
                n=n,
                x_0=0,
                y_0=0,
                ellip=0,
                theta=0,
            )
            astropy_density = astropy_sersic(xx.value, yy.value)

            # Normalize and compare
            astropy_norm = np.sum(astropy_density)
            np.testing.assert_allclose(
                synth_density / synth_norm,
                astropy_density / astropy_norm,
                rtol=1e-10,
                err_msg=f"Sersic profiles differ for n={n}",
            )


class TestParameterValidation:
    """Test parameter validation for all morphology classes."""

    def test_gaussian2d_parameter_validation(self):
        """Test parameter validation for Gaussian2D."""
        # Valid parameters
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0)
        assert gauss.x_mean == 0 * kpc
        assert gauss.y_mean == 0 * kpc
        assert gauss.stddev_x == 1 * kpc
        assert gauss.stddev_y == 1 * kpc
        assert gauss.rho == 0

        # Test correlation coefficient bounds
        gauss_corr = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=0.5)
        assert gauss_corr.rho == 0.5

        # Test negative correlation
        gauss_neg_corr = Gaussian2D(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, rho=-0.8
        )
        assert gauss_neg_corr.rho == -0.8

    def test_sersic2d_parameter_validation(self):
        """Test parameter validation for Sersic2D."""
        # Valid parameters
        sersic = Sersic2D(
            r_eff=2 * kpc,
            amplitude=1.5,
            sersic_index=4.0,
            x_0=0.5 * kpc,
            y_0=-0.3 * kpc,
            theta=np.pi / 4,
            ellipticity=0.7,
        )
        assert sersic.r_eff == 2 * kpc
        assert sersic.amplitude == 1.5
        assert sersic.sersic_index == 4.0
        assert sersic.x_0 == 0.5 * kpc
        assert sersic.y_0 == -0.3 * kpc
        assert sersic.theta == np.pi / 4
        assert sersic.ellipticity == 0.7

        # Test fractional Sersic indices
        sersic_frac = Sersic2D(r_eff=1 * kpc, sersic_index=1.5)
        assert sersic_frac.sersic_index == 1.5

    def test_point_source_parameter_validation(self):
        """Test parameter validation for PointSource."""
        # Test with kpc offset
        ps_kpc = PointSource(offset=unyt_array([1.0, -0.5], kpc))
        assert np.allclose(ps_kpc.offset_kpc.value, [1.0, -0.5])

        # Test with mas offset
        ps_mas = PointSource(offset=unyt_array([100, -50], mas))
        assert np.allclose(ps_mas.offset_mas.value, [100, -50])

        # Test zero offset
        ps_zero = PointSource(offset=unyt_array([0, 0], kpc))
        assert np.allclose(ps_zero.offset_kpc.value, [0, 0])

    def test_annuli_parameter_validation(self):
        """Test parameter validation for annuli classes."""
        radii = unyt_array([0.5, 1.5, 3.0], kpc)

        # Gaussian2DAnnuli
        gauss_ann = Gaussian2DAnnuli(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, radii=radii
        )
        assert len(gauss_ann.radii) == len(radii) + 1  # +1 for infinity
        assert gauss_ann.n_annuli == len(radii) + 1

        # Sersic2DAnnuli
        sersic_ann = Sersic2DAnnuli(
            r_eff=2 * kpc, radii=radii, x_0=0 * kpc, y_0=0 * kpc
        )
        assert len(sersic_ann.radii) == len(radii) + 1
        assert sersic_ann.n_annuli == len(radii) + 1


class TestEdgeCases:
    """Test edge cases and extreme parameter values."""

    def test_zero_standard_deviation_gaussian(self):
        """Test Gaussian2D behavior with very small standard deviations."""
        # Small but reasonable standard deviations (very small ones lead to
        # numerical issues with extremely narrow peaks)
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 0.01 * kpc, 0.01 * kpc, rho=0)
        xx = unyt_array(np.linspace(-1, 1, 50), kpc)
        yy = unyt_array(np.linspace(-1, 1, 50), kpc)
        xx_grid, yy_grid = np.meshgrid(xx, yy)

        density, norm = gauss.compute_density_grid(xx_grid, yy_grid)
        assert np.isfinite(density).all(), "Density should be finite"
        assert norm > 0, "Normalization should be positive"

    def test_extreme_sersic_indices(self, grid_coords):
        """Test Sersic2D with extreme Sersic indices."""
        xx, yy = grid_coords

        # Very small Sersic index
        sersic_small = Sersic2D(r_eff=1 * kpc, sersic_index=0.1)
        density_small, norm_small = sersic_small.compute_density_grid(xx, yy)
        assert np.isfinite(density_small).all()
        assert norm_small > 0

        # Large Sersic index
        sersic_large = Sersic2D(r_eff=1 * kpc, sersic_index=10.0)
        density_large, norm_large = sersic_large.compute_density_grid(xx, yy)
        assert np.isfinite(density_large).all()
        assert norm_large > 0

    def test_high_ellipticity_sersic(self, grid_coords):
        """Test Sersic2D with high ellipticity."""
        xx, yy = grid_coords

        # High ellipticity (but not 1.0 to avoid division issues)
        sersic_ellip = Sersic2D(
            r_eff=1 * kpc, sersic_index=2.0, ellipticity=0.99
        )
        density, norm = sersic_ellip.compute_density_grid(xx, yy)
        assert np.isfinite(density).all()
        assert norm > 0

    def test_large_offsets(self, grid_coords):
        """Test morphologies with large coordinate offsets."""
        xx, yy = grid_coords

        # Gaussian with large offset
        gauss_offset = Gaussian2D(10 * kpc, -8 * kpc, 1 * kpc, 1 * kpc, rho=0)
        density, norm = gauss_offset.compute_density_grid(xx, yy)
        assert np.isfinite(density).all()

        # Sersic with large offset
        sersic_offset = Sersic2D(r_eff=1 * kpc, x_0=15 * kpc, y_0=-12 * kpc)
        density, norm = sersic_offset.compute_density_grid(xx, yy)
        assert np.isfinite(density).all()


class TestUnitCompatibility:
    """Test unit compatibility and conversions."""

    def test_mixed_unit_errors(self):
        """Test that mixed units raise appropriate errors."""
        # Gaussian2D with mismatched units should raise during computation
        gauss = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc)

        xx_kpc = unyt_array(np.linspace(-5, 5, 10), kpc)
        yy_mas = unyt_array(np.linspace(-5, 5, 10), mas)
        xx_grid, yy_grid = np.meshgrid(xx_kpc, yy_mas)

        with pytest.raises(exceptions.InconsistentUnits):
            gauss.compute_density_grid(xx_grid, yy_grid)

    def test_sersic_unit_mismatch_errors(self):
        """Test Sersic2D unit mismatch errors."""
        # Create Sersic with kpc units
        sersic = Sersic2D(r_eff=1 * kpc, x_0=0 * kpc, y_0=0 * kpc)

        # Try to use with mas coordinate grids
        xx_mas = unyt_array(np.linspace(-100, 100, 10), mas)
        yy_mas = unyt_array(np.linspace(-100, 100, 10), mas)
        xx_grid, yy_grid = np.meshgrid(xx_mas, yy_mas)

        with pytest.raises(exceptions.InconsistentUnits):
            sersic.compute_density_grid(xx_grid, yy_grid)

    def test_point_source_unit_consistency(self):
        """Test PointSource unit consistency requirements."""
        # PointSource with kpc offset should work with kpc grids
        ps_kpc = PointSource(offset=unyt_array([0, 0], kpc))
        xx_kpc = unyt_array(np.linspace(-2, 2, 10), kpc)
        yy_kpc = unyt_array(np.linspace(-2, 2, 10), kpc)
        xx_grid, yy_grid = np.meshgrid(xx_kpc, yy_kpc)

        density, norm = ps_kpc.compute_density_grid(xx_grid, yy_grid)
        assert np.count_nonzero(density) == 1
        assert norm == 1.0

    def test_cosmology_unit_conversions(self):
        """Test cosmology-dependent unit conversions."""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        redshift = 1.0

        # PointSource with cosmology should handle both kpc and mas
        ps = PointSource(
            offset=unyt_array([1, 0], kpc), cosmo=cosmo, redshift=redshift
        )

        # Should have both kpc and mas offsets after cosmology conversion
        assert ps.offset_kpc is not None
        assert ps.offset_mas is not None

        # Test Sersic with cosmology
        sersic = Sersic2D(r_eff=2 * kpc, cosmo=cosmo, redshift=redshift)

        # Should have both kpc and mas effective radii
        assert sersic.r_eff_kpc is not None
        assert sersic.r_eff_mas is not None


class TestNormalization:
    """Test normalization properties across all morphologies."""

    def test_gaussian_normalization_invariance(self):
        """Test that Gaussian normalization is independent of parameters."""
        test_params = [
            (0, 0, 1, 1, 0),
            (2, -1, 0.5, 2, 0.3),
            (-1, 3, 3, 0.2, -0.7),
        ]

        for x_mean, y_mean, sx, sy, rho in test_params:
            gauss = Gaussian2D(
                x_mean * kpc, y_mean * kpc, sx * kpc, sy * kpc, rho=rho
            )
            grid = gauss.get_density_grid(RESOLUTION, NPIX)
            assert pytest.approx(1.0, rel=1e-6) == grid.sum()

    def test_sersic_normalization_invariance(self):
        """Test that Sersic normalization is independent of parameters."""
        test_params = [
            (1, 1, 0, 0, 0, 0),
            (2, 4, 1, -1, 0.5, np.pi / 4),
            (0.5, 8, -2, 2, 0.8, np.pi / 2),
        ]

        for r_eff, n, x0, y0, ellip, theta in test_params:
            sersic = Sersic2D(
                r_eff=r_eff * kpc,
                sersic_index=n,
                x_0=x0 * kpc,
                y_0=y0 * kpc,
                ellipticity=ellip,
                theta=theta,
            )
            grid = sersic.get_density_grid(RESOLUTION, NPIX)
            assert pytest.approx(1.0, rel=1e-6) == grid.sum()

    def test_annuli_normalization_consistency(self):
        """Test that annuli preserve total normalization."""
        radii = unyt_array([0.5, 1.0, 2.0], kpc)

        # Gaussian annuli
        gauss_ann = Gaussian2DAnnuli(
            0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc, radii=radii
        )

        # Sum all annuli should be less than or equal to full profile
        total_ann = 0
        for i in range(gauss_ann.n_annuli - 1):
            ann_grid = gauss_ann.get_density_grid(RESOLUTION, NPIX, annulus=i)
            total_ann += ann_grid.sum()

        gauss_full = Gaussian2D(0 * kpc, 0 * kpc, 1 * kpc, 1 * kpc)
        full_grid = gauss_full.get_density_grid(RESOLUTION, NPIX)

        # Total should be close to full (within numerical precision)
        assert total_ann <= full_grid.sum() + 1e-10


class TestParameterRecovery:
    """Test that input parameters can be recovered from morphology objects."""

    def test_gaussian2d_parameter_recovery(self):
        """Test that Gaussian2D input parameters are stored and recoverable."""
        # Define test parameters
        x_mean = 1.5 * kpc
        y_mean = -0.8 * kpc
        stddev_x = 2.3 * kpc
        stddev_y = 1.1 * kpc
        rho = 0.6

        # Create Gaussian2D
        gauss = Gaussian2D(x_mean, y_mean, stddev_x, stddev_y, rho=rho)

        # Verify all parameters are recoverable
        assert gauss.x_mean == x_mean
        assert gauss.y_mean == y_mean
        assert gauss.stddev_x == stddev_x
        assert gauss.stddev_y == stddev_y
        assert gauss.rho == rho

    def test_sersic2d_parameter_recovery(self):
        """Test that Sersic2D input parameters are stored and recoverable."""
        # Define test parameters
        r_eff = 3.7 * kpc
        amplitude = 2.5
        sersic_index = 2.8
        x_0 = 0.9 * kpc
        y_0 = -1.2 * kpc
        theta = np.pi / 3
        ellipticity = 0.4

        # Create Sersic2D
        sersic = Sersic2D(
            r_eff=r_eff,
            amplitude=amplitude,
            sersic_index=sersic_index,
            x_0=x_0,
            y_0=y_0,
            theta=theta,
            ellipticity=ellipticity,
        )

        # Verify all parameters are recoverable
        assert sersic.r_eff == r_eff
        assert sersic.amplitude == amplitude
        assert sersic.sersic_index == sersic_index
        assert sersic.x_0 == x_0
        assert sersic.y_0 == y_0
        assert sersic.theta == theta
        assert sersic.ellipticity == ellipticity

    def test_point_source_parameter_recovery(self):
        """Test PointSource input parameters are stored and recoverable."""
        # Test with kpc offset
        offset_kpc = unyt_array([2.1, -1.7], kpc)
        ps_kpc = PointSource(offset=offset_kpc)

        assert np.allclose(ps_kpc.offset_kpc.value, offset_kpc.value)
        assert ps_kpc.offset_kpc.units == offset_kpc.units

        # Test with mas offset
        offset_mas = unyt_array([150, -200], mas)
        ps_mas = PointSource(offset=offset_mas)

        assert np.allclose(ps_mas.offset_mas.value, offset_mas.value)
        assert ps_mas.offset_mas.units == offset_mas.units

    def test_sersic2d_cosmology_parameter_recovery(self):
        """Test Sersic2D parameter recovery with cosmology conversions."""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        redshift = 1.5
        r_eff = 2.0 * kpc

        # Create Sersic with cosmology
        sersic = Sersic2D(r_eff=r_eff, cosmo=cosmo, redshift=redshift)

        # Original parameters should be recoverable
        assert sersic.r_eff == r_eff
        assert sersic.cosmo == cosmo
        assert sersic.redshift == redshift

        # Both kpc and mas versions should be available
        assert sersic.r_eff_kpc is not None
        assert sersic.r_eff_mas is not None
        assert sersic.r_eff_kpc == r_eff  # Original was in kpc

    def test_annuli_parameter_recovery(self):
        """Test that annuli morphology parameters are recoverable."""
        radii = unyt_array([0.5, 1.5, 3.0], kpc)

        # Gaussian2DAnnuli
        gauss_ann = Gaussian2DAnnuli(
            1 * kpc, -0.5 * kpc, 0.8 * kpc, 1.2 * kpc, radii=radii, rho=0.3
        )

        # Parent class parameters should be recoverable
        assert gauss_ann.x_mean == 1 * kpc
        assert gauss_ann.y_mean == -0.5 * kpc
        assert gauss_ann.stddev_x == 0.8 * kpc
        assert gauss_ann.stddev_y == 1.2 * kpc
        assert gauss_ann.rho == 0.3

        # Annuli-specific parameters (note: radii gets infinity appended)
        assert np.allclose(gauss_ann.radii[:-1].value, radii.value)
        assert gauss_ann.radii[:-1].units == radii.units
        assert gauss_ann.n_annuli == len(radii) + 1

        # Sersic2DAnnuli
        sersic_ann = Sersic2DAnnuli(
            r_eff=2.5 * kpc,
            radii=radii,
            amplitude=1.8,
            sersic_index=3.2,
            x_0=0.3 * kpc,
            y_0=-0.7 * kpc,
            theta=np.pi / 6,
            ellipticity=0.6,
        )

        # Parent class parameters should be recoverable
        assert sersic_ann.r_eff == 2.5 * kpc
        assert sersic_ann.amplitude == 1.8
        assert sersic_ann.sersic_index == 3.2
        assert sersic_ann.x_0 == 0.3 * kpc
        assert sersic_ann.y_0 == -0.7 * kpc
        assert sersic_ann.theta == np.pi / 6
        assert sersic_ann.ellipticity == 0.6

        # Annuli-specific parameters
        assert np.allclose(sersic_ann.radii[:-1].value, radii.value)
        assert sersic_ann.radii[:-1].units == radii.units
        assert sersic_ann.n_annuli == len(radii) + 1


class TestProfileFitting:
    """Test parameter recovery through profile fitting."""

    def test_sersic_parameter_fitting_recovery(self, grid_coords):
        """Test Sersic parameters can be recovered by fitting the profile."""
        from scipy.optimize import curve_fit

        xx, yy = grid_coords

        # Known input parameters (use simpler values for robust fitting)
        true_amplitude = 1.0
        true_r_eff = 1.5  # kpc
        true_n = 1.0  # exponential profile
        true_x_0 = 0.2  # kpc - smaller offset
        true_y_0 = -0.1  # kpc - smaller offset
        true_ellip = 0.2  # lower ellipticity
        true_theta = np.pi / 8  # smaller rotation

        # Create Sersic profile with known parameters
        sersic = Sersic2D(
            r_eff=true_r_eff * kpc,
            amplitude=true_amplitude,
            sersic_index=true_n,
            x_0=true_x_0 * kpc,
            y_0=true_y_0 * kpc,
            theta=true_theta,
            ellipticity=true_ellip,
        )

        # Generate density grid
        density, _ = sersic.compute_density_grid(xx, yy)

        # Define fitting function that matches Sersic2D implementation
        def sersic_model(coords, amplitude, r_eff, n, x_0, y_0, ellip, theta):
            x, y = coords
            # Apply coordinate transformation (same as in Sersic2D)
            a = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta)
            b = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta)

            # Compute elliptical radius
            radius = np.sqrt(a**2 + (b / (1 - ellip)) ** 2)

            # Sersic profile (same formula as in compute_density_grid)
            import scipy.special

            b_n = scipy.special.gammaincinv(2 * n, 0.5)
            profile = amplitude * np.exp(
                -b_n * ((radius / r_eff) ** (1.0 / n) - 1.0)
            )
            return profile

        # Prepare data for fitting
        coords = (xx.value.ravel(), yy.value.ravel())
        data = density.ravel()

        # Initial guess (slightly perturbed from true values)
        p0 = [
            true_amplitude * 1.1,
            true_r_eff * 0.9,
            true_n * 1.05,
            true_x_0 * 1.2,
            true_y_0 * 0.8,
            true_ellip * 1.1,
            true_theta * 0.95,
        ]

        # Fit the profile
        popt, _ = curve_fit(sersic_model, coords, data, p0=p0, maxfev=2000)

        (
            fit_amplitude,
            fit_r_eff,
            fit_n,
            fit_x_0,
            fit_y_0,
            fit_ellip,
            fit_theta,
        ) = popt

        # Check parameter recovery (allow tolerance for numerical fitting)
        assert abs(fit_amplitude - true_amplitude) < 0.1, (
            f"Amplitude: expected {true_amplitude}, got {fit_amplitude}"
        )
        assert abs(fit_r_eff - true_r_eff) < 0.2, (
            f"r_eff: expected {true_r_eff}, got {fit_r_eff}"
        )
        assert abs(fit_n - true_n) < 0.2, (
            f"Sersic index: expected {true_n}, got {fit_n}"
        )
        assert abs(fit_x_0 - true_x_0) < 0.2, (
            f"x_0: expected {true_x_0}, got {fit_x_0}"
        )
        assert abs(fit_y_0 - true_y_0) < 0.2, (
            f"y_0: expected {true_y_0}, got {fit_y_0}"
        )
        assert abs(fit_ellip - true_ellip) < 0.1, (
            f"Ellipticity: expected {true_ellip}, got {fit_ellip}"
        )
        assert abs(fit_theta - true_theta) < 0.2, (
            f"Theta: expected {true_theta}, got {fit_theta}"
        )

    def test_gaussian_parameter_fitting_recovery(self, grid_coords):
        """Test Gaussian parameters can be recovered by fitting the profile."""
        from scipy.optimize import curve_fit

        xx, yy = grid_coords

        # Known input parameters (use simpler values for robust fitting)
        true_x_mean = 0.3  # kpc - smaller offset
        true_y_mean = -0.2  # kpc - smaller offset
        true_stddev_x = 1.2  # kpc
        true_stddev_y = 1.0  # kpc
        true_rho = 0.2  # lower correlation

        # Create Gaussian profile with known parameters
        gauss = Gaussian2D(
            true_x_mean * kpc,
            true_y_mean * kpc,
            true_stddev_x * kpc,
            true_stddev_y * kpc,
            rho=true_rho,
        )

        # Generate density grid
        density, _ = gauss.compute_density_grid(xx, yy)

        # Define fitting function that matches Gaussian2D implementation
        def gaussian_model(coords, x_mean, y_mean, stddev_x, stddev_y, rho):
            x, y = coords

            # Covariance matrix (same as in Gaussian2D)
            cov_mat = np.array(
                [
                    [stddev_x**2, rho * stddev_x * stddev_y],
                    [rho * stddev_x * stddev_y, stddev_y**2],
                ]
            )

            # Invert covariance matrix
            inv_cov = np.linalg.inv(cov_mat)
            det_cov = np.linalg.det(cov_mat)

            # Position deviations
            dx = x - x_mean
            dy = y - y_mean

            # Coefficient
            coeff = 1 / (2 * np.pi * np.sqrt(det_cov))

            # Exponent calculation
            exp_term = (
                dx**2 * inv_cov[0, 0]
                + 2 * dx * dy * inv_cov[0, 1]
                + dy**2 * inv_cov[1, 1]
            )

            return coeff * np.exp(-0.5 * exp_term)

        # Prepare data for fitting
        coords = (xx.value.ravel(), yy.value.ravel())
        data = density.ravel()

        # Initial guess (slightly perturbed from true values)
        p0 = [
            true_x_mean * 1.1,
            true_y_mean * 0.9,
            true_stddev_x * 1.05,
            true_stddev_y * 0.95,
            true_rho * 1.1,
        ]

        # Fit the profile
        popt, _ = curve_fit(gaussian_model, coords, data, p0=p0, maxfev=2000)

        fit_x_mean, fit_y_mean, fit_stddev_x, fit_stddev_y, fit_rho = popt

        # Check parameter recovery
        assert abs(fit_x_mean - true_x_mean) < 0.1, (
            f"x_mean: expected {true_x_mean}, got {fit_x_mean}"
        )
        assert abs(fit_y_mean - true_y_mean) < 0.1, (
            f"y_mean: expected {true_y_mean}, got {fit_y_mean}"
        )
        assert abs(fit_stddev_x - true_stddev_x) < 0.1, (
            f"stddev_x: expected {true_stddev_x}, got {fit_stddev_x}"
        )
        assert abs(fit_stddev_y - true_stddev_y) < 0.1, (
            f"stddev_y: expected {true_stddev_y}, got {fit_stddev_y}"
        )
        assert abs(fit_rho - true_rho) < 0.1, (
            f"rho: expected {true_rho}, got {fit_rho}"
        )

    def test_simple_sersic_cases_fitting(self, grid_coords):
        """Test parameter recovery for simple Sersic cases (centered)."""
        from scipy.optimize import curve_fit

        xx, yy = grid_coords

        # Test simple cases that should fit reliably
        test_cases = [
            {"r_eff": 1.5, "n": 1.0, "amp": 1.0},  # Exponential
            {"r_eff": 2.0, "n": 4.0, "amp": 1.0},  # de Vaucouleurs
        ]

        for case in test_cases:
            true_r_eff = case["r_eff"]
            true_n = case["n"]
            true_amp = case["amp"]

            # Create simple centered Sersic profile
            sersic = Sersic2D(
                r_eff=true_r_eff * kpc,
                amplitude=true_amp,
                sersic_index=true_n,
                x_0=0 * kpc,
                y_0=0 * kpc,
                theta=0,
                ellipticity=0,
            )

            density, _ = sersic.compute_density_grid(xx, yy)

            # Simple 3-parameter model for centered, circular profiles
            def simple_sersic(coords, amplitude, r_eff, n):
                x, y = coords
                radius = np.sqrt(x**2 + y**2)
                import scipy.special

                b_n = scipy.special.gammaincinv(2 * n, 0.5)
                return amplitude * np.exp(
                    -b_n * ((radius / r_eff) ** (1.0 / n) - 1.0)
                )

            coords = (xx.value.ravel(), yy.value.ravel())
            data = density.ravel()

            # Use better initial guess for simple cases
            p0 = [true_amp, true_r_eff, true_n]

            popt, _ = curve_fit(
                simple_sersic, coords, data, p0=p0, maxfev=1000
            )
            fit_amp, fit_r_eff, fit_n = popt

            # Stricter tolerance for simple cases
            assert abs(fit_amp - true_amp) < 0.05, (
                f"Case {case}: Amplitude exp {true_amp}, got {fit_amp}"
            )
            assert abs(fit_r_eff - true_r_eff) < 0.1, (
                f"Case {case}: r_eff exp {true_r_eff}, got {fit_r_eff}"
            )
            assert abs(fit_n - true_n) < 0.1, (
                f"Case {case}: Sersic index expected {true_n}, got {fit_n}"
            )
