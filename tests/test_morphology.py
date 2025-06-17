"""Test suite for synthesizer.parametric.morphology module.

This module provides end-to-end and unit tests for the following classes:

- MorphologyBase
- PointSource
- Gaussian2D
- Gaussian2DAnnuli
- Sersic2D
- Sersic2DAnnuli
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
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
