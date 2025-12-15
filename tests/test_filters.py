"""A comprehensive test suite for Filter and FilterCollection classes.

This module contains tests for all Filter and FilterCollection functionality
including:
- FilterCollection addition with different wavelength array types
- Wavelength array handling and resampling
- Filter interpolation and transmission preservation
- Top-hat filter creation and manipulation
- Edge cases in filter combination
- HDF5 I/O operations
- Filter properties and calculations
- FilterCollection operations (select, find_filter, etc.)
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from unyt import Hz, angstrom, erg, s

from synthesizer import exceptions
from synthesizer.instruments import Filter, FilterCollection
from synthesizer.instruments.filters import UVJ


@pytest.fixture
def lam_log():
    """Return a logarithmically spaced wavelength array (UV to IR)."""
    return np.logspace(np.log10(1000), np.log10(1e5), 1000) * angstrom


@pytest.fixture
def lam_linear():
    """Return a linearly spaced wavelength array (UV to optical)."""
    return np.linspace(1000, 5000, 500) * angstrom


@pytest.fixture
def lam_wide():
    """Return a wide wavelength range."""
    return np.linspace(1000, 10000, 1000) * angstrom


@pytest.fixture
def lam_narrow():
    """Return a narrow wavelength range within the wide range."""
    return np.linspace(2000, 5000, 500) * angstrom


class TestFilterCollectionAddition:
    """Tests for FilterCollection addition operations."""

    def test_filter_addition_with_logspaced_wavelengths(
        self, lam_log, lam_linear
    ):
        """Test adding FilterCollections with logarithmic wavelengths.

        This tests the fix for the bug where adding two FilterCollections
        with log-spaced wavelength arrays (like those from grids) would
        result in filters losing their transmission due to incorrect
        linear interpolation across many orders of magnitude.
        """
        # Create first FilterCollection with tophat filters on log spacing
        tophat_dict_1 = {
            "UV1500": {"lam_eff": 1500 * angstrom, "lam_fwhm": 300 * angstrom},
            "UV2800": {"lam_eff": 2800 * angstrom, "lam_fwhm": 300 * angstrom},
        }
        fc1 = FilterCollection(tophat_dict=tophat_dict_1, new_lam=lam_log)

        # Create second FilterCollection with different tophat filters
        tophat_dict_2 = {
            "Optical3500": {
                "lam_eff": 3500 * angstrom,
                "lam_fwhm": 500 * angstrom,
            },
            "Optical4500": {
                "lam_eff": 4500 * angstrom,
                "lam_fwhm": 500 * angstrom,
            },
        }
        fc2 = FilterCollection(tophat_dict=tophat_dict_2, new_lam=lam_linear)

        # Verify initial filters have non-zero transmission
        for f in fc1:
            assert np.sum(f.t > 0) > 0, (
                f"{f.filter_code} should have non-zero transmission before "
                "addition"
            )
        for f in fc2:
            assert np.sum(f.t > 0) > 0, (
                f"{f.filter_code} should have non-zero transmission before "
                "addition"
            )

        # Add the two FilterCollections
        fc_combined = fc1 + fc2

        # Verify combined FilterCollection has all filters
        assert fc_combined.nfilters == 4
        assert "UV1500" in fc_combined
        assert "UV2800" in fc_combined
        assert "Optical3500" in fc_combined
        assert "Optical4500" in fc_combined

        # CRITICAL TEST: All filters should still have non-zero transmission
        # after addition and resampling
        for f in fc_combined:
            n_nonzero = np.sum(f.t > 0)
            assert n_nonzero > 0, (
                f"{f.filter_code} has zero transmission after addition! "
                f"This indicates the interpolation failed."
            )
            # Also verify reasonable number of non-zero points
            assert n_nonzero > 10, (
                f"{f.filter_code} has only {n_nonzero} non-zero transmission "
                "points, which is suspiciously low"
            )

    def test_filter_addition_identical_wavelengths(self, lam_linear):
        """Test that adding FilterCollections with identical wavelengths."""
        tophat_dict_1 = {
            "filter1": {"lam_eff": 2000 * angstrom, "lam_fwhm": 300 * angstrom}
        }
        tophat_dict_2 = {
            "filter2": {"lam_eff": 3000 * angstrom, "lam_fwhm": 300 * angstrom}
        }

        fc1 = FilterCollection(tophat_dict=tophat_dict_1, new_lam=lam_linear)
        fc2 = FilterCollection(tophat_dict=tophat_dict_2, new_lam=lam_linear)

        fc_combined = fc1 + fc2

        # Should have both filters
        assert fc_combined.nfilters == 2
        # Wavelength array should be the same
        assert np.allclose(fc_combined._lam, lam_linear.value)

    def test_filter_addition_one_covers_other(self, lam_wide, lam_narrow):
        """Test adding FilterCollections where 1 wavelength are is enough."""
        tophat_dict_wide = {
            "wide_filter": {
                "lam_eff": 5000 * angstrom,
                "lam_fwhm": 1000 * angstrom,
            }
        }
        tophat_dict_narrow = {
            "narrow_filter": {
                "lam_eff": 3500 * angstrom,
                "lam_fwhm": 500 * angstrom,
            }
        }

        fc_wide = FilterCollection(
            tophat_dict=tophat_dict_wide, new_lam=lam_wide
        )
        fc_narrow = FilterCollection(
            tophat_dict=tophat_dict_narrow, new_lam=lam_narrow
        )

        # Add narrow to wide
        fc_combined = fc_wide + fc_narrow

        # Should use the wide wavelength array since it covers everything
        assert fc_combined.nfilters == 2
        # Both filters should have non-zero transmission
        for f in fc_combined:
            assert np.sum(f.t > 0) > 0

    def test_filter_addition_with_none_wavelengths(self, lam_linear):
        """Test adding FilterCollections where one has None wavelength."""
        # First collection with wavelength array
        tophat_dict_1 = {
            "filter1": {"lam_eff": 2000 * angstrom, "lam_fwhm": 300 * angstrom}
        }
        fc1 = FilterCollection(tophat_dict=tophat_dict_1, new_lam=lam_linear)

        # Second collection without explicit wavelength array
        tophat_dict_2 = {
            "filter2": {"lam_eff": 3000 * angstrom, "lam_fwhm": 300 * angstrom}
        }
        fc2 = FilterCollection(tophat_dict=tophat_dict_2)

        # Add them
        fc_combined = fc1 + fc2

        # Should have both filters
        assert fc_combined.nfilters == 2
        # Both should have non-zero transmission
        for f in fc_combined:
            assert np.sum(f.t > 0) > 0

    def test_filter_addition_disjoint_ranges(self):
        """Test adding FilterCollections with non-overlapping wavelengths.

        This should create a new combined wavelength array.
        """
        # UV range
        lam_uv = np.linspace(1000, 3000, 500) * angstrom
        # IR range (non-overlapping)
        lam_ir = np.linspace(8000, 15000, 500) * angstrom

        tophat_dict_uv = {
            "UV_filter": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 500 * angstrom,
            }
        }
        tophat_dict_ir = {
            "IR_filter": {
                "lam_eff": 10000 * angstrom,
                "lam_fwhm": 1000 * angstrom,
            }
        }

        fc_uv = FilterCollection(tophat_dict=tophat_dict_uv, new_lam=lam_uv)
        fc_ir = FilterCollection(tophat_dict=tophat_dict_ir, new_lam=lam_ir)

        fc_combined = fc_uv + fc_ir

        # Should create new combined wavelength array
        assert fc_combined.nfilters == 2
        # New wavelength array should span both ranges
        assert fc_combined.lam.min() <= lam_uv.min()
        assert fc_combined.lam.max() >= lam_ir.max()
        # Both filters should have non-zero transmission
        for f in fc_combined:
            assert np.sum(f.t > 0) > 0


class TestFilterCreation:
    """Tests for Filter creation and basic functionality."""

    def test_tophat_filter_creation(self):
        """Test creating a top-hat filter."""
        filt = Filter(
            "test_tophat",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )
        assert filt.filter_code == "test_tophat"
        assert filt.filter_type == "TopHat"
        assert np.sum(filt.t > 0) > 0

    def test_tophat_filter_with_new_lam(self, lam_wide):
        """Test creating a top-hat filter with a new wavelength array."""
        filt = Filter(
            "test_tophat",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
            new_lam=lam_wide,
        )
        assert filt.filter_code == "test_tophat"
        assert np.allclose(filt._lam, lam_wide.value)
        assert np.sum(filt.t > 0) > 0


class TestFilterInterpolation:
    """Tests for Filter interpolation functionality."""

    def test_filter_interpolation_preserves_transmission(self, lam_log):
        """Test that interpolating a filter preserves non-zero transmission."""
        # Create a top-hat filter
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        # Store original non-zero count
        original_nonzero = np.sum(filt.t > 0)
        assert original_nonzero > 0

        # Interpolate onto a new wavelength array (log-spaced)
        filt._interpolate_wavelength(new_lam=lam_log)

        # Should still have non-zero transmission
        new_nonzero = np.sum(filt.t > 0)
        assert new_nonzero > 0, (
            "Filter lost all transmission after interpolation"
        )


class TestFilterCollectionInitialization:
    """Tests for FilterCollection initialization."""

    def test_empty_filter_collection(self):
        """Test creating an empty FilterCollection."""
        fc = FilterCollection()
        assert fc.lam is None
        assert len(fc.filters) == 0
        # nfilters is only set after filters are added
        assert not hasattr(fc, "nfilters") or fc.nfilters == 0

    def test_filter_collection_with_generic_dict(self, lam_linear):
        """Test creating FilterCollection with generic_dict."""
        transmission = np.ones(len(lam_linear))
        transmission[100:200] = 0.5  # Some structure

        generic_dict = {
            "test_generic": transmission,
        }

        fc = FilterCollection(generic_dict=generic_dict, new_lam=lam_linear)

        assert fc.nfilters == 1
        assert "test_generic" in fc
        assert fc["test_generic"].filter_type == "Generic"
        assert np.allclose(fc["test_generic"].t, transmission)

    def test_uvj_function(self):
        """Test the UVJ convenience function."""
        # UVJ filters have wide wavelength range (U~3650Å, V~5510Å, J~12200Å)
        # Use a wide wavelength array to accommodate all filters
        lam_wide = np.linspace(1000, 15000, 1000) * angstrom
        fc = UVJ(new_lam=lam_wide)

        assert fc.nfilters == 3
        assert "U" in fc
        assert "V" in fc
        assert "J" in fc


class TestFilterCollectionOperations:
    """Tests for FilterCollection operations."""

    def test_len(self, lam_linear):
        """Test __len__ method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        assert len(fc) == 2

    def test_str(self, lam_linear):
        """Test __str__ method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        str_repr = str(fc)
        assert "FILTERCOLLECTION" in str_repr
        assert "filter1" in str_repr

    def test_iteration(self, lam_linear):
        """Test __iter__ and __next__ methods."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # Test iteration
        filters = [f for f in fc]
        assert len(filters) == 2
        assert all(isinstance(f, Filter) for f in filters)

    def test_getitem(self, lam_linear):
        """Test __getitem__ method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        filt = fc["filter1"]
        assert isinstance(filt, Filter)
        assert filt.filter_code == "filter1"

    def test_contains(self, lam_linear):
        """Test __contains__ method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        assert "filter1" in fc
        assert "filter2" not in fc

    def test_equality(self, lam_linear):
        """Test __eq__ and __ne__ methods."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }

        fc1 = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)
        fc2 = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)
        fc3 = FilterCollection(
            tophat_dict={
                "filter2": {
                    "lam_eff": 3000 * angstrom,
                    "lam_fwhm": 300 * angstrom,
                }
            },
            new_lam=lam_linear,
        )

        assert fc1 == fc2
        assert fc1 != fc3
        assert not (fc1 == fc3)
        assert not (fc1 != fc2)

    def test_select(self, lam_linear):
        """Test select method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter3": {
                "lam_eff": 4000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # Select subset of filters
        fc_subset = fc.select("filter1", "filter3")

        assert fc_subset.nfilters == 2
        assert "filter1" in fc_subset
        assert "filter3" in fc_subset
        assert "filter2" not in fc_subset

    def test_get_non_zero_lam_lims(self, lam_linear):
        """Test get_non_zero_lam_lims method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 4000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        min_lam, max_lam = fc.get_non_zero_lam_lims()

        # Check that limits span both filters
        assert min_lam < 2000 * angstrom
        assert max_lam > 4000 * angstrom

    def test_calc_pivot_lams(self, lam_linear):
        """Test calc_pivot_lams method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        pivot_lams = fc.calc_pivot_lams()

        assert len(pivot_lams) == 2
        assert pivot_lams[0] < pivot_lams[1]  # Should be ordered

    def test_calc_mean_lams(self, lam_linear):
        """Test calc_mean_lams method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        mean_lams = fc.calc_mean_lams()

        assert len(mean_lams) == 2
        assert mean_lams[0] < mean_lams[1]

    def test_pivot_lams_property(self, lam_linear):
        """Test pivot_lams property."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        pivot_lams = fc.pivot_lams
        assert len(pivot_lams) == 1

    def test_mean_lams_property(self, lam_linear):
        """Test mean_lams property."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        mean_lams = fc.mean_lams
        assert len(mean_lams) == 1

    def test_find_filter_pivot(self, lam_linear):
        """Test find_filter method with pivot method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # Should find filter with pivot closest to 2100 Å
        filt = fc.find_filter(2100 * angstrom, method="pivot")
        assert filt.filter_code == "filter1"

    def test_find_filter_mean(self, lam_linear):
        """Test find_filter method with mean method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        filt = fc.find_filter(2900 * angstrom, method="mean")
        assert filt.filter_code == "filter2"

    def test_find_filter_transmission(self, lam_linear):
        """Test find_filter method with transmission method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # Should find filter with transmission at 2000 Å
        filt = fc.find_filter(2000 * angstrom, method="transmission")
        assert filt.filter_code == "filter1"

    def test_find_filter_with_redshift(self, lam_linear):
        """Test find_filter method with redshift."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 4000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # At z=1, rest-frame 2000 Å is observed at 4000 Å
        filt = fc.find_filter(2000 * angstrom, redshift=1.0, method="pivot")
        assert filt.filter_code == "filter2"

    def test_find_filter_invalid_method(self, lam_linear):
        """Test find_filter with invalid method raises error."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        with pytest.raises(exceptions.InconsistentArguments):
            fc.find_filter(2000 * angstrom, method="invalid")

    def test_resample_filters_with_lam_size(self):
        """Test resample_filters with lam_size parameter."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict)

        # Resample with fixed size
        fc.resample_filters(lam_size=500, verbose=False)

        assert fc.lam.size == 500

    def test_resample_filters_with_fill_gaps(self):
        """Test resample_filters with fill_gaps parameter."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 5000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict)

        # Resample with gap filling
        fc.resample_filters(fill_gaps=True, verbose=False)

        # Should have wavelength array spanning both filters
        assert fc.lam.min() < 2000 * angstrom
        assert fc.lam.max() > 5000 * angstrom

    def test_plot_transmission_curves(self, lam_linear):
        """Test plot_transmission_curves method."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        # Create plot
        fig, ax = fc.plot_transmission_curves()

        assert isinstance(fig, plt.Figure)
        assert ax is not None

        plt.close(fig)


class TestFilterCollectionHDF5:
    """Tests for FilterCollection HDF5 I/O operations."""

    def test_write_and_load_filters(self, lam_linear):
        """Test writing and loading FilterCollection to/from HDF5."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
            "filter2": {
                "lam_eff": 3000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc_original = FilterCollection(
            tophat_dict=tophat_dict, new_lam=lam_linear
        )

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fc_original.write_filters(tmp_path)

            # Load back
            fc_loaded = FilterCollection(path=tmp_path)

            # Verify
            assert fc_loaded.nfilters == fc_original.nfilters
            assert fc_loaded.filter_codes == fc_original.filter_codes
            assert np.allclose(fc_loaded._lam, fc_original._lam)

            for fcode in fc_original.filter_codes:
                assert np.allclose(fc_loaded[fcode].t, fc_original[fcode].t)
        finally:
            Path(tmp_path).unlink()


class TestFilterCreationExtended:
    """Extended tests for Filter creation."""

    def test_tophat_with_lam_min_max(self):
        """Test creating a top-hat filter with lam_min and lam_max."""
        filt = Filter(
            "test_tophat",
            lam_min=4000 * angstrom,
            lam_max=6000 * angstrom,
        )

        assert filt.filter_type == "TopHat"
        # Check transmission is 1 inside range, 0 outside
        assert np.all(filt.t[(filt._lam > 4000) & (filt._lam < 6000)] == 1.0)
        assert filt.t[filt._lam < 4000].sum() == 0
        assert filt.t[filt._lam > 6000].sum() == 0

    def test_generic_filter_creation(self, lam_linear):
        """Test creating a generic filter."""
        transmission = np.exp(-((lam_linear.value - 3000) ** 2) / (2 * 500**2))

        filt = Filter(
            "test_generic",
            transmission=transmission,
            new_lam=lam_linear,
        )

        assert filt.filter_type == "Generic"
        assert np.allclose(filt.t, transmission)

    def test_filter_add_filter(self):
        """Test adding two Filter objects."""
        filt1 = Filter(
            "filter1",
            lam_eff=2000 * angstrom,
            lam_fwhm=300 * angstrom,
        )
        filt2 = Filter(
            "filter2",
            lam_eff=3000 * angstrom,
            lam_fwhm=300 * angstrom,
        )

        fc = filt1 + filt2

        assert isinstance(fc, FilterCollection)
        assert fc.nfilters == 2
        assert "filter1" in fc
        assert "filter2" in fc

    def test_filter_str(self):
        """Test Filter __str__ method."""
        filt = Filter(
            "test_filter",
            lam_eff=2000 * angstrom,
            lam_fwhm=300 * angstrom,
        )

        str_repr = str(filt)
        assert "FILTER" in str_repr
        assert "test_filter" in str_repr

    def test_filter_transmission_property(self):
        """Test Filter transmission property."""
        filt = Filter(
            "test_filter",
            lam_eff=2000 * angstrom,
            lam_fwhm=300 * angstrom,
        )

        # transmission property should return same as t
        assert np.allclose(filt.transmission, filt.t)

    def test_clip_transmission(self, lam_linear):
        """Test clip_transmission method."""
        # Create transmission with out-of-range values
        transmission = np.ones(len(lam_linear)) * 1.5  # Above 1
        transmission[0] = -0.1  # Below 0

        filt = Filter(
            "test_filter",
            transmission=transmission,
            new_lam=lam_linear,
        )

        # Should be clipped to [0, 1]
        assert np.all(filt.t >= 0)
        assert np.all(filt.t <= 1)


class TestFilterProperties:
    """Tests for Filter property calculations."""

    def test_pivwv(self):
        """Test pivwv (pivot wavelength) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        piv = filt.pivwv()

        # Pivot should be near effective wavelength for top-hat
        assert 4500 * angstrom < piv < 5500 * angstrom

    def test_pivT(self):
        """Test pivT (transmission at pivot) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        piv_t = filt.pivT()

        # For top-hat, should be 1.0 if pivot is inside range
        assert 0 < piv_t <= 1.0

    def test_meanwv(self):
        """Test meanwv (mean wavelength) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        mean = filt.meanwv()

        # Mean should be near effective wavelength for top-hat
        assert 4500 * angstrom < mean < 5500 * angstrom

    def test_bandw(self):
        """Test bandw (bandwidth) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        bw = filt.bandw()

        # Bandwidth should be positive
        assert bw > 0 * angstrom

    def test_fwhm(self):
        """Test fwhm (full width half maximum) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        fwhm = filt.fwhm()

        # FWHM should be related to bandwidth
        assert fwhm > 0 * angstrom

    def test_Tpeak(self):
        """Test Tpeak (peak transmission) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        tpeak = filt.Tpeak()

        # For top-hat, peak should be 1.0
        assert tpeak == 1.0

    def test_rectw(self):
        """Test rectw (rectangular width) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        rw = filt.rectw()

        # Rectangular width should be positive
        assert rw > 0

    def test_max(self):
        """Test max (maximum wavelength with T>0.01) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        max_lam = filt.max()

        # Should be near upper limit of filter
        assert max_lam > 5000 * angstrom

    def test_min(self):
        """Test min (minimum wavelength with T>0.01) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        min_lam = filt.min()

        # Should be near lower limit of filter
        assert min_lam < 5000 * angstrom

    def test_mnmx(self):
        """Test mnmx (min and max wavelengths) calculation."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        min_lam, max_lam = filt.mnmx()

        # Min should be less than max
        assert min_lam < max_lam


class TestFilterApply:
    """Tests for Filter.apply_filter method."""

    def test_apply_filter_basic(self, lam_linear):
        """Test basic apply_filter functionality."""
        filt = Filter(
            "test_filter",
            lam_eff=3000 * angstrom,
            lam_fwhm=500 * angstrom,
            new_lam=lam_linear,
        )

        # Create a flat spectrum
        spectrum = np.ones(len(lam_linear))

        # Apply filter with explicit integration method
        result = filt.apply_filter(
            spectrum, lam=lam_linear, integration_method="trapz"
        )

        # Should return a scalar for 1D spectrum
        assert np.isscalar(result) or result.shape == ()

    def test_apply_filter_with_units(self, lam_linear):
        """Test apply_filter with unyt arrays."""
        filt = Filter(
            "test_filter",
            lam_eff=3000 * angstrom,
            lam_fwhm=500 * angstrom,
            new_lam=lam_linear,
        )

        # Create spectrum with units
        spectrum = np.ones(len(lam_linear)) * erg / s / Hz

        # Apply filter with explicit integration method
        result = filt.apply_filter(
            spectrum, lam=lam_linear, integration_method="trapz"
        )

        assert result > 0

    def test_apply_filter_multidimensional(self, lam_linear):
        """Test apply_filter with 2D spectrum array."""
        filt = Filter(
            "test_filter",
            lam_eff=3000 * angstrom,
            lam_fwhm=500 * angstrom,
            new_lam=lam_linear,
        )

        # Create 2D spectrum array (e.g., multiple objects)
        n_objects = 10
        spectrum = np.ones((n_objects, len(lam_linear)))

        # Apply filter with explicit integration method
        result = filt.apply_filter(
            spectrum, lam=lam_linear, integration_method="trapz"
        )

        # Should return array with one value per object
        assert result.shape == (n_objects,)

    def test_apply_filter_with_frequencies(self, lam_linear):
        """Test apply_filter using frequencies instead of wavelengths."""
        filt = Filter(
            "test_filter",
            lam_eff=3000 * angstrom,
            lam_fwhm=500 * angstrom,
            new_lam=lam_linear,
        )

        # Create spectrum and frequency array
        spectrum = np.ones(len(lam_linear))
        nu = (3e18 / lam_linear.value) * Hz  # c/lambda

        # Apply filter with frequencies and explicit integration method
        result = filt.apply_filter(spectrum, nu=nu, integration_method="trapz")

        assert np.isscalar(result) or result.shape == ()


class TestFilterCollectionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_filter_collection_add_invalid_type(self, lam_linear):
        """Test that adding invalid type raises error."""
        tophat_dict = {
            "filter1": {
                "lam_eff": 2000 * angstrom,
                "lam_fwhm": 300 * angstrom,
            },
        }
        fc = FilterCollection(tophat_dict=tophat_dict, new_lam=lam_linear)

        with pytest.raises(exceptions.InconsistentAddition):
            fc + "invalid"

    def test_filter_invalid_combination(self):
        """Test that invalid Filter arguments raise error."""
        with pytest.raises(exceptions.InconsistentArguments):
            # No valid combination of arguments
            Filter("test_filter")

    def test_filter_interpolation_zero_transmission(self):
        """Test that interpolation with zero result raises error."""
        filt = Filter(
            "test_filter",
            lam_eff=5000 * angstrom,
            lam_fwhm=1000 * angstrom,
        )

        # Try to interpolate onto wavelength range that doesn't overlap
        new_lam = np.linspace(100, 500, 100) * angstrom

        with pytest.raises(exceptions.InconsistentWavelengths):
            filt._interpolate_wavelength(new_lam=new_lam)
