"""A comprehensive test suite for the Grid class.

This module contains tests for all Grid functionality including:
- Grid initialization and basic properties
- Axis handling and attribute access
- Wavelength and spectral manipulation
- Grid reduction methods
- Line handling
- SED and line extraction
- Utility methods
"""

import numpy as np
import pytest
from unyt import Hz, angstrom, erg, s

from synthesizer import exceptions
from synthesizer.grid import Grid, Template
from synthesizer.instruments.filters import UVJ


@pytest.fixture
def test_grid_name():
    """Return the test grid name."""
    return "test_grid.hdf5"


class TestGridInitialization:
    """Tests for Grid initialization and basic properties."""

    def test_grid_returned(self, test_grid):
        """Test that a Grid object is returned."""
        assert isinstance(test_grid, Grid)

    def test_grid_basic_properties(self, test_grid):
        """Test basic Grid properties."""
        assert test_grid.grid_name is not None
        assert test_grid.grid_filename is not None
        assert test_grid.naxes > 0
        assert len(test_grid.axes) == test_grid.naxes
        assert test_grid.has_spectra or test_grid.has_lines

    def test_grid_shape_properties(self, test_grid):
        """Test Grid shape and dimension properties."""
        shape = test_grid.shape
        assert len(shape) == test_grid.ndim
        assert test_grid.ndim == test_grid.naxes + 1  # +1 for wavelength

        if test_grid.has_spectra:
            assert test_grid.nlam == len(test_grid.lam)
            assert shape[-1] == test_grid.nlam

    def test_grid_with_ignore_spectra(self, test_grid_name):
        """Test Grid initialization with ignore_spectra=True."""
        grid = Grid(test_grid_name, ignore_spectra=True)
        assert not grid.has_spectra
        assert len(grid.spectra) == 0
        assert grid.lam is None

    def test_grid_with_ignore_lines(self, test_grid_name):
        """Test Grid initialization with ignore_lines=True."""
        grid = Grid(test_grid_name, ignore_lines=True)
        assert not grid.lines_available
        assert not grid.has_lines
        assert len(grid.line_lums) == 0
        assert len(grid.line_conts) == 0


class TestGridAxes:
    """Tests for Grid axis handling and attribute access."""

    def test_grid_axes(self, test_grid):
        """Test that the axes are returned correctly."""
        # Test the automatic extraction defined by the __getattr__ overload
        assert getattr(test_grid.ages, "units", None) is not None
        assert getattr(test_grid.metallicities, "units", None) is not None
        assert isinstance(test_grid._ages, np.ndarray)
        assert np.allclose(test_grid._ages, 10**test_grid.log10ages)

        # Test the silly singular and plural nonsense works (hopefully we can
        # remove this in the future)
        assert np.allclose(test_grid._ages, test_grid._age)
        assert np.allclose(test_grid._metallicities, test_grid._metallicity)
        assert np.allclose(test_grid.log10ages, test_grid.log10age)

    def test_axes_values_property(self, test_grid):
        """Test the axes_values property."""
        axes_values = test_grid.axes_values
        assert isinstance(axes_values, dict)
        for axis in test_grid.axes:
            assert axis in axes_values
            assert isinstance(axes_values[axis], np.ndarray)

    def test_get_flattened_axes_values(self, test_grid):
        """Test getting flattened axes values."""
        flattened = test_grid.get_flattened_axes_values()
        assert isinstance(flattened, dict)

        # Check that flattened arrays have the right size
        expected_size = np.prod(
            test_grid.shape[:-1]
        )  # Exclude wavelength axis
        for axis in test_grid.axes:
            assert axis in flattened
            assert len(flattened[axis]) == expected_size

    def test_get_grid_point(self, test_grid):
        """Test grid point identification."""
        # Get the first axis and pick a value
        first_axis = test_grid.axes[0]
        axis_values = getattr(test_grid, first_axis)
        test_value = axis_values[len(axis_values) // 2]  # Middle value

        # Get grid point
        kwargs = {first_axis: test_value}
        indices = test_grid.get_grid_point(**kwargs)

        assert isinstance(indices, tuple)
        assert len(indices) == test_grid.naxes
        assert isinstance(
            indices[0], (int, np.integer)
        )  # First should be an index

        # Other indices should be slices if not specified
        for i in range(1, len(indices)):
            assert isinstance(indices[i], slice)

    def test_get_nearest_index(self, test_grid):
        """Test getting nearest index in an array."""
        # Test with a simple array
        test_array = np.array([1, 3, 5, 7, 9])

        # Test exact match
        assert Grid.get_nearest_index(5, test_array) == 2

        # Test nearest match
        assert Grid.get_nearest_index(4, test_array) == 1  # Closer to 3
        assert Grid.get_nearest_index(6, test_array) == 2  # Closer to 5

        # Test with units
        if hasattr(test_grid, "ages"):
            ages = test_grid.ages
            test_age = ages[len(ages) // 2]
            index = Grid.get_nearest_index(test_age, ages)
            assert isinstance(index, (int, np.integer))
            assert 0 <= index < len(ages)


class TestGridSpectra:
    """Tests for Grid spectra handling."""

    def test_spectra_properties(self, test_grid):
        """Test spectra-related properties."""
        if test_grid.has_spectra:
            assert len(test_grid.available_spectra) > 0
            assert len(test_grid.available_spectra_emissions) > 0
            assert test_grid.lam is not None
            assert len(test_grid.lam) > 0

            for spectra_type in test_grid.available_spectra:
                assert spectra_type in test_grid.spectra
                spectra = test_grid.spectra[spectra_type]
                assert spectra.shape[-1] == len(test_grid.lam)

    def test_interp_spectra(self, test_grid):
        """Test spectral interpolation."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Create a test wavelength array
        original_shape = test_grid.shape

        # Create new wavelength array with fewer points
        new_lam = test_grid.lam[::2]  # Every other wavelength

        # Interpolate
        test_grid.interp_spectra(new_lam)

        # Check that wavelength array changed
        assert len(test_grid.lam) == len(new_lam)
        assert np.allclose(test_grid.lam, new_lam)

        # Check that spectra shapes changed
        new_shape = test_grid.shape
        assert (
            new_shape[:-1] == original_shape[:-1]
        )  # Other dimensions unchanged
        assert new_shape[-1] == len(new_lam)  # Wavelength dimension changed

    def test_get_sed_methods(self, test_grid):
        """Test SED extraction methods."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        spectra_type = test_grid.available_spectra[0]

        # Test getting SED for full grid
        sed_full = test_grid.get_sed_for_full_grid(spectra_type)
        assert sed_full.lam.shape == test_grid.lam.shape
        assert sed_full.lnu.shape == test_grid.spectra[spectra_type].shape

        # Test getting SED at a grid point
        grid_point = tuple(0 for _ in range(test_grid.naxes))
        sed_point = test_grid.get_sed_at_grid_point(grid_point, spectra_type)
        assert sed_point.lam.shape == test_grid.lam.shape
        assert len(sed_point.lnu.shape) == 1  # Should be 1D

        # Test the general get_sed method
        sed_general_full = test_grid.get_sed(None, spectra_type)
        sed_general_point = test_grid.get_sed(grid_point, spectra_type)

        assert np.allclose(sed_full.lnu, sed_general_full.lnu)
        assert np.allclose(sed_point.lnu, sed_general_point.lnu)


class TestGridLines:
    """Tests for Grid line handling."""

    def test_lines_properties(self, test_grid):
        """Test line-related properties."""
        if test_grid.has_lines:
            assert len(test_grid.available_lines) > 0
            assert len(test_grid.available_line_emissions) > 0
            assert test_grid.line_lams is not None
            assert len(test_grid.line_lams) > 0
            assert test_grid.nlines == len(test_grid.available_lines)

            for emission_type in test_grid.available_line_emissions:
                assert emission_type in test_grid.line_lums
                assert emission_type in test_grid.line_conts

                line_lums = test_grid.line_lums[emission_type]
                line_conts = test_grid.line_conts[emission_type]

                assert line_lums.shape[-1] == len(test_grid.available_lines)
                assert line_conts.shape[-1] == len(test_grid.available_lines)

    def test_lines_different(self, test_grid):
        """Test that the lines are different."""
        if not test_grid.has_lines:
            pytest.skip("Grid has no lines")

        # Ensure that all values aren't the same for the whole grid
        luminosities = test_grid.line_lums["nebular"]
        conts = test_grid.line_conts["nebular"]
        assert not np.unique(luminosities).size == 1, (
            f"All line luminosities are the same {luminosities.min()}"
        )
        assert not np.unique(conts).size == 1, (
            f"All line conts are the same {conts.min()}"
        )

        # Ensure that none of the lines are all the same
        non_unique_lum_lines = []
        non_unique_cont_lines = []
        non_unique_lum_vals = []
        non_unique_cont_vals = []
        for ind, line in enumerate(test_grid.available_lines):
            # Skip entirely zeroed lines, these have meaning
            if np.sum(luminosities[..., ind]) == 0:
                continue
            if np.unique(luminosities[..., ind]).size == 1:
                non_unique_lum_lines.append(line)
                non_unique_lum_vals.append(luminosities[..., ind].min())

            if np.unique(conts[..., ind]).size == 1:
                non_unique_cont_lines.append(line)
                non_unique_cont_vals.append(conts[..., ind].min())

        assert non_unique_lum_lines == [], (
            f"{non_unique_lum_lines} found with constant luminosity "
            f"values in the grid ({non_unique_lum_vals})"
        )
        assert non_unique_cont_lines == [], (
            f"{non_unique_cont_lines} found with constant continuum "
            f"values in the grid ({non_unique_cont_vals})"
        )

    def test_get_lines_methods(self, test_grid):
        """Test line extraction methods."""
        if not test_grid.has_lines:
            pytest.skip("Grid has no lines")

        # Test getting lines for full grid
        lines_full = test_grid.get_lines_for_full_grid()
        assert len(lines_full.line_ids) == len(test_grid.available_lines)

        # Test getting lines at a grid point
        grid_point = tuple(0 for _ in range(test_grid.naxes))
        lines_point = test_grid.get_lines_at_grid_point(grid_point)
        assert len(lines_point.line_ids) == len(test_grid.available_lines)

        # Test getting specific line
        if len(test_grid.available_lines) > 0:
            line_id = test_grid.available_lines[0]
            specific_line = test_grid.get_lines(grid_point, line_id)
            # Check that we got the right line
            assert hasattr(specific_line, "line_id") or hasattr(
                specific_line, "line_ids"
            )
            if hasattr(specific_line, "line_id"):
                assert specific_line.line_id == line_id
            else:
                assert line_id in specific_line.line_ids

        # Test the general get_lines method
        lines_general_full = test_grid.get_lines(None)
        lines_general_point = test_grid.get_lines(grid_point)

        assert len(lines_general_full.line_ids) == len(lines_full.line_ids)
        assert len(lines_general_point.line_ids) == len(lines_point.line_ids)


class TestGridReductionMethods:
    """Tests for the new grid reduction methods."""

    def test_reduce_rest_frame_range(self, test_grid_name):
        """Test reducing grid to a rest frame wavelength range."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        original_lam = grid.lam.copy()

        # Define a wavelength range
        lam_min = original_lam[len(original_lam) // 4]
        lam_max = original_lam[3 * len(original_lam) // 4]

        # Reduce the grid (returns new grid by default)
        reduced_grid = grid.reduce_rest_frame_range(lam_min, lam_max)

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Reduced grid should be modified
        assert len(reduced_grid.lam) < len(original_lam)
        assert reduced_grid.lam[0] >= lam_min
        assert reduced_grid.lam[-1] <= lam_max

        # Check spectra shapes changed accordingly
        for spectra_type in reduced_grid.available_spectra:
            assert reduced_grid.spectra[spectra_type].shape[-1] == len(
                reduced_grid.lam
            )

    def test_reduce_rest_frame_range_inplace(self, test_grid_name):
        """Test reducing grid to a rest frame wavelength range in-place."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        original_lam = grid.lam.copy()

        # Define a wavelength range
        lam_min = original_lam[len(original_lam) // 4]
        lam_max = original_lam[3 * len(original_lam) // 4]

        # Reduce the grid in-place
        result = grid.reduce_rest_frame_range(lam_min, lam_max, inplace=True)

        # Should return None for in-place operations
        assert result is None

        # Original grid should be modified
        assert len(grid.lam) < len(original_lam)
        assert grid.lam[0] >= lam_min
        assert grid.lam[-1] <= lam_max

        # Check spectra shapes changed accordingly
        for spectra_type in grid.available_spectra:
            assert grid.spectra[spectra_type].shape[-1] == len(grid.lam)

    def test_reduce_rest_frame_range_invalid_args(self, test_grid_name):
        """Test reduce_rest_frame_range with invalid arguments."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Test with lam_min >= lam_max
        with pytest.raises(exceptions.InconsistentArguments):
            grid.reduce_rest_frame_range(5000 * angstrom, 4000 * angstrom)

    def test_reduce_observed_range(self, test_grid_name):
        """Test reducing grid to an observed frame wavelength range."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        original_lam = grid.lam.copy()
        redshift = 1.0

        # Define an observed wavelength range
        obs_lam_min = original_lam[len(original_lam) // 4] * (1 + redshift)
        obs_lam_max = original_lam[3 * len(original_lam) // 4] * (1 + redshift)

        # Reduce the grid (returns new grid by default)
        reduced_grid = grid.reduce_observed_range(
            obs_lam_min, obs_lam_max, redshift
        )

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Check wavelength array was reduced
        assert len(reduced_grid.lam) < len(original_lam)

        # Check that the rest frame limits are correct
        rest_lam_min = obs_lam_min / (1 + redshift)
        rest_lam_max = obs_lam_max / (1 + redshift)
        assert reduced_grid.lam[0] >= rest_lam_min
        assert reduced_grid.lam[-1] <= rest_lam_max

        # More rigorous physical correctness checks
        # The reduced grid should contain wavelengths that when redshifted
        # fall within the observed range
        redshifted_lam_min = reduced_grid.lam[0] * (1 + redshift)
        redshifted_lam_max = reduced_grid.lam[-1] * (1 + redshift)

        # Allow small tolerance for numerical precision
        tolerance = 0.01  # 1% tolerance
        assert redshifted_lam_min >= obs_lam_min * (1 - tolerance)
        assert redshifted_lam_max <= obs_lam_max * (1 + tolerance)

        # Verify no wavelengths outside the expected rest-frame range exist
        # in the original grid that should have been included
        expected_rest_range_mask = (original_lam >= rest_lam_min) & (
            original_lam <= rest_lam_max
        )
        expected_count = np.sum(expected_rest_range_mask)
        # The reduced grid should have approximately the same number of points
        # (within reasonable tolerance for edge effects)
        assert abs(len(reduced_grid.lam) - expected_count) <= 2

    def test_reduce_rest_frame_lam(self, test_grid_name):
        """Test reducing grid to a new rest frame wavelength array."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        original_lam = grid.lam.copy()
        # Create a new wavelength array with fewer points
        new_lam = grid.lam[::2]  # Every other wavelength

        # Reduce the grid (returns new grid by default)
        reduced_grid = grid.reduce_rest_frame_lam(new_lam)

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Check wavelength array changed
        assert len(reduced_grid.lam) == len(new_lam)
        assert np.allclose(reduced_grid.lam, new_lam)

    def test_reduce_observed_lam(self, test_grid_name):
        """Test reducing grid to a new observed frame wavelength array."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        original_lam = grid.lam.copy()
        redshift = 1.0

        # Create a new observed wavelength array
        obs_lam = grid.lam[::2] * (
            1 + redshift
        )  # Every other wavelength, redshifted

        # Reduce the grid (returns new grid by default)
        reduced_grid = grid.reduce_observed_lam(obs_lam, redshift)

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Check wavelength array changed to rest frame equivalent
        expected_rest_lam = obs_lam / (1 + redshift)
        assert len(reduced_grid.lam) == len(expected_rest_lam)
        assert np.allclose(reduced_grid.lam, expected_rest_lam)

    def test_redshift_transformations_physical_correctness(
        self, test_grid_name
    ):
        """Test that all observed frame methods handle redshift correctly."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Test with various redshifts to ensure consistency
        redshifts = [0.5, 1.0, 2.0, 5.0]

        for z in redshifts:
            original_lam = grid.lam.copy()

            # Test observed range reduction
            # Pick a rest-frame range
            rest_min = original_lam[len(original_lam) // 3]
            rest_max = original_lam[2 * len(original_lam) // 3]

            # Convert to observed frame
            obs_min = rest_min * (1 + z)
            obs_max = rest_max * (1 + z)

            # Reduce and verify
            reduced_grid = grid.reduce_observed_range(obs_min, obs_max, z)

            # The cosmological redshift formula: λ_obs = λ_rest * (1 + z)
            # So λ_rest = λ_obs / (1 + z)

            # Check that the actual range matches expectations
            actual_rest_min = obs_min / (1 + z)
            actual_rest_max = obs_max / (1 + z)

            # Allow small numerical tolerance
            tolerance = 1e-10
            assert abs(actual_rest_min - rest_min) < tolerance
            assert abs(actual_rest_max - rest_max) < tolerance

            # Verify the reduced grid has the right rest-frame range
            assert reduced_grid.lam[0] >= actual_rest_min * 0.99
            assert reduced_grid.lam[-1] <= actual_rest_max * 1.01

            # Test observed wavelength array reduction
            # Create observed array and verify transformation
            obs_lam_test = (
                np.array([3000, 4000, 5000, 6000]) * (1 + z) * angstrom
            )
            reduced_grid_lam = grid.reduce_observed_lam(obs_lam_test, z)

            expected_rest = obs_lam_test / (1 + z)
            assert np.allclose(reduced_grid_lam.lam, expected_rest)

    def test_grid_new_lam_interpolation(self, test_grid_name):
        """Test Grid initialization with new_lam interpolation."""
        # Create a custom wavelength array
        new_lams = np.logspace(2, 5, 1000) * angstrom

        # Test Grid creation with new_lam
        grid = Grid(test_grid_name, new_lam=new_lams)

        # Check that the wavelength array was updated
        assert len(grid.lam) == len(new_lams)
        assert np.allclose(grid.lam.value, new_lams.value)
        assert grid.lam.units == new_lams.units

        # Check that spectra were interpolated correctly
        assert grid.has_spectra
        for spectra_type in grid.available_spectra:
            # The last dimension should match the new wavelength array
            assert grid.spectra[spectra_type].shape[-1] == len(new_lams)

        # Test with a smaller wavelength range
        small_lams = np.logspace(3, 4, 100) * angstrom
        small_grid = Grid(test_grid_name, new_lam=small_lams)

        assert len(small_grid.lam) == 100
        assert np.allclose(small_grid.lam.value, small_lams.value)

        # Test that it works with lines (even if ignore_lines has issues)
        grid_with_lines = Grid(test_grid_name, new_lam=new_lams)
        assert len(grid_with_lines.lam) == len(new_lams)
        # Lines may or may not be available depending on grid processing

    def test_reduce_axis(self, test_grid_name):
        """Test reducing a grid axis to a specified range."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)

        # Get the first axis and its values
        axis_name = grid.axes[0]
        axis_values = getattr(grid, axis_name)
        original_shape = grid.shape

        # Define a range that's a subset of the axis
        axis_low = axis_values[1]  # Second value
        axis_high = axis_values[-2]  # Second to last value

        # Reduce the axis (returns new grid by default)
        reduced_grid = grid.reduce_axis(axis_low, axis_high, axis_name)

        # Original grid should be unchanged
        assert getattr(grid, axis_name).shape == axis_values.shape
        assert grid.shape == original_shape

        # Check that the axis was reduced
        new_axis_values = getattr(reduced_grid, axis_name)
        assert len(new_axis_values) < len(axis_values)
        assert new_axis_values[0] >= axis_low
        assert new_axis_values[-1] <= axis_high

        # Check that grid shape changed
        new_shape = reduced_grid.shape
        assert new_shape[0] < original_shape[0]  # First axis should be smaller

        # Check that spectra and lines were reduced accordingly
        if reduced_grid.has_spectra:
            for spectra_type in reduced_grid.available_spectra:
                assert reduced_grid.spectra[spectra_type].shape == new_shape

        if reduced_grid.has_lines:
            for emission_type in reduced_grid.available_line_emissions:
                expected_line_shape = new_shape[:-1] + (reduced_grid.nlines,)
                assert (
                    reduced_grid.line_lums[emission_type].shape
                    == expected_line_shape
                )
                assert (
                    reduced_grid.line_conts[emission_type].shape
                    == expected_line_shape
                )

    def test_reduce_axis_invalid_args(self, test_grid_name):
        """Test reduce_axis with invalid arguments."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        axis_name = grid.axes[0]
        axis_values = getattr(grid, axis_name)

        # Test with axis_low >= axis_high
        with pytest.raises(exceptions.InconsistentArguments):
            grid.reduce_axis(axis_values[5], axis_values[2], axis_name)

        # Test with values outside axis range
        with pytest.raises(exceptions.InconsistentArguments):
            # Need to handle units properly
            min_val = axis_values.min()
            if hasattr(min_val, "units"):
                out_of_range_low = min_val - min_val * 0.1  # 10% below minimum
            else:
                out_of_range_low = min_val - 1
            grid.reduce_axis(out_of_range_low, axis_values[2], axis_name)

        # Test with invalid axis name
        with pytest.raises(exceptions.InconsistentArguments):
            grid.reduce_axis(axis_values[1], axis_values[2], "invalid_axis")

    def test_reduce_rest_frame_filters(self, test_grid_name):
        """Test reducing grid to rest frame filter ranges."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Create test filters
        filters = UVJ()  # FilterCollection object
        original_lam = grid.lam.copy()

        # Reduce the grid to filter range (returns new grid by default)
        reduced_grid = grid.reduce_rest_frame_filters(filters)

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Check wavelength array was reduced
        assert len(reduced_grid.lam) < len(original_lam)

        # Check that wavelengths are within filter range
        filter_min = min(f.lam_eff for f in filters)
        filter_max = max(f.lam_eff for f in filters)
        # Allow some tolerance since we're comparing effective wavelengths
        # to range edges
        assert reduced_grid.lam[0] >= filter_min * 0.8
        assert reduced_grid.lam[-1] <= filter_max * 1.2

    def test_reduce_observed_filters(self, test_grid_name):
        """Test reducing grid to observed frame filter ranges."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        if not grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Create test filters and redshift
        filters = UVJ()  # FilterCollection object
        redshift = 1.0
        original_lam = grid.lam.copy()

        # Reduce the grid to observed filter range (returns new grid by
        # default)
        reduced_grid = grid.reduce_observed_filters(filters, redshift)

        # Original grid should be unchanged
        assert len(grid.lam) == len(original_lam)

        # Check wavelength array was reduced
        assert len(reduced_grid.lam) < len(original_lam)

        # Physical correctness check: verify redshift transformation
        # Get the rest-frame filter range
        filter_min = min(f.lam_eff for f in filters)
        filter_max = max(f.lam_eff for f in filters)
        rest_filter_min = filter_min / (1 + redshift)
        rest_filter_max = filter_max / (1 + redshift)

        # The reduced grid should span approximately this rest-frame range
        # (allowing for filter width and edge effects)
        assert reduced_grid.lam[0] <= rest_filter_max * 1.2
        assert reduced_grid.lam[-1] >= rest_filter_min * 0.8

        # More stringent check: verify the observed wavelengths of the
        # reduced grid would fall within a reasonable filter range
        obs_lam_min = reduced_grid.lam[0] * (1 + redshift)
        obs_lam_max = reduced_grid.lam[-1] * (1 + redshift)

        # These should overlap significantly with the filter range
        # Check for reasonable overlap (allowing for filter widths)
        assert obs_lam_min <= filter_max * 1.5
        assert obs_lam_max >= filter_min * 0.5

        # Allow some tolerance since we're comparing effective wavelengths
        # to range edges
        assert reduced_grid.lam[0] >= rest_filter_min * 0.8
        assert reduced_grid.lam[-1] <= rest_filter_max * 1.2


class TestGridCollapse:
    """Tests for the grid collapse functionality."""

    def test_collapse_marginalize(self, test_grid_name):
        """Test collapsing grid by marginalizing over an axis."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        original_shape = grid.shape
        original_naxes = grid.naxes
        original_axes = grid.axes.copy()

        # Collapse the first axis by averaging (returns new grid by default)
        axis_to_collapse = grid.axes[0]
        collapsed_grid = grid.collapse(
            axis_to_collapse,
            method="marginalize",
            marginalize_function=np.mean,
        )

        # Original grid should be unchanged
        assert grid.naxes == original_naxes
        assert grid.axes == original_axes
        assert grid.shape == original_shape

        # Check that dimensionality was reduced in collapsed grid
        assert collapsed_grid.naxes == original_naxes - 1
        assert axis_to_collapse not in collapsed_grid.axes

        # Check that shapes changed correctly
        new_shape = collapsed_grid.shape
        assert len(new_shape) == len(original_shape) - 1
        assert new_shape == original_shape[1:]  # First axis removed

        # Check that the collapsed axis is no longer accessible
        with pytest.raises(AttributeError):
            getattr(collapsed_grid, axis_to_collapse)

    def test_collapse_interpolate(self, test_grid_name):
        """Test collapsing grid by interpolating to a specific value."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        original_shape = grid.shape
        original_naxes = grid.naxes
        original_axes = grid.axes.copy()

        # Get axis and pick a value to interpolate to
        axis_to_collapse = grid.axes[0]
        axis_values = getattr(grid, axis_to_collapse)

        # Pick a value in the middle of the range
        interp_value = axis_values[len(axis_values) // 2]

        # Collapse by interpolation (returns new grid by default)
        collapsed_grid = grid.collapse(
            axis_to_collapse, method="interpolate", value=interp_value
        )

        # Original grid should be unchanged
        assert grid.naxes == original_naxes
        assert grid.axes == original_axes
        assert grid.shape == original_shape

        # Check that dimensionality was reduced
        assert collapsed_grid.naxes == original_naxes - 1
        assert axis_to_collapse not in collapsed_grid.axes

        # Check that shapes changed correctly
        new_shape = collapsed_grid.shape
        assert len(new_shape) == len(original_shape) - 1
        assert new_shape == original_shape[1:]  # First axis removed

    def test_collapse_nearest(self, test_grid_name):
        """Test collapsing grid by extracting nearest value."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        original_shape = grid.shape
        original_naxes = grid.naxes
        original_axes = grid.axes.copy()

        # Get axis and pick a value to extract
        axis_to_collapse = grid.axes[0]
        axis_values = getattr(grid, axis_to_collapse)

        # Pick a value close to one of the grid points
        nearest_value = axis_values[len(axis_values) // 2]

        # Collapse by nearest extraction (returns new grid by default)
        collapsed_grid = grid.collapse(
            axis_to_collapse, method="nearest", value=nearest_value
        )

        # Original grid should be unchanged
        assert grid.naxes == original_naxes
        assert grid.axes == original_axes
        assert grid.shape == original_shape

        # Check that dimensionality was reduced
        assert collapsed_grid.naxes == original_naxes - 1
        assert axis_to_collapse not in collapsed_grid.axes

        # Check that shapes changed correctly
        new_shape = collapsed_grid.shape
        assert len(new_shape) == len(original_shape) - 1
        assert new_shape == original_shape[1:]  # First axis removed

    def test_collapse_invalid_args(self, test_grid_name):
        """Test collapse with invalid arguments."""
        # Create a fresh grid instance
        grid = Grid(test_grid_name)
        axis_name = grid.axes[0]

        # Test with invalid axis
        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse("invalid_axis")

        # Test with invalid method
        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse(axis_name, method="invalid_method")

        # Test interpolate/nearest without value
        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse(axis_name, method="interpolate")

        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse(axis_name, method="nearest")

        # Test with value outside axis range
        axis_values = getattr(grid, axis_name)
        # Need to handle units properly
        max_val = axis_values.max()
        if hasattr(max_val, "units"):
            out_of_range_value = max_val + max_val * 0.1  # 10% above maximum
        else:
            out_of_range_value = max_val + 1

        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse(
                axis_name, method="interpolate", value=out_of_range_value
            )

        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse(
                axis_name, method="nearest", value=out_of_range_value
            )


class TestGridCollapseAxisNaming:
    """Tests for grid collapse with various axis naming conventions."""

    def test_collapse_with_regular_axis_names(self, test_grid_name):
        """Test collapse with regular axis names (ages, metallicities)."""
        grid = Grid(test_grid_name)
        original_naxes = grid.naxes

        # Test with the regular axis name
        axis_name = "ages"
        axis_values = getattr(grid, axis_name)
        middle_value = axis_values[len(axis_values) // 2]

        # Test all three collapse methods with regular axis name
        for method in ["marginalize", "interpolate", "nearest"]:
            grid_copy = Grid(test_grid_name)

            if method == "marginalize":
                collapsed_grid = grid_copy.collapse(axis_name, method=method)
            else:
                collapsed_grid = grid_copy.collapse(
                    axis_name, method=method, value=middle_value
                )

            # Check dimensionality reduction
            assert collapsed_grid.naxes == original_naxes - 1
            assert axis_name not in collapsed_grid.axes

            # Check that the axis is no longer accessible
            with pytest.raises(AttributeError):
                getattr(collapsed_grid, axis_name)

    def test_collapse_with_plural_axis_names(self, test_grid_name):
        """Test collapse with plural axis names (ages, metallicities)."""
        grid = Grid(test_grid_name)
        original_naxes = grid.naxes

        # Test with metallicities (already plural)
        axis_name = "metallicities"

        # Test marginalize method
        collapsed_grid = grid.collapse(axis_name, method="marginalize")

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert axis_name not in collapsed_grid.axes

        # Check that the axis is no longer accessible
        with pytest.raises(AttributeError):
            getattr(collapsed_grid, axis_name)

    def test_collapse_with_singular_axis_names(self, test_grid_name):
        """Test collapse with singular forms of axis names."""
        grid = Grid(test_grid_name)
        original_naxes = grid.naxes

        # Test with singular form "age" (should work for "ages" axis)
        axis_name = "age"  # singular of "ages"
        axis_values = getattr(grid, "ages")  # Get values from plural form
        middle_value = axis_values[len(axis_values) // 2]

        # Test interpolate method with singular axis name
        collapsed_grid = grid.collapse(
            axis_name, method="interpolate", value=middle_value
        )

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert "ages" not in collapsed_grid.axes  # The actual axis name

        # Test with singular form "metallicity"
        grid_copy = Grid(test_grid_name)
        axis_name = "metallicity"  # singular of "metallicities"
        axis_values = getattr(
            grid_copy, "metallicities"
        )  # Get values from plural form
        middle_value = axis_values[len(axis_values) // 2]

        collapsed_grid = grid_copy.collapse(
            axis_name, method="nearest", value=middle_value
        )

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert "metallicities" not in collapsed_grid.axes

    def test_collapse_with_log10_axis_names(self, test_grid_name):
        """Test collapse with log10 axis names."""
        grid = Grid(test_grid_name)
        original_naxes = grid.naxes

        # Test with log10ages
        axis_name = "log10ages"
        log_axis_values = getattr(grid, axis_name)
        middle_value = log_axis_values[len(log_axis_values) // 2]

        # Test marginalize method with log10 axis
        collapsed_grid = grid.collapse(axis_name, method="marginalize")

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert (
            "ages" not in collapsed_grid.axes
        )  # The actual axis should be removed

        # Test with log10metallicities
        grid_copy = Grid(test_grid_name)
        axis_name = "log10metallicities"
        log_axis_values = getattr(grid_copy, axis_name)
        middle_value = log_axis_values[len(log_axis_values) // 2]

        collapsed_grid = grid_copy.collapse(
            axis_name, method="interpolate", value=middle_value
        )

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert "metallicities" not in collapsed_grid.axes

    def test_collapse_with_log10_singular_axis_names(self, test_grid_name):
        """Test collapse with log10 singular axis names."""
        grid = Grid(test_grid_name)
        original_naxes = grid.naxes

        # Test with log10age (singular)
        axis_name = "log10age"
        log_axis_values = getattr(grid, "log10ages")  # Get from plural form
        middle_value = log_axis_values[len(log_axis_values) // 2]

        collapsed_grid = grid.collapse(
            axis_name, method="nearest", value=middle_value
        )

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert "ages" not in collapsed_grid.axes

        # Test with log10metallicity (singular)
        grid_copy = Grid(test_grid_name)
        axis_name = "log10metallicity"
        log_axis_values = getattr(grid_copy, "log10metallicities")
        middle_value = log_axis_values[len(log_axis_values) // 2]

        collapsed_grid = grid_copy.collapse(
            axis_name, method="interpolate", value=middle_value
        )

        # Check dimensionality reduction
        assert collapsed_grid.naxes == original_naxes - 1
        assert "metallicities" not in collapsed_grid.axes

    def test_collapse_axis_name_variations_consistency(self, test_grid_name):
        """Test that different axis name variations give consistent results."""
        # Test that collapsing with different name variations of the same axis
        # gives the same result (for the same collapse method and value)

        axis_variations = [
            ("ages", "age", "log10ages", "log10age"),
            (
                "metallicities",
                "metallicity",
                "log10metallicities",
                "log10metallicity",
            ),
        ]

        for variations in axis_variations:
            regular_axis, singular_axis, log_axis, log_singular_axis = (
                variations
            )

            # Get a test value for interpolation (use regular axis values)
            grid_test = Grid(test_grid_name)
            axis_values = getattr(grid_test, regular_axis)
            middle_value = axis_values[len(axis_values) // 2]

            # For log10 axes, we need the log10 of the middle value
            log_middle_value = getattr(grid_test, log_axis)[
                len(axis_values) // 2
            ]

            results = []
            axis_names_to_test = [regular_axis, singular_axis]
            values_to_use = [middle_value, middle_value]

            # Add log variants
            axis_names_to_test.extend([log_axis, log_singular_axis])
            values_to_use.extend([log_middle_value, log_middle_value])

            for axis_name, value in zip(axis_names_to_test, values_to_use):
                grid_copy = Grid(test_grid_name)
                collapsed_grid = grid_copy.collapse(
                    axis_name, method="interpolate", value=value
                )

                # Store the resulting grid shape and a sample spectrum value
                result_shape = collapsed_grid.shape
                if collapsed_grid.has_spectra:
                    sample_spectrum = collapsed_grid.spectra[
                        collapsed_grid.available_spectra[0]
                    ][0, 100]  # Sample point
                else:
                    sample_spectrum = 0

                results.append((result_shape, sample_spectrum))

            # All results should be the same (within numerical tolerance)
            reference_result = results[0]
            for i, result in enumerate(results[1:], 1):
                assert result[0] == reference_result[0], (
                    "Shape mismatch for axis variation "
                    f"{axis_names_to_test[i]}: "
                    f"got {result[0]}, expected {reference_result[0]}"
                )

                if isinstance(result[1], (int, float)) and isinstance(
                    reference_result[1], (int, float)
                ):
                    assert abs(result[1] - reference_result[1]) < 1e-10, (
                        "Spectrum value mismatch for axis variation "
                        f"{axis_names_to_test[i]}: "
                        f"got {result[1]}, expected {reference_result[1]}"
                    )

    def test_collapse_invalid_axis_names(self, test_grid_name):
        """Test collapse with invalid axis names."""
        grid = Grid(test_grid_name)

        # Test with completely invalid axis name
        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse("invalid_axis", method="marginalize")

        # Test with almost-correct but wrong axis name
        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse("age_typo", method="marginalize")

        with pytest.raises(exceptions.InconsistentParameter):
            grid.collapse("log10ages_typo", method="marginalize")

    def test_collapse_preserves_grid_consistency(self, test_grid_name):
        """Test collapse preserves grid naming conventions."""
        grid = Grid(test_grid_name)

        # Test that after collapsing one axis, the remaining axis still works
        # with all naming conventions
        collapsed_grid = grid.collapse("ages", method="marginalize")

        # The remaining axis should still be accessible by all its name
        # variations
        remaining_axis = "metallicities"

        # These should all work and return the same values
        values_regular = getattr(collapsed_grid, remaining_axis)
        values_singular = getattr(collapsed_grid, "metallicity")
        values_log = getattr(collapsed_grid, "log10metallicities")
        values_log_singular = getattr(collapsed_grid, "log10metallicity")

        # Check consistency
        assert len(values_regular) == len(values_singular)
        assert len(values_log) == len(values_log_singular)
        assert np.allclose(values_regular, values_singular)
        assert np.allclose(values_log, values_log_singular)


class TestGridUtilities:
    """Tests for Grid utility methods."""

    def test_string_representation(self, test_grid):
        """Test Grid string representation."""
        grid_str = str(test_grid)
        assert isinstance(grid_str, str)
        assert len(grid_str) > 0
        assert "GRID" in grid_str  # The table header uses uppercase

    def test_get_delta_lambda(self, test_grid):
        """Test delta lambda calculation."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        lam, delta_lam = test_grid.get_delta_lambda()

        assert len(delta_lam) == len(lam) - 1
        assert np.all(delta_lam > 0)  # Should be positive

    def test_reprocessed_property(self, test_grid):
        """Test the reprocessed property."""
        # This should be a boolean
        assert isinstance(test_grid.reprocessed, bool)

    def test_lines_available_property(self, test_grid):
        """Test the lines_available property."""
        # This should be a boolean
        assert isinstance(test_grid.lines_available, bool)

        # Should be consistent with has_lines
        if test_grid.lines_available:
            assert test_grid.has_lines

    def test_stellar_fraction_property(self, test_grid):
        """Test the stellar_fraction property if available."""
        try:
            stellar_frac = test_grid.stellar_fraction
            assert isinstance(stellar_frac, np.ndarray)
            assert stellar_frac.ndim == 2  # Should be 2D (age, metallicity)
        except exceptions.GridError:
            # Grid doesn't have stellar fraction, which is fine
            pass

    def test_new_line_format_property(self, test_grid):
        """Test the new_line_format property."""
        # This should be a boolean
        assert isinstance(test_grid.new_line_format, bool)

    def test_line_ids_property(self, test_grid):
        """Test the line_ids property."""
        if test_grid.has_lines:
            line_ids = test_grid.line_ids
            assert isinstance(line_ids, (list, np.ndarray))
            assert len(line_ids) == len(test_grid.available_lines)
        else:
            # Should still work even without lines
            line_ids = test_grid.line_ids
            assert isinstance(line_ids, (list, np.ndarray))

    def test_available_spectra_property(self, test_grid):
        """Test the available_spectra property."""
        available_spectra = test_grid.available_spectra
        assert isinstance(available_spectra, list)
        if test_grid.has_spectra:
            assert len(available_spectra) > 0
        else:
            assert len(available_spectra) == 0

    def test_plot_specific_ionising_lum_method_exists(self, test_grid):
        """Test that the plot_specific_ionising_lum method exists."""
        # Just check that the method exists, don't actually plot
        assert hasattr(test_grid, "plot_specific_ionising_lum")
        assert callable(getattr(test_grid, "plot_specific_ionising_lum"))

    def test_animate_grid_method_exists(self, test_grid):
        """Test that the animate_grid method exists."""
        # Just check that the method exists, don't actually animate
        assert hasattr(test_grid, "animate_grid")
        assert callable(getattr(test_grid, "animate_grid"))


class TestTemplate:
    """Tests for the Template class."""

    def test_template_initialization(self):
        """Test Template initialization."""
        # Create simple test data
        lam = np.linspace(1000, 10000, 100) * angstrom
        lnu = np.ones_like(lam.value) * erg / s / Hz

        template = Template(lam, lnu)

        assert hasattr(template, "lam")
        assert hasattr(template, "normalisation")
        assert len(template.lam) == len(lam)

    def test_template_get_spectra(self):
        """Test Template spectra generation."""
        # Create simple test data
        lam = np.linspace(1000, 10000, 100) * angstrom
        lnu = np.ones_like(lam.value) * erg / s / Hz

        template = Template(lam, lnu)

        # Test scaling by bolometric luminosity
        bol_lum = 1e42 * erg / s
        scaled_sed = template.get_spectra(bol_lum)

        assert hasattr(scaled_sed, "lam")
        assert hasattr(scaled_sed, "lnu")
        assert len(scaled_sed.lam) == len(lam)

    def test_template_unify_with_grid(self, test_grid):
        """Test Template unification with a Grid."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        # Create simple test data with different wavelength array
        lam = np.linspace(1000, 10000, 50) * angstrom  # Different from grid
        lnu = np.ones_like(lam.value) * erg / s / Hz

        template = Template(lam, lnu, unify_with_grid=test_grid)

        # Check that template wavelength matches grid
        assert len(template.lam) == len(test_grid.lam)
        assert np.allclose(template.lam, test_grid.lam)


class TestGridErrorHandling:
    """Tests for Grid error handling and edge cases."""

    def test_invalid_spectra_type(self, test_grid):
        """Test error handling for invalid spectra types."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        with pytest.raises(exceptions.InconsistentParameter):
            test_grid.get_sed_at_grid_point(
                (0,) * test_grid.naxes, "invalid_spectra"
            )

    def test_invalid_grid_point(self, test_grid):
        """Test error handling for invalid grid points."""
        if not test_grid.has_spectra:
            pytest.skip("Grid has no spectra")

        spectra_type = test_grid.available_spectra[0]

        # Test with wrong number of dimensions
        with pytest.raises(exceptions.InconsistentParameter):
            test_grid.get_sed_at_grid_point((0,), spectra_type)

        # Test with out of bounds indices
        invalid_point = tuple(1000 for _ in range(test_grid.naxes))
        with pytest.raises(IndexError):
            test_grid.get_sed_at_grid_point(invalid_point, spectra_type)

    def test_no_spectra_no_lines_error(self, test_grid_name):
        """Test error when grid has neither spectra nor lines."""
        # Create a grid with both spectra and lines ignored
        grid = Grid(test_grid_name, ignore_spectra=True, ignore_lines=True)

        # Verify that the grid was created but has no spectra or lines
        assert not grid.has_spectra
        assert not grid.has_lines
        assert len(grid.spectra) == 0
        assert len(grid.line_lums) == 0
        assert len(grid.line_conts) == 0

        # Accessing shape should raise an UnrecognisedOption error
        with pytest.raises(exceptions.UnrecognisedOption):
            _ = grid.shape
