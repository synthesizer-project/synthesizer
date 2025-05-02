"""A test suite for the Grid class.

TODO: This needs to actually be implemented with proper useful tests.
"""

import numpy as np

from synthesizer.grid import Grid


def test_grid_returned(test_grid):
    """Test that a Grid object is returned."""
    assert isinstance(test_grid, Grid)


def test_grid_axes(test_grid):
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


class TestSPSGridLines:
    """Tests for SPS lines grids."""

    def test_lines_different(self, test_grid):
        """Test that the lines are different."""
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
