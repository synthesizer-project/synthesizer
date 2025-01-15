import numpy as np

from synthesizer.grid import Grid


def test_grid_returned(test_grid):
    """
    Test that a Grid object is returned.
    """
    assert isinstance(test_grid, Grid)


def test_grid_axes(test_grid):
    """
    Test that the axes are returned correctly.
    """
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
