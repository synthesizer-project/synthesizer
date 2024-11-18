from synthesizer.grid import Grid


def test_grid_returned(test_grid):
    """
    Test that a Grid object is returned.
    """
    assert isinstance(test_grid, Grid)
