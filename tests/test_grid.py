import pytest
import numpy as np

from synthesizer.grid import Grid


@pytest.fixture
def open_grid():
    """ returns a Grid object """

    return Grid('test_grid', grid_dir='/tests/test_grid')
