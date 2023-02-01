"""
Example for creating an Sed object straight from the Grid. This example will:
- extract an Sed from a grid point
- demonstrating adding two Sed objects together
"""

import sys
import numpy as np

from synthesizer.sed import Sed
from synthesizer.grid import Grid


grid_dir = '../tests/test_grid'
grid_name = 'test_grid'
grid = Grid(grid_name, grid_dir=grid_dir)

print(grid)

sed1 = grid.get_sed(5, 5)  # get stellar SED at ia = 5, iZ = 5
sed2 = grid.get_sed(3, 5)  # get stellar SED at ia = 5, iZ = 5

# add together two Seds (MAGIC!)
sed = sed1 + sed2

print(sed)
