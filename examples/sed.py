

import sys
import numpy as np

from synthesizer.sed import Sed
from synthesizer.grid import Grid


if len(sys.argv) > 1:
    grid_dir = str(sys.argv[1])
else:
    grid_dir = None

model = 'bc03_chabrier03'
grid = Grid(model, grid_dir=grid_dir)

sed1 = grid.get_sed(5, 5)  # get stellar SED at ia = 5, iZ = 5

sed2 = grid.get_sed(3, 5)  # get stellar SED at ia = 5, iZ = 5


sed = sed1 + sed2
