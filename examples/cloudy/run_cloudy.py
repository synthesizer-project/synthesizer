

"""
Run a single cloudy model based on an SPS grid
"""

import os
import sys
import argparse
import numpy as np

from synthesizer.abundances_sw import Abundances
from synthesizer.grid import SpectralGrid
from synthesizer.cloudy_sw import create_cloudy_input

from synthesizer.utils import read_params



CO = 0.0
d2m = 0.0
alpha = 0.0
scaling = 'Wilkins+2020'
ia = 0
iZ = 8
log10U = -2.
model_name = 'test'


sps_grid = 'bpass-v2.2.1-bin_chab-100'


# ---- load SPS grid
grid = SpectralGrid(sps_grid)



# --- get metallicity
Z = grid.metallicities[iZ]

# ---- initialise abundances object
abundances = Abundances().generate_abundances(Z, alpha, CO, d2m, scaling = scaling) # abundances object


lam = grid.lam
lnu = grid.spectra['stellar'][ia, iZ]

create_cloudy_input(model_name, lam, lnu, abundances, log10U)

# --- define output filename


cloudy_path = '/Users/stephenwilkins/Dropbox/Research/software/cloudy/c17.01/source/cloudy.exe'

os.system(f'{cloudy_path} -r {model_name}')
