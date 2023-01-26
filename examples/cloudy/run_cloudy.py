

"""
Run a single cloudy model based on an SPS grid
"""

import os
import sys
import argparse
import numpy as np

from synthesizer.abundances_sw import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy_sw import create_cloudy_input


default_params = {

    # --- sps parameters
    'sps_grid': 'bpass-v2.2.1-bin_chab-100',
    'ia': 0,  # 1 Myr
    'iZ': 8,  # Z = 0.01

    # --- cloudy model
    'cloudy_version': 'c17.03',
    'log10U': -2,

    # --- abundance parameters,  these are used, alongside the total metallicity (Z), to define the abundance pattern
    'CO': 0.0,
    'd2m': 0.3,
    'alpha': 0.0,
    'scaling': None,

    # --- cloudy parameters
    'log10radius': -2,  # radius in log10 parsecs
    # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
    'covering_factor': 1.0,
    'stop_T': 4000,  # K
    'stop_efrac': -2,
    'T_floor': 100,  # K
    'log10n_H': 2,  # Hydrogen density
    'z': 0.,
    'CMB': False,
    'cosmic_rays': False
}


params = {
    'log10U': -1
}


model_name = '_'.join(['default']+[f'{k}-{v}' for k, v in params.items()])

print(model_name)

params = default_params | params

for k, v in params.items():
    print(k, v)


# ---- load SPS grid
grid = SpectralGrid(params['sps_grid'])

# --- get metallicity
Z = grid.metallicities[params['iZ']]

# ---- initialise abundances object
abundances = Abundances().generate_abundances(
    Z, params['alpha'], params['CO'], params['d2m'], scaling=params['scaling'])  # abundances object


lam = grid.lam
lnu = grid.spectra['stellar'][params['ia'], params['iZ']]

create_cloudy_input(model_name, lam, lnu, abundances,
                    output_dir='./data/', **params)

# --- define output filename


cloudy_path = f'/Users/stephenwilkins/Dropbox/Research/software/cloudy/{params["cloudy_version"]}/source/cloudy.exe'

os.chdir('./data')
os.system(f'{cloudy_path} -r {model_name}')
