

"""
Run a single cloudy model based on an SPS grid
"""

import os
import sys
import argparse
import numpy as np

from synthesizer.abundances import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy import create_cloudy_input, ShapeCommands


grid_dir = '/Users/sw376/Dropbox/Research/data/synthesizer/grids/'
# grid_dir = '/its/home/sw376/astrodata/synthesizer/grids/'


default_params = {

    # --- sps parameters
    'sps_grid' : 'bpass-2.2.1-bin_chabrier03-0.1,300.0',
    'ia' : 0, # 1 Myr
    'iZ' : 8, # Z = 0.01

    # --- abundance parameters,  these are used, alongside the total metallicity (Z), to define the abundance pattern
    'CO' : 0.0,
    'd2m' : 0.3,
    'alpha' : 0.0,
    'scaling' : None,

    # --- cloudy model
    'cloudy_version' : 'c17.03',

    # --- cloudy parameters
    'log10U': -2,
    'log10radius': -2,  # radius in log10 parsecs, only important for spherical geometry
    'covering_factor': 1.0, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
    'stop_T': 4000,  # K
    'stop_efrac': -2,
    'T_floor': 100,  # K
    'log10n_H': 2.5,  # Hydrogen density
    'z': 0.,
    'CMB': False,
    'cosmic_rays': False,
    'grains': False,
    'geometry': 'planeparallel',
    'resolution': 1.0, # relative resolution the saved continuum spectra
    'output_abundances': True, # output abundances
    'output_cont': True, # output continuum
    'output_lines': True, # output lines
    }


params = {
    'resolution': 0.1, # relative resolution
}






model_name = '_'.join(['default']+[f'{k}:{v}' for k, v in params.items()])


params =  default_params | params

for k, v in params.items():
    print(k, v)



# --- define cloudy path
cloudy_path = f'~/Dropbox/Research/software/cloudy/{params["cloudy_version"]}/source/cloudy.exe'
# cloudy_path = f'/its/home/sw376/flare/software/cloudy/{params["cloudy_version"]}/source/cloudy.exe'



# ---- load SPS grid
grid = Grid(params['sps_grid'], grid_dir=grid_dir)

# --- get metallicity
Z = grid.metallicities[params['iZ']]

# ---- initialise abundances object
abundances = Abundances(Z=Z, alpha=params['alpha'], CO=params['CO'], d2m=params['d2m']) # abundances object


lam = grid.lam
lnu = grid.spectra['stellar'][params['ia'], params['iZ']]

# this returns the relevant shape commands, in this case for a tabulated SED
shape_commands = ShapeCommands.table_sed(model_name, lam, lnu, output_dir = './data/')

create_cloudy_input(model_name, shape_commands, abundances, output_dir = './data/', **params)

# --- define output filename

os.chdir('./data')
os.system(f'{cloudy_path} -r {model_name}')
