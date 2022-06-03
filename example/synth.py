"""
Front end for handling data and parameter files, setting up runs
"""
import sys
synth_dir = "/cosma7/data/dp004/dc-love2/codes/synthesizer/"
sys.path.append(synth_dir)

import argparse

import numpy as np

from synthesizer import grid
from synthesizer.load_data import load_FLARES
from synthesizer.utils import read_params

# parser = argparse.ArgumentParser()
# parser.add_argument("param_file", help="parameter file", type=str)
# args = parser.parse_args()
# params = read_params(args.param_file)
params = read_params('parameters')


# ---- load SPS grid
_grid = grid.sps_grid(params.sps_grid, lines=True)

# ---- load FLARES
# _f = '/cosma7/data/dp004/dc-love2/codes/flares/data/FLARES_30_sp_info.hdf5'
_f = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.h5'
tag = '/010_z005p000/'
regions = [f'{_r:02}' for _r in np.arange(40)]
R23 = {}

for region in regions:
    gals = load_FLARES(_f, region, tag)

    [_g.calculate_stellar_line_luminosities(_grid, save=True) for _g in gals];

    R23[region] = np.array([np.sum([_g.stellar_line_luminosities[_line]\
             for _line in ['OII3726','OIII4959','OIII5007']]) /\
                _g.stellar_line_luminosities['HI4861'] for _g in gals\
                if _g.stellar_line_luminosities['HI4861'] > 0.])


[_g.calculate_stellar_spectrum(_grid, save=True) for _g in gals];

