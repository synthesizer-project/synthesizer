"""
Front end for handling data and parameter files, setting up runs
"""
import sys
synth_dir = "/cosma7/data/dp004/dc-love2/codes/synthesizer/"
sys.path.append(synth_dir)

import argparse
import numpy as np

from synthesizer import grid
from synthesizer.cloudy import (create_cloudy_binary, write_cloudy_input,
                                write_submission_script_cosma, measure_Q, calculate_Q,
                                calculate_U)

from synthesizer.utils import read_params

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

params = read_params(args.param_file)


# ---- load SPS grid
_grid = grid.sps_grid(params.sps_grid)

create_cloudy_binary(_grid, params)

parameters = np.array([(iZ, ia) for iZ in np.arange(len(_grid.metallicities))\
        for ia in np.arange(np.sum(_grid.ages <= 7))])

"""
write grid of outputs
"""

if params.fixed_U:
    # ---- Fixed ionisation parameter
    _U = params.U_target  # -2
    
    for i, (iZ, ia) in enumerate(parameters):
        write_cloudy_input(f'fsps_{i}', _grid, int(ia), int(iZ), _U,
                           output_dir=params.cloudy_output_dir)
else:
    # ---- grid of ionisation parameters, based on reference
    age_mask = _grid.ages <= 7
    Q_grid = np.zeros((len(_grid.metallicities), len(_grid.ages[age_mask])))
    for i in np.arange(len(_grid.metallicities)):
        for j in np.arange(len(_grid.ages[age_mask])):
            Q_grid[i,j] = measure_Q(_grid.wl, _grid.spectra[i, j] * 1.1964952e40)
    
    
    ref_ia = 20  # 1 Myr
    ref_iZ = 8  # ~ -2
    Q_U = calculate_Q(10**-2)
    Q_ref = measure_Q(_grid.wl, _grid.spectra[ref_iZ, ref_ia] * 1.1964952e40)
    mass_scaling = Q_U / Q_ref
    U_grid = calculate_U(Q_grid * mass_scaling)
    
    U_grid = np.log10(U_grid)
    
    for i, (_U, (iZ, ia)) in enumerate(zip(U_grid.flatten(), parameters)):
        write_cloudy_input(f'fsps_{i}', _grid, int(ia), int(iZ),_U,
                           output_dir=params.cloudy_output_dir)


write_submission_script_cosma(N=len(parameters), input_prefix='fsps',
                              params=params)
