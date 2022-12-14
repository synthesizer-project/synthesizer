"""
Front end for handling data and parameter files, setting up runs
"""

import os
import argparse
import numpy as np

from synthesizer import grid
from synthesizer.cloudy import (create_cloudy_binary, write_cloudy_input,
                                write_submission_script_cosma, measure_Q,
                                calculate_Q, calculate_U)

from synthesizer.utils import read_params

parser = argparse.ArgumentParser()
arser.add_argument("-dir", "--directory", type=str, required=True)
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

params = read_params(args.param_file)

# synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')
output_dir = f'{synthesizer_data_dir}/cloudy_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---- load SPS grid
_grid = grid.SpectralGrid(params.sps)
create_cloudy_binary(_grid, params)

parameters = np.array([(ia, iZ)
                       for ia in np.arange(np.sum(_grid.ages <= 1e7))
                       for iZ in np.arange(len(_grid.metallicities))])


"""
write grid of outputs
"""

if params.fixed_U:
    # ---- Fixed ionisation parameter
    _U = params.U_target  # -2

    for i, (iZ, ia) in enumerate(parameters):
        write_cloudy_input(f'fsps_{i}', _grid, int(ia), int(iZ), _U,
                           output_dir=output_dir)
else:
    # ---- grid of ionisation parameters, based on reference
    age_mask = _grid.ages <= 1e7
    Q_grid = np.zeros((len(_grid.ages[age_mask]), len(_grid.metallicities)))
    for i in np.arange(len(_grid.ages[age_mask])):
        for j in np.arange(len(_grid.metallicities)):
            Q_grid[i, j] = measure_Q(_grid.lam,
                                     _grid.spectra['stellar'][i, j] * 1.1964952e40)

    # ref_ia = 20  # 1 Myr
    # ref_iZ = 8  # ~ -2
    Q_U = calculate_Q(10**-2)
    Q_ref = measure_Q(_grid.lam,
                      _grid.spectra['stellar'][params.index_age,
                                               params.index_Z] * 1.1964952e40)

    mass_scaling = Q_U / Q_ref
    U_grid = calculate_U(Q_grid * mass_scaling)
    U_grid = np.log10(U_grid)

    for i, (_U, (ia, iZ)) in enumerate(zip(U_grid.flatten(), parameters)):
        write_cloudy_input(f'fsps_{i}', _grid, int(ia), int(iZ), _U,
                           output_dir=output_dir)


write_submission_script_cosma(N=len(parameters), input_prefix='fsps',
                              params=params, output_dir=output_dir)
