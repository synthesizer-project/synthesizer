import sys
synth_dir = "/cosma7/data/dp004/dc-love2/codes/synthesizer/"
sys.path.append(synth_dir)

import argparse
import numpy as np
import h5py

from synthesizer import grid
from synthesizer.utils import write_data_h5py, write_attribute
from synthesizer.cloudy import (read_continuum, read_lines,
                            measure_Q, calculate_U, calculate_Q)

from synthesizer.utils import read_params

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

params = read_params(args.param_file)


# params = __import__('parameters')
_grid = grid.sps_grid(params.sps_grid)
age_mask = _grid.ages <= 7
Nage = np.sum(age_mask)
NZ = len(_grid.metallicities)

"""
calculate grid mass weighting
"""

if params.fixed_U:
    # ---- grid mass weighting for fixed U
    Q = np.zeros(_grid.spectra[:,age_mask,:].shape[:2])
    mass_ratio = np.zeros(_grid.spectra[:,age_mask,:].shape[:2])
    for i in np.arange(NZ):
        for j in np.arange(Nage):
            _spec = _grid.spectra[i,j] * 1.1964952e40  # erg s^-1 AA^-1
            Q[i,j] = measure_Q(_grid.wl, _spec)
            # U_0 = calculate_U(Q[i,j])
            # mass_ratio[i,j] = 10**params.U_target / U_0
    
    mass_ratio = calculate_Q(10**params.U_target) / Q
else:
    # ---- grid mass weighting for varying U
    ref_ia = 20  # 1 Myr
    ref_iZ = 8  # ~ -2
    Q_U = calculate_Q(10**-2)
    Q_ref = measure_Q(_grid.wl, _grid.spectra[ref_iZ, ref_ia] * 1.1964952e40)
    mass_ratio = Q_U / Q_ref


parameters = np.array([(iZ, ia) for iZ in np.arange(len(_grid.metallicities))\
        for ia in np.arange(np.sum(age_mask))])


# ---- get all lines
linelist = ['OIII5007', 'OIII4959', 'OII3726', 'HI4861', 'HI6563', 'NII6583', 
            'SII6731', 'SII6716', 'NeIII3869']
Nline = len(linelist)
intrinsic_lines = []
emergent_lines = []

# for i, (iZ, ia, _U) in enumerate(parameters):
for i, (iZ, ia) in enumerate(parameters):
    n_Lam, n_intrinsic, n_emergent = read_lines(params.cloudy_output_dir,
                                                f'fsps_{i}', lines=linelist)

    intrinsic_lines.append(n_intrinsic)
    emergent_lines.append(n_emergent)


intrinsic_lines = 10**np.vstack(intrinsic_lines)
emergent_lines = 10**np.vstack(emergent_lines)

intrinsic_lines = intrinsic_lines.reshape((NZ, Nage, Nline))
emergent_lines = emergent_lines.reshape((NZ, Nage, Nline))


# --- apply grid weighting, save to new hdf5 file
intrinsic_lines /= np.expand_dims(mass_ratio, axis=-1)

with h5py.File(params.sps_grid, 'a') as hf:
    hf.require_group('lines')
    hf.require_group('lines/intrinsic')
    hf.require_group('lines/emergent')


for i,_line in enumerate(linelist):
    for _lines_arr,_type in zip([intrinsic_lines, emergent_lines],
                                ['intrinsic','emergent']):
        write_data_h5py(params.sps_grid, f'lines/{_type}/{_line}', data=_lines_arr[:,:,i], overwrite=True)
        write_attribute(params.sps_grid, f'lines/{_type}', 'Description',
                        f'{_type} line luminosities (1 Msol) [Z,Age]')
        write_attribute(params.sps_grid, f'lines/{_type}', 'Units', 'erg s^-1')
