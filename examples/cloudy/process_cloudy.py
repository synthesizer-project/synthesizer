
""" This reads in a cloudy grid of models and creates a new SPS grid including the various outputs """



import argparse
import numpy as np
import h5py

from synthesizer import grid_sw as grid
from synthesizer.utils import write_data_h5py, write_attribute
from synthesizer.cloudy import (read_wavelength, read_continuum, read_lines,
                            measure_Q, calculate_U, calculate_Q)

from synthesizer.utils import read_params

import shutil
import os
from scipy import integrate

h = 6.626E-34
c = 3.E8


path_to_sps_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'

sps_model = 'bpass-v2.2.1_chab100-bin'
cloudy_model = 'cloudy-v17.0_logUref-2'
# spec_names = ['incident','transmitted','nebular','nebular_continuum','total','linecont']
spec_names = ['incident','transmitted','nebular','linecont']

src = f'{path_to_sps_grid}/{sps_model}.h5'
dst = f'{path_to_sps_grid}/{sps_model}_{cloudy_model}.h5'

hf_sps = h5py.File(src,'r')

hf_sps.visit(print)
print('-'*20)

hf = h5py.File(dst,'w')

for ds in ['metallicities', 'log10metallicities', 'log10ages', 'log10Q', 'star_fraction', 'remnant_fraction']:
    hf_sps.copy(hf_sps[ds], hf['/'], ds)

metallicities = hf['metallicities']
log10ages = hf['log10ages']

nZ = len(metallicities)
na = len(log10ages)


# --- read first grid point to get length

lam = read_wavelength(f'data/{sps_model}_{cloudy_model}/0_0')


spectra = hf.create_group('spectra')
spectra.attrs['spec_names'] = spec_names

spectra['wavelength'] = lam



for spec_name in spec_names:
    spectra[spec_name] = np.zeros((na, nZ, len(lam)))


for iZ, Z in enumerate(metallicities):
    for ia, log10age in enumerate(log10ages):

        infile = f'data/{sps_model}_{cloudy_model}/{ia}_{iZ}'
        if os.path.isfile(infile+'.cont'):
            exists = 'exists'
            spec_dict = read_continuum(infile, return_dict = True)
            for spec_name in spec_names:
                spectra[spec_name][ia, iZ] = spec_dict[spec_name]



            # --- we need to rescale the cloudy spectra to the original SPS spectra. This is done here using the ionising photon luminosity, though could in principle by done at a fixed wavelength.

            # --- calculate log10Q [could encapsulate as a function as its used elsewhere as well]
            Lnu = 1E-7 * spectra['incident'][ia, iZ] # W s^-1 Hz^-1
            Llam = Lnu * c / (lam**2*1E-10) # W s^-1 \AA^-1
            nlam = (Llam*lam*1E-10)/(h*c) # s^-1 \AA^-1

            f = lambda l: np.interp(l, lam[::-1], nlam[::-1])
            n_LyC = integrate.quad(f, 10.0, 912.0)[0]

            for spec_name in spec_names:
                spectra[spec_name][ia, iZ] *=10**(hf['log10Q'][ia, iZ] - np.log10(n_LyC))

        else:
            exists = 'does not exist'

        print(ia, iZ, log10age, Z, exists)




hf.visit(print)
hf.flush()




#
#
# # params = __import__('parameters')
# _grid = grid.sps_grid(params.sps_grid)
# age_mask = _grid.ages <= 7
# Nage = np.sum(age_mask)
# NZ = len(_grid.metallicities)
#
# """
# calculate grid mass weighting
# """
#
# if params.fixed_U:
#     # ---- grid mass weighting for fixed U
#     Q = np.zeros(_grid.spectra[:,age_mask,:].shape[:2])
#     mass_ratio = np.zeros(_grid.spectra[:,age_mask,:].shape[:2])
#     for i in np.arange(NZ):
#         for j in np.arange(Nage):
#             _spec = _grid.spectra[i,j] * 1.1964952e40  # erg s^-1 AA^-1
#             Q[i,j] = measure_Q(_grid.wl, _spec)
#             # U_0 = calculate_U(Q[i,j])
#             # mass_ratio[i,j] = 10**params.U_target / U_0
#
#     mass_ratio = calculate_Q(10**params.U_target) / Q
# else:
#     # ---- grid mass weighting for varying U
#     ref_ia = 20  # 1 Myr
#     ref_iZ = 8  # ~ -2
#     Q_U = calculate_Q(10**-2)
#     Q_ref = measure_Q(_grid.wl, _grid.spectra[ref_iZ, ref_ia] * 1.1964952e40)
#     mass_ratio = Q_U / Q_ref
#
#
# parameters = np.array([(iZ, ia) for iZ in np.arange(len(_grid.metallicities))\
#         for ia in np.arange(np.sum(age_mask))])
#
#
# # ---- get all lines
# linelist = ['OIII5007', 'OIII4959', 'OII3726', 'HI4861', 'HI6563', 'NII6583',
#             'SII6731', 'SII6716', 'NeIII3869']
# Nline = len(linelist)
# intrinsic_lines = []
# emergent_lines = []
#
# # for i, (iZ, ia, _U) in enumerate(parameters):
# for i, (iZ, ia) in enumerate(parameters):
#     n_Lam, n_intrinsic, n_emergent = read_lines(params.cloudy_output_dir,
#                                                 f'fsps_{i}', lines=linelist)
#
#     intrinsic_lines.append(n_intrinsic)
#     emergent_lines.append(n_emergent)
#
#
# intrinsic_lines = 10**np.vstack(intrinsic_lines)
# emergent_lines = 10**np.vstack(emergent_lines)
#
# intrinsic_lines = intrinsic_lines.reshape((NZ, Nage, Nline))
# emergent_lines = emergent_lines.reshape((NZ, Nage, Nline))
#
#
# # --- apply grid weighting, save to new hdf5 file
# intrinsic_lines /= np.expand_dims(mass_ratio, axis=-1)
#
# with h5py.File(params.sps_grid, 'a') as hf:
#     hf.require_group('lines')
#     hf.require_group('lines/intrinsic')
#     hf.require_group('lines/emergent')
#
#
# for i,_line in enumerate(linelist):
#     for _lines_arr,_type in zip([intrinsic_lines, emergent_lines],
#                                 ['intrinsic','emergent']):
#         write_data_h5py(params.sps_grid, f'lines/{_type}/{_line}', data=_lines_arr[:,:,i], overwrite=True)
#         write_attribute(params.sps_grid, f'lines/{_type}', 'Description',
#                         f'{_type} line luminosities (1 Msol) [Z,Age]')
#         write_attribute(params.sps_grid, f'lines/{_type}', 'Units', 'erg s^-1')
