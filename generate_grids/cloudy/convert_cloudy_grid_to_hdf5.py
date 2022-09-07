
""" This reads in a cloudy grid of models and creates a new SPS grid including the various outputs """



import argparse
import numpy as np
import h5py

#
# from unyt import c, h

h = 6.626E-34
c = 3.E8

from synthesizer import grid_sw as grid
from synthesizer.sed import calculate_Q
from synthesizer.utils import write_data_h5py, write_attribute
from synthesizer.cloudy import read_wavelength, read_continuum

from synthesizer.utils import read_params

import shutil
import os
from scipy import integrate




path_to_sps_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'

sps_model = 'bpass-v2.2.1_chab100-bin' # --- the SPS/IMF model
cloudy_model = 'cloudy-v17.0_logUref-2' # --- the cloudy grid
# spec_names = ['incident','transmitted','nebular','nebular_continuum','total','linecont']
spec_names = ['incident','transmitted','nebular','linecont'] # --- the cloudy spectra to save (others can be generated later)


# --- open the original SPS model grid
fn = f'{path_to_sps_grid}/{sps_model}.h5'
hf_sps = h5py.File(fn,'r')

# --- open the new grid file
fn = f'{path_to_sps_grid}/{sps_model}_{cloudy_model}.h5' # the new cloudy grid
hf = h5py.File(fn,'w')

# --- copy various quantities (all excluding the spectra) from the original sps grid
for ds in ['metallicities', 'log10metallicities', 'log10ages', 'log10Q', 'star_fraction', 'remnant_fraction']:
    hf_sps.copy(hf_sps[ds], hf['/'], ds)

# --- short hand for later
metallicities = hf['metallicities']
log10ages = hf['log10ages']

nZ = len(metallicities)  # number of metallicity grid points
na = len(log10ages)  # number of age grid points


# --- read first spectra from the first grid point to get length and wavelength grid
lam = read_wavelength(f'data/{sps_model}_{cloudy_model}/0_0')


spectra = hf.create_group('spectra')  # create a group holding the spectra in the grid file
spectra.attrs['spec_names'] = spec_names  # save list of spectra as attribute

spectra['wavelength'] = lam  # save the wavelength

nlam = len(lam)  # number of wavelength points

# --- make spectral grids and set them to zero
for spec_name in spec_names:
    spectra[spec_name] = np.zeros((na, nZ, nlam))


# --- now loop over meallicity and ages

for iZ, Z in enumerate(metallicities):
    for ia, log10age in enumerate(log10ages):


        infile = f'data/{sps_model}_{cloudy_model}/{ia}_{iZ}'

        if os.path.isfile(infile+'.cont'):  # attempt to open run.
            exists = 'exists'
            spec_dict = read_continuum(infile, return_dict = True)
            for spec_name in spec_names:
                spectra[spec_name][ia, iZ] = spec_dict[spec_name]

            # --- we need to rescale the cloudy spectra to the original SPS spectra. This is done here using the ionising photon luminosity, though could in principle by done at a fixed wavelength.

            log10Q = np.log10(calculate_Q(lam, spectra['incident'][ia, iZ, :]))  # calculate log10Q

            for spec_name in spec_names:
                spectra[spec_name][ia, iZ] *= 10**(hf['log10Q'][ia, iZ] - log10Q)

        else:
            exists = 'does not exist'






hf.visit(print)
hf.flush()
