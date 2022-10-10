
""" This reads in a cloudy grid of models and creates a new SPS grid including the various outputs """



import argparse
import numpy as np
import h5py

#
# from unyt import c, h

h = 6.626E-34
c = 3.E8

from synthesizer.sed import calculate_Q
from synthesizer.cloudy_sw import read_wavelength, read_continuum, default_lines, read_lines

from synthesizer.utils import read_params

import shutil
import os
from scipy import integrate


synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')

path_to_grids = f'{synthesizer_data_dir}/grids'
path_to_cloudy_files = f'{synthesizer_data_dir}/cloudy'


cloudy_models = ['cloudy-v17.03_log10Uref-2'] # --- the cloudy grid

sps_grids = [
    'bc03_chabrier03',
    'bpass-v2.2.1-bin_100-100',
    'bpass-v2.2.1-bin_100-300',
    'bpass-v2.2.1-bin_135-100',
    'bpass-v2.2.1-bin_135-300',
    'bpass-v2.2.1-bin_135all-100',
    'bpass-v2.2.1-bin_170-100',
    'bpass-v2.2.1-bin_170-300',
    'fsps-v3.2_Chabrier03',
    'bpass-v2.2.1-bin_chab-100',
    'bpass-v2.2.1-bin_chab-300',
    'maraston-rhb_kroupa',
    'maraston-rhb_salpeter',
    'bc03-2016-Stelib_chabrier03',
    'bc03-2016-BaSeL_chabrier03',
    'bc03-2016-Miles_chabrier03',
]



for sps_model in sps_grids:

    for cloudy_model in cloudy_models:


        # spec_names = ['incident','transmitted','nebular','nebular_continuum','total','linecont']
        spec_names = ['incident','transmitted','nebular','linecont'] # --- the cloudy spectra to save (others can be generated later)


        # --- open the original SPS model grid
        fn = f'{path_to_grids}/{sps_model}.h5'
        hf_sps = h5py.File(fn,'r')

        # --- open the new grid file
        fn = f'{path_to_grids}/{sps_model}_{cloudy_model}.h5' # the new cloudy grid
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
        lam = read_wavelength(f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/0_0')


        spectra = hf.create_group('spectra')  # create a group holding the spectra in the grid file
        spectra.attrs['spec_names'] = spec_names  # save list of spectra as attribute


        print(spec_names)

        spectra['wavelength'] = lam  # save the wavelength

        nlam = len(lam)  # number of wavelength points

        # --- make spectral grids and set them to zero
        for spec_name in spec_names:
            spectra[spec_name] = np.zeros((na, nZ, nlam))


        # --- now loop over meallicity and ages

        dlog10Q =  np.zeros((na, nZ))


        for iZ, Z in enumerate(metallicities):
            for ia, log10age in enumerate(log10ages):


                infile = f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/{ia}_{iZ}'

                if os.path.isfile(infile+'.cont'):  # attempt to open run.
                    exists = 'exists'
                    spec_dict = read_continuum(infile, return_dict = True)
                    for spec_name in spec_names:
                        spectra[spec_name][ia, iZ] = spec_dict[spec_name]

                    # --- we need to rescale the cloudy spectra to the original SPS spectra. This is done here using the ionising photon luminosity, though could in principle by done at a fixed wavelength.

                    log10Q = np.log10(calculate_Q(lam, spectra['incident'][ia, iZ, :]))  # calculate log10Q
                    dlog10Q[ia, iZ] = hf['log10Q'][ia, iZ] - log10Q

                    for spec_name in spec_names:
                        spectra[spec_name][ia, iZ] *= 10**dlog10Q[ia, iZ]

                else:
                    exists = 'does not exist'



        # -- get list of lines

        lines = hf.create_group('lines')
        lines.attrs['lines'] = default_lines  # save list of spectra as attribute

        for line_id in default_lines:
            lines[f'{line_id}/luminosity'] = np.zeros((na, nZ))
            lines[f'{line_id}/stellar_continuum'] = np.zeros((na, nZ))
            lines[f'{line_id}/nebular_continuum'] = np.zeros((na, nZ))
            lines[f'{line_id}/continuum'] = np.zeros((na, nZ))


        for iZ, Z in enumerate(metallicities):
            for ia, log10age in enumerate(log10ages):

                infile = f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/{ia}_{iZ}'

                # --- get line quantities
                line_ids, line_wavelengths, _, line_luminosities = read_lines(infile)

                # --- get TOTAL continuum spectra
                nebular_continuum = spectra['nebular'][ia, iZ] - spectra['linecont'][ia, iZ]
                continuum = spectra['transmitted'][ia, iZ] + nebular_continuum

                for line_id, line_wv, line_lum in zip(line_ids, line_wavelengths, line_luminosities):
                    lines[line_id].attrs['wavelength'] = line_wv
                    lines[f'{line_id}/luminosity'][ia, iZ] = 10**(line_lum + dlog10Q[ia, iZ]) # erg s^-1
                    lines[f'{line_id}/stellar_continuum'][ia, iZ] = np.interp(line_wv, lam, spectra['transmitted'][ia, iZ]) # erg s^-1 Hz^-1
                    lines[f'{line_id}/nebular_continuum'][ia, iZ] = np.interp(line_wv, lam, nebular_continuum) # erg s^-1 Hz^-1
                    lines[f'{line_id}/continuum'][ia, iZ] = np.interp(line_wv, lam, continuum) # erg s^-1 Hz^-1






        hf.visit(print)
        hf.flush()
