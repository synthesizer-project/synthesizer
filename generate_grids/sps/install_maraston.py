"""
Download BC03 and convert to HDF5 synthesizer grid.
"""

import numpy as np
import os
import sys
import re
import wget
from utils import write_data_h5py, write_attribute
import tarfile
import glob
import gzip
import shutil

from synthesizer.sed import convert_flam_to_fnu

from pathlib import Path








# --- these could be replaced by our own mirror


def download_data():

    filename = wget.download(original_data_url[imf]) # download the original data to the working directory



    Path(input_dir).mkdir(parents=True, exist_ok=True)

    # --- untar main directory
    tar = tarfile.open(filename)
    tar.extractall(path = input_dir)
    tar.close()
    os.remove(filename)

    # # --- unzip the individual files that need reading
    # model_dir = f'{sythesizer_data_dir}/input_files/bc03/models/Padova2000/chabrier'
    # files = glob.glob(f'{model_dir}/bc2003_hr_m*_chab_ssp.ised_ASCII.gz')
    #
    # for file in files:
    #     with gzip.open(file, 'rb') as f_in:
    #         with open('.'.join(file.split('.')[:-1]), 'wb') as f_out:
    #             shutil.copyfileobj(f_in, f_out)
    #



def make_grid():
    """ Main function to convert BC03 grids and
        produce grids used by synthesizer """


    # Define output
    fname = f'{synthesizer_data_dir}/grids/{model_name}-{hr_morphology}_{imf}.h5'


    metallicities = np.array([0.001, 0.01, 0.02, 0.04])  # array of avialable metallicities
    # NOTE THE LOWEST METALLICITY MODEL DOES NOT HAVE YOUNG AGES

    log10metallicities = np.log10(metallicities)

    metallicity_code = {0.0001: '10m4', 0.001: '0001', 0.01: '001', 0.02: '002', 0.04: '004', 0.07: '007'} # codes for converting metallicty


    # --- open first file to get age
    fn = f'{input_dir}/sed.{imf_code[imf]}z{metallicity_code[metallicities[0]]}.{hr_morphology}'
    ages_, _, lam_, flam_ = np.loadtxt(fn).T

    ages_Gyr = np.sort(np.array(list(set(ages_)))) # Gyr
    ages = ages_Gyr * 1E9 # yr
    log10ages = np.log10(ages)

    lam = lam_[ages_==ages_[0]]

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    for iZ, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):

            fn = f'{input_dir}/sed.{imf_code[imf]}z{metallicity_code[metallicity]}.{hr_morphology}'
            print(iZ, ia, fn)
            ages_, _, lam_, flam_ = np.loadtxt(fn).T

            flam = flam_[ages_==age_Gyr]
            fnu = convert_flam_to_fnu(lam, flam)
            spec[ia, iZ] = fnu



    write_data_h5py(fname, 'spectra/wavelength', data=lam, overwrite=True)
    write_attribute(fname, 'spectra/wavelength', 'Description',
            'Wavelength of the spectra grid')
    write_attribute(fname, 'spectra/wavelength', 'Units', 'AA')

    write_data_h5py(fname, 'ages', data=ages, overwrite=True)
    write_attribute(fname, 'ages', 'Description',
            'Stellar population ages years')
    write_attribute(fname, 'ages', 'Units', 'yr')

    write_data_h5py(fname, 'log10ages', data=log10ages, overwrite=True)
    write_attribute(fname, 'log10ages', 'Description',
            'Stellar population ages in log10 years')
    write_attribute(fname, 'log10ages', 'Units', 'log10(yr)')

    write_data_h5py(fname, 'metallicities', data=metallicities, overwrite=True)
    write_attribute(fname, 'metallicities', 'Description',
            'raw abundances')
    write_attribute(fname, 'metallicities', 'Units', 'dimensionless [Z]')

    write_data_h5py(fname, 'log10metallicities', data=log10metallicities, overwrite=True)
    write_attribute(fname, 'log10metallicities', 'Description',
            'raw abundances in log10')
    write_attribute(fname, 'log10metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(fname, 'spectra/stellar', data=spec, overwrite=True)
    write_attribute(fname, 'spectra/stellar', 'Description',
                    'Three-dimensional spectra grid, [age, metallicity, wavelength]')
    write_attribute(fname, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')


# Lets include a way to call this script not via an entry point
if __name__ == "__main__":

    synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA') # this is the place to store the data

    model_name = 'maraston'

    input_dir = f'{synthesizer_data_dir}/input_files/{model_name}'  # the location to untar the original data

    imfs = ['salpeter']
    # imfs = ['salpeter','kroupa']

    original_data_url = {}
    original_data_url['salpeter'] = 'http://www.icg.port.ac.uk/~maraston/SSPn/SED/Sed_Mar05_SSP_Salpeter.tar.gz'
    imf_code = {'salpeter': 'ss', 'kroupa': 'kr'}

    for imf in imfs: #,

        # download_data()

        for hr_morphology in ['rhb']:

            make_grid()

            # filename = f'{sythesizer_data_dir}/grids/{model_name}.h5'
            # add_log10Q(filename)
