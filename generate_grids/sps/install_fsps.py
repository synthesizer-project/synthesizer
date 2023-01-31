import os
import sys

import numpy as np
import fsps

from utils import write_data_h5py, write_attribute, get_model_filename, add_log10Q


def generate_grid(model):
    """ Main function to create fsps grids used by synthesizer """

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f'{synthesizer_data_dir}/grids/{synthesizer_model_name}.hdf5'

    # set up StellarPopulation grid
    imf_type = model['imf_type']
    imf_masses = model['imf_masses']

    if imf_type == 'chabrier03':
        sp = fsps.StellarPopulation(
            imf_type=1, imf_upper_limit=imf_masses[-1], imf_lower_limit=imf_masses[0])
    elif imf_type == 'bpl':
        imf_slopes = model['imf_slopes']
        # NOTE THIS ASSUMES boundaries of 0.5, 1.0

        if (imf_masses[1] != 0.5) or (imf_masses[2] != 1.0):
            # raise exception
            print(
                'WARNING: this IMF definition requires that the boundaries are [m_low, 0.5, 1.0, m_high]')

        sp = fsps.StellarPopulation(
            imf_type=2, imf_upper_limit=imf_masses[-1], imf_lower_limit=imf_masses[0], imf1=imf_slopes[0], imf2=imf_slopes[1], imf3=imf_slopes[2])
    else:
        # raise exception
        pass

    lam = sp.wavelengths  # units: Angstroms
    log10ages = sp.log_age  # units: log10(years)
    ages = 10**log10ages
    metallicities = sp.zlegend  # units: log10(Z)
    log10metallicities = np.log10(metallicities)

    na = len(log10ages)
    nZ = len(metallicities)

    log10Q = np.zeros((na, nZ))  # the ionising photon production rate
    spec = np.zeros((na, nZ, len(lam)))

    for iZ in range(nZ):
        spec_ = sp.get_spectrum(zmet=iZ+1)[1]   # 2D array Lsol / AA
        for ia in range(na):

            fnu = spec_[ia]  # Lsol / Hz
            fnu *= 3.826e33  # erg s^-1 Hz^-1 Msol^-1

            # fnu = convert_flam_to_fnu(lam, flam)
            spec[ia, iZ] = fnu

    # write out model parameters as top level attribute
    for key, value in model.items():
        write_attribute(out_filename, '/', key, (value))

    write_data_h5py(out_filename, 'spectra/wavelength', data=lam, overwrite=True)
    write_attribute(out_filename, 'spectra/wavelength', 'Description',
                    'Wavelength of the spectra grid')
    write_attribute(out_filename, 'spectra/wavelength', 'Units', 'Angstrom')

    write_data_h5py(out_filename, 'log10ages', data=log10ages, overwrite=True)
    write_attribute(out_filename, 'log10ages', 'Description',
                    'Stellar population ages in log10 years')
    write_attribute(out_filename, 'log10ages', 'Units', 'dex(yr)')

    write_data_h5py(out_filename, 'metallicities', data=metallicities, overwrite=True)
    write_attribute(out_filename, 'metallicities', 'Description',
                    'raw abundances')
    write_attribute(out_filename, 'metallicities', 'Units', 'dimensionless [Z]')

    write_data_h5py(out_filename, 'spectra/stellar', data=spec, overwrite=True)
    write_attribute(out_filename, 'spectra/stellar', 'Description',
                    'Three-dimensional spectra grid, [age, metallicity, wavelength]')
    write_attribute(out_filename, 'spectra/stellar', 'Units', 'erg/s/Hz')

    return out_filename


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="BPASS_2.2.1 download and grid creation")
    # parser.add_argument('--download-data', default=False, action='store_true',
    #                     help=("download bpass data directly in current directory "
    #                           "and untar in sunthesizer data directory"))
    #
    # args = parser.parse_args()

    synthesizer_data_dir = '/its/research/astrodata/highz/synthesizer/'

    default_model = {'sps_name': 'fsps',
                     'sps_version': '3.2',
                     'sps_variant': '',
                     'imf_type': 'bpl',  # named IMF or bpl (broken power law)
                     'imf_masses': [0.08, 0.5, 1, 120],
                     'imf_slopes': [1.3, 2.3, 2.3],
                     'alpha': '',
                     }

    # two standard models
    models = [{},  # default model
              {'imf_type': 'chabrier03', 'imf_masses': [0.08, 120]},  # chabrier03
              ]

    # different high-mass slopes
    # models = [{'imf_slopes': [1.3, 2.3, a3]} for a3 in np.arange(1.5, 3.0, 0.1)]

    # different high-mass cut-offs
    # models = [{'imf_masses': [0.08, 0.5, 1, hmc]} for hmc in [5, 10, 50, 100]]

    for model_ in models:

        model = default_model | model_

        synthesizer_model_name = get_model_filename(model)

        print(synthesizer_model_name, model)

        # make grid
        generate_grid(model)

        # this is the full path to the ultimate HDF5 grid file
        out_filename = f'{synthesizer_data_dir}/grids/{synthesizer_model_name}.hdf5'

        # add log10Q for different ions
        add_log10Q(out_filename)
