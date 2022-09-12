"""
Download BPASS v2.3 and convert to HDF5 synthesizer grid.
"""

import sys
import os
from hoki import load
import numpy as np
from utils import write_data_h5py, write_attribute
import gdown
import tarfile
import h5py
from scipy import integrate

from unyt import h, c
from synthesizer.sed import calculate_Q




def download_data(model):

    if model in model_url.keys():
        filename = gdown.download(model_url[model], quiet=False, fuzzy=True)
        return filename
    else:
        print('ERROR: no url for that model')


def untar_data(model, remove_archive = False):


    input_dir = f'{parent_model_dir}/{model}'
    tar = tarfile.open(f'{parent_model_dir}/{model}.tar')
    tar.extractall(path = input_dir)
    tar.close()
    if remove_archive: os.remove(f'{parent_model_dir}/{model}.tar')


def make_single_alpha_grid(model, ae = '+00', bs = 'bin'):

    """ make a grid for a single alpha enhancement """

    _, version, imf = model.split('_') # extract bpass version and imf from model designation. Used to make model name later

    input_dir = f'{parent_model_dir}/{model}'


    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {'zem5': 0.00001, 'zem4': 0.0001, 'z001': 0.001, 'z002': 0.002, 'z003': 0.003, 'z004': 0.004, 'z006': 0.006, 'z008': 0.008, 'z010': 0.01, 'z014': 0.014, 'z020': 0.020, 'z030': 0.030, 'z040': 0.040}
    Z_to_Zk = {k:v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    log10Zs = np.log10(Zs)


    # --- get ages
    fn_ = f'{input_dir}/starmass-{bs}-imf_{imf}.a{ae}.{Z_to_Zk[Zs[0]]}.dat'
    starmass = load.model_output(fn_)
    log10ages = starmass['log_age'].values

    # --- get wavelength grid
    fn_ = f'spectra-{bs}-imf_{imf}.a{ae}.{Z_to_Zk[Zs[0]]}.dat'
    spec = load.model_output(f'{input_dir}/{fn_}')
    wavelengths = spec['WL'].values # \AA
    nu = 3E8/(wavelengths*1E-10)


    nZ = len(log10Zs)
    na = len(log10ages)


    # --- set up outputs

    model_name = f'bpass-{version}-{bs}-{ae}_{imf}' # this is the name of the ultimate HDF5 file

    stellar_mass = np.zeros((na,nZ))
    remnant_mass = np.zeros((na,nZ))

    log10Q = np.zeros((na,nZ)) # the ionising photon production rate
    log10Q_original = np.zeros((na,nZ)) # provided by BPASS, sanity check for above

    spectra = np.zeros((na, nZ, len(wavelengths)))

    out_filename = f'{grid_dir}/{model_name}.h5'

    for iZ, Z in enumerate(Zs):

        print(iZ, Z)

        # --- get remaining and remnant fraction
        fn_ = f'{input_dir}/starmass-{bs}-imf_{imf}.a{ae}.{Z_to_Zk[Z]}.dat'
        starmass = load.model_output(fn_)
        stellar_mass[:, iZ] = starmass['stellar_mass'].values/1E6  # convert to per M_sol
        remnant_mass[:, iZ] = starmass['remnant_mass'].values/1E6  # convert to per M_sol


        # --- get original log10Q
        fn_ = f'{input_dir}/ionizing-{bs}-imf_{imf}.a{ae}.{Z_to_Zk[Z]}.dat'
        ionising = load.model_output(fn_)
        log10Q_original[:, iZ] = ionising['prod_rate'].values - 6   # convert to per M_sol

        # --- get spectra
        fn_ = f'{input_dir}/spectra-{bs}-imf_{imf}.a{ae}.{Z_to_Zk[Z]}.dat'
        spec = load.model_output(fn_)

        for ia, log10age in enumerate(log10ages):

            spec_ = spec[str(log10age)].values # Lsol AA^-1 10^6 Msol^-1

            # --- convert from Llam to Lnu
            spec_ /= 1E6 # Lsol AA^-1 Msol^-1
            spec_ *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
            spec_ *= wavelengths/nu # erg s^-1 Hz^-1 Msol^-1
            spectra[ia, iZ, :] = spec_

            # --- calcualte ionising photon luminosity
            log10Q[ia, iZ] = np.log10(calculate_Q(wavelengths, spec_))




    write_data_h5py(out_filename, 'star_fraction', data=stellar_mass, overwrite=True)
    write_attribute(out_filename, 'star_fraction', 'Description',
                    'Two-dimensional remaining stellar fraction grid, [age,Z]')

    write_data_h5py(out_filename, 'remnant_fraction', data=remnant_mass, overwrite=True)
    write_attribute(out_filename, 'remnant_fraction', 'Description',
                    'Two-dimensional remaining remnant fraction grid, [age,Z]')

    write_data_h5py(out_filename, 'log10Q_original', data=log10Q_original, overwrite=True)
    write_attribute(out_filename, 'log10Q_original', 'Description',
                    'Two-dimensional (original) ionising photon production rate grid, [age,Z]')

    write_data_h5py(out_filename, 'log10Q', data=log10Q, overwrite=True)
    write_attribute(out_filename, 'log10Q', 'Description',
                    'Two-dimensional ionising photon production rate grid, [age,Z]')

    write_data_h5py(out_filename, 'spectra/stellar', data=spectra, overwrite=True)
    write_attribute(out_filename, 'spectra/stellar', 'Description',
                    'Three-dimensional spectra grid, [Z,Age,wavelength]')
    write_attribute(out_filename, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')

    write_data_h5py(out_filename, 'log10ages', data=log10ages, overwrite=True)
    write_attribute(out_filename, 'log10ages', 'Description',
            'Stellar population ages in log10 years')
    write_attribute(out_filename, 'log10ages', 'Units', 'log10(yr)')

    write_data_h5py(out_filename, 'metallicities', data=Zs, overwrite=True)
    write_attribute(out_filename, 'metallicities', 'Description',
            'raw abundances')
    write_attribute(out_filename, 'metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(out_filename, 'log10metallicities', data=log10Zs, overwrite=True)
    write_attribute(out_filename, 'log10metallicities', 'Description',
            'raw abundances in log10')
    write_attribute(out_filename, 'log10metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(out_filename, 'spectra/wavelength', data=wavelengths, overwrite=True)
    write_attribute(out_filename, 'spectra/wavelength', 'Description',
            'Wavelength of the spectra grid')
    write_attribute(out_filename, 'spectra/wavelength', 'Units', 'AA')

    return out_filename


def make_full_grid(model, bs = 'bin'):

    """ make a grid for a single alpha enhancement """



    _, version, imf = model.split('_') # extract bpass version and imf from model designation. Used to make model name later

    input_dir = f'{parent_model_dir}/{model}'


    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {'zem5': 0.00001, 'zem4': 0.0001, 'z001': 0.001, 'z002': 0.002, 'z003': 0.003, 'z004': 0.004, 'z006': 0.006, 'z008': 0.008, 'z010': 0.01, 'z014': 0.014, 'z020': 0.020, 'z030': 0.030, 'z040': 0.040}
    Z_to_Zk = {k:v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    log10Zs = np.log10(Zs)

    # --- create alpha-enhancement grid

    alpha_enhancements = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])  # list of alpha enhancements
    ae_to_aek = {-0.2:'-02', 0.0:'+00', 0.2:'+02', 0.4:'+04', 0.6:'+06'}  # look up dictionary for filename

    # --- get ages
    fn_ = f'{input_dir}/starmass-bin-imf_{imf}.a+00.{Z_to_Zk[Zs[0]]}.dat'
    starmass = load.model_output(fn_)
    log10ages = starmass['log_age'].values

    # --- get wavelength grid
    fn_ = f'spectra-bin-imf_{imf}.a+00.{Z_to_Zk[Zs[0]]}.dat'
    spec = load.model_output(f'{input_dir}/{fn_}')
    wavelengths = spec['WL'].values # \AA
    nu = 3E8/(wavelengths*1E-10)

    na = len(log10ages)
    nZ = len(log10Zs)
    nae = len(alpha_enhancements)



    # --- set up outputs

    model_name = f'bpass-{version}-{bs}_{imf}' # this is the name of the ultimate HDF5 file
    out_filename = f'{grid_dir}/{model_name}.h5' # this is the full path to the ultimate HDF5 grid file

    stellar_mass = np.zeros((na,nZ,nae))
    remnant_mass = np.zeros((na,nZ,nae))

    log10Q = np.zeros((na,nZ,nae)) # the ionising photon production rate
    log10Q_original = np.zeros((na,nZ,nae)) # provided by BPASS, sanity check for above

    spectra = np.zeros((na, nZ, nae, len(wavelengths)))



    for iZ, Z in enumerate(Zs):

        for iae, alpha_enhancement in enumerate(alpha_enhancements):

            print(Z, alpha_enhancement)

            aek = ae_to_aek[alpha_enhancement]
            Zk = Z_to_Zk[Z]

            # --- get remaining and remnant fraction
            fn_ = f'{input_dir}/starmass-{bs}-imf_{imf}.a{aek}.{Zk}.dat'
            starmass = load.model_output(fn_)
            stellar_mass[:, iZ, iae] = starmass['stellar_mass'].values/1E6
            remnant_mass[:, iZ, iae] = starmass['remnant_mass'].values/1E6

            # --- get original log10Q
            fn_ = f'{input_dir}/ionizing-{bs}-imf_{imf}.a{aek}.{Zk}.dat'
            ionising = load.model_output(fn_)
            log10Q_original[:, iZ, iae] = ionising['prod_rate'].values - 6   # convert to per M_sol

            # --- get spectra
            fn_ = f'{input_dir}/spectra-{bs}-imf_{imf}.a{aek}.{Zk}.dat'
            spec = load.model_output(fn_)

            for ia, log10age in enumerate(log10ages):

                spec_ = spec[str(log10age)].values # Lsol AA^-1 10^6 Msol^-1

                # --- convert from Llam to Lnu
                spec_ /= 1E6 # Lsol AA^-1 Msol^-1
                spec_ *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
                spec_ *= wavelengths/nu # erg s^-1 Hz^-1 Msol^-1

                spectra[ia, iZ, iae, :] = spec_ # Lsol AA^-1 10^6 Msol^-1

                # --- calcualte ionising photon luminosity
                log10Q[ia, iZ, iae] = np.log10(calculate_Q(wavelengths, spec_))



    write_data_h5py(out_filename, 'star_fraction', data=stellar_mass, overwrite=True)
    write_attribute(out_filename, 'star_fraction', 'Description',
                    'Two-dimensional remaining stellar fraction grid, [age,Z]')

    write_data_h5py(out_filename, 'remnant_fraction', data=remnant_mass, overwrite=True)
    write_attribute(out_filename, 'remnant_fraction', 'Description',
                    'Two-dimensional remaining remnant fraction grid, [age,Z]')

    write_data_h5py(out_filename, 'log10Q_original', data=log10Q_original, overwrite=True)
    write_attribute(out_filename, 'log10Q_original', 'Description',
                    'Two-dimensional (original) ionising photon production rate grid, [age,Z]')

    write_data_h5py(out_filename, 'log10Q', data=log10Q, overwrite=True)
    write_attribute(out_filename, 'log10Q', 'Description',
                    'Two-dimensional ionising photon production rate grid, [age,Z]')


    write_data_h5py(out_filename, 'spectra/stellar', data=spectra, overwrite=True)
    write_attribute(out_filename, 'spectra/stellar', 'Description',
                    'Three-dimensional spectra grid, [age, Z, ae, wavelength]')
    write_attribute(out_filename, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')

    write_data_h5py(out_filename, 'log10ages', data=log10ages, overwrite=True)
    write_attribute(out_filename, 'log10ages', 'Description',
            'Stellar population ages in log10 years')
    write_attribute(out_filename, 'log10ages', 'Units', 'log10(yr)')

    write_data_h5py(out_filename, 'metallicities', data=Zs, overwrite=True)
    write_attribute(out_filename, 'metallicities', 'Description',
            'raw abundances')
    write_attribute(out_filename, 'metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(out_filename, 'log10metallicities', data=log10Zs, overwrite=True)
    write_attribute(out_filename, 'log10metallicities', 'Description',
            'raw abundances in log10')
    write_attribute(out_filename, 'log10metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(out_filename, 'alpha_enhancements', data=alpha_enhancements, overwrite=True)
    write_attribute(out_filename, 'alpha_enhancements', 'Description',
            'alpha enhancements in log10')
    write_attribute(out_filename, 'alpha_enhancements', 'Units', 'dimensionless')

    write_data_h5py(out_filename, 'spectra/wavelength', data=wavelengths, overwrite=True)
    write_attribute(out_filename, 'spectra/wavelength', 'Description',
            'Wavelength of the spectra grid')
    write_attribute(out_filename, 'spectra/wavelength', 'Units', 'AA')

    return out_filename










if __name__ == "__main__":

    synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')
    parent_model_dir = f'{synthesizer_data_dir}/input_files/bpass/'
    grid_dir = f'{synthesizer_data_dir}/grids'

    models = ['bpass_v2.3_chab300']

    for model in models:

        # download_data(model)
        # untar_data(model)

        for bs in ['bin']:

            # make a grid with a single alpha enahancement value
            for ae in ['-02','+00','+02','+04','+06']:
                out_filename = make_single_alpha_grid(model, ae = ae, bs = bs)


            # make a full grid
            # out_filename = make_full_grid(model, bs = bs)
