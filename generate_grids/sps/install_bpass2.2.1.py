"""
Download BPASS v2.2.1 and convert to HDF5 synthesizer grid.
"""

import os
import argparse
import numpy as np
import gdown
import tarfile

from hoki import load

from utils import write_data_h5py, write_attribute, add_log10Q


def download_data(model):

    model_url = {}
    model_url['bpass_v2.2.1_chab100-bin'] = \
            ("https://drive.google.com/file/d/"
             "1az7_hP3RDovr-BN9sXgDuaYqOZHHUeXD/view?usp=sharing")
    model_url['bpass_v2.2.1_chab300-bin'] = \
            ("https://drive.google.com/file/d/"
             "1JcUM-qyOQD16RdfWjhGKSTwdNfRUW4Xu/view?usp=sharing")
    print(model_url)
    if model in model_url.keys():
        filename = gdown.download(model_url[model], quiet=False, fuzzy=True)
        return filename
    else:
        raise ValueError('ERROR: no url for that model')


def untar_data(model, filename, synthesizer_data_dir):

    model_dir = f'{synthesizer_data_dir}/input_files/bpass/{model}'
    with tarfile.open(filename) as tar:
        tar.extractall(path=model_dir)
    
    os.remove(filename)


def make_grid(model):
    
    # extract bpass version and imf from model designation.
    _, version, imf = model.split('_')
    imf = imf.split('-')[0]

    input_dir = f'{synthesizer_data_dir}/input_files/bpass/{model}/'

    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {'zem5': 0.00001, 'zem4': 0.0001, 'z001': 0.001,
               'z002': 0.002, 'z003': 0.003, 'z004': 0.004,
               'z006': 0.006, 'z008': 0.008, 'z010': 0.01,
               'z014': 0.014, 'z020': 0.020, 'z030': 0.030,
               'z040': 0.040}

    Z_to_Zk = {k: v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    log10Zs = np.log10(Zs)
    # print('metallicities', Zs)

    for bs in ['bin', 'sin']:
    
        # --- get ages
        fn_ = f'starmass-{bs}-imf_{imf}.{Z_to_Zk[Zs[0]]}.dat.gz'
        print(f'{input_dir}/{fn_}')
        starmass = load.model_output(f'{input_dir}/{fn_}')
        log10ages = starmass['log_age'].values
    
        nZ = len(log10Zs)
        na = len(log10ages)
    
        # --- get wavelength grid
        fn_ = f'spectra-{bs}-imf_{imf}.{Z_to_Zk[Zs[0]]}.dat.gz'
        spec = load.model_output(f'{input_dir}/{fn_}')
        wavelengths = spec['WL'].values  # \AA
        nu = 3E8/(wavelengths*1E-10)

        # this is the name of the ultimate HDF5 file
        model_name = f'bpass-{version}-{bs}_{imf}'

        # this is the full path to the ultimate HDF5 grid file
        out_filename = f'{grid_dir}/{model_name}.h5'

        stellar_mass = np.zeros((na, nZ))
        remnant_mass = np.zeros((na, nZ))
        spectra = np.zeros((na, nZ, len(wavelengths)))

        for iZ, Z in enumerate(Zs):

            # --- get remaining and remnant fraction
            fn_ = f'starmass-{bs}-imf_{imf}.{Z_to_Zk[Z]}.dat.gz'
            starmass = load.model_output(f'{input_dir}/{fn_}')
            stellar_mass[:, iZ] = starmass['stellar_mass'].values/1E6
            remnant_mass[:, iZ] = starmass['remnant_mass'].values/1E6

            # --- get spectra
            fn_ = f'spectra-{bs}-imf_{imf}.{Z_to_Zk[Z]}.dat.gz'
            spec = load.model_output(f'{input_dir}/{fn_}')

            for ia, log10age in enumerate(log10ages):
                # Lsol AA^-1 10^6 Msol^-1
                spectra[ia, iZ, :] = spec[str(log10age)].values

        spectra /= 1E6  # Lsol AA^-1 Msol^-1
        spectra *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
        spectra *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1

        write_data_h5py(out_filename, 'star_fraction', data=stellar_mass,
                        overwrite=True)
        write_attribute(out_filename, 'star_fraction', 'Description',
                        ('Two-dimensional remaining stellar '
                         'fraction grid, [age,Z]'))

        write_data_h5py(out_filename, 'remnant_fraction', data=remnant_mass,
                        overwrite=True)
        write_attribute(out_filename, 'remnant_fraction', 'Description',
                        ('Two-dimensional remaining remnant '
                         'fraction grid, [age,Z]'))

        write_data_h5py(out_filename, 'spectra/stellar', data=spectra,
                        overwrite=True)
        write_attribute(out_filename, 'spectra/stellar', 'Description',
                        'Three-dimensional spectra grid, [Z,Age,wavelength]')
        write_attribute(out_filename, 'spectra/stellar', 'Units',
                        'erg s^-1 Hz^-1')

        write_data_h5py(out_filename, 'log10ages', data=log10ages,
                        overwrite=True)
        write_attribute(out_filename, 'log10ages', 'Description',
                        'Stellar population ages in log10 years')
        write_attribute(out_filename, 'log10ages', 'Units', 'log10(yr)')

        write_data_h5py(out_filename, 'metallicities', data=Zs, overwrite=True)
        write_attribute(out_filename, 'metallicities', 'Description',
                        'raw abundances')
        write_attribute(out_filename, 'metallicities', 'Units',
                        'dimensionless [log10(Z)]')

        # write_data_h5py(out_filename, 'log10metallicities', data=log10Zs,
        #                 overwrite=True)
        # write_attribute(out_filename, 'log10metallicities', 'Description',
        #                 'raw abundances in log10')
        # write_attribute(out_filename, 'log10metallicities', 'Units',
        #                 'dimensionless [log10(Z)]')

        write_data_h5py(out_filename, 'spectra/wavelength', data=wavelengths,
                        overwrite=True)
        write_attribute(out_filename, 'spectra/wavelength', 'Description',
                        'Wavelength of the spectra grid')
        write_attribute(out_filename, 'spectra/wavelength', 'Units', 'AA')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="BPASS_2.2.1 download and grid creation")
    parser.add_argument('--download-data', default=False, action='store_true',
                        help=("download bpass data directly in current directory "
                             "and untar in sunthesizer data directory"))

    args = parser.parse_args()

    synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')
    grid_dir = f'{synthesizer_data_dir}/grids'

    models = ['bpass_v2.2.1_chab100-bin']
 
    for model in models:
        
        if args.download_data:
            filename = download_data(model)
            untar_data(model, filename, synthesizer_data_dir)

        make_grid(model)

        for bs in ['sin', 'bin']:
            filename = f'{synthesizer_data_dir}/grids/{model}.h5'
            add_log10Q(filename)
