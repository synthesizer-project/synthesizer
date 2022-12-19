"""
Download Populations 3 star spectra from Yggdrasil models and convert to HDF5
synthesizer grid.

There are 3 versions implemented:
    1. Pop III.1 - A zero-metallicity population with an extremely top-heavy
                   IMF (50-500 Msolar, Salpeter slope), using a SSP from
                   Schaerer et al. (2002, A&A, 382, 28)
    2. Pop III.2 - A zero-metallicity population with a moderately top-heavy
                   IMF (log-normal with characteristic mass M_c=10 Msolar,
                   dispersion sigma=1 Msolar and wings extending from 1-500
                   Msolar) from Raiter et al. (2010, A&A 523, 64)
    3. Pop III, Kroupa IMF - A zero-metallicity population with a normal
                             IMF (universal Kroupa 2001 IMF in the interval
                             0.1-100 Msolar), based on a rescaled SSP from
                             Schaerer et al. (2002, A&A, 382, 28)

We also just pick the instantaneous burst model with the 3 different gas
covering factor of 0 (no nebular contribution), 0.5, 1 (maximal nebular
contribution)

Warning: the nebular procesed grids here differ from the rest of the
nebular processing implementation in synthesizer, where we self consistently
run pure stellar spectra through CLOUDY. For full self consistency the
nebular grids here should not be used, but we provide anyway for reference.
"""

import numpy as np
import os
import pathlib
import re
import subprocess
import argparse
from utils import write_data_h5py, write_attribute

from synthesizer.sed import calculate_Q


def download_data(synthesizer_data_dir, ver, fcov):

    filename = F"PopIII{ver}_fcov_{fcov}_SFR_inst_Spectra"
    url = F"https://www.astro.uu.se/~ez/yggdrasil/YggdrasilSpectra/{filename}"

    subprocess.call('wget --no-check-certificate ' + url, shell=True)

    mv_folder = F"{synthesizer_data_dir}/input_files/popIII/Yggdrasil/"
    pathlib.Path(mv_folder).mkdir(parents=True, exist_ok=True)
    subprocess.call(F'mv {filename} {mv_folder}', shell=True)

    return mv_folder+filename


def convertPOPIII(synthesizer_data_dir, ver, fcov):
    """Convert POPIII outputs

    """

    fileloc = download_data(synthesizer_data_dir, ver, fcov)

    # Initialise ---------------------------------------------------------
    ageBins = None
    lambdaBins = None
    metalBins = np.array([0])
    seds = np.array([[[None]]])

    print('Reading POPIII files and converting...')
    # Open SED table containing different ages
    print('Converting file ', fileloc)
    data = open(fileloc, 'r')
    text = data.read()

    # Get age values
    ages = re.findall(r'Age\s(.*?)\n', text)
    ageBins = np.array([float(re.findall(
            r" [+\-]?[^\w]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)",
            ages[ii])[0]) for ii in range(len(ages))])

    # Get the number of wavelength points: lam_num
    lam_num = np.array(re.findall(r'points:(.*?)\n', text), dtype=int)
    diff = np.diff(lam_num)
    if np.sum(diff) != 0:
        print('Age bins are not identical everywhere!!!')
        print('CANCELLING CONVERSION!!!')
        return

    seds = np.zeros((len(ageBins), len(metalBins), lam_num[0]))

    """ Format of the file is 10 header lines at begining followed by
        lam_num lines of wavelength and flux, then one empty line and
        7 string lines giving the ages """
    data = open(fileloc, 'r')
    tmp = data.readlines()
    mass = np.float(re.findall(r"\d+\.\d+", tmp[0])[0])
    begin = 9
    end = begin + lam_num[0]
    for ii in range(len(ageBins)):

        this_data = tmp[begin:end]
        if ii == 0:
            lambdaBins = np.array([float(re.findall(
                r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', jj)[0]) for jj in this_data])

        seds[ii, 0] = np.array([float(re.findall(
            r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', jj)[1]) *
            (10**float(re.findall(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', jj)[2]))
            for jj in this_data])

        begin = end + 8
        end = begin + lam_num[0]

    return (np.array(seds/mass, dtype=np.float64),
            np.array(metalBins, dtype=np.float64),
            np.array(ageBins, dtype=np.float64),
            np.array(lambdaBins, dtype=np.float64))


def make_grid(synthesizer_data_dir, ver, fcov):
    """ Main function to convert POPIII grids and
        produce grids used by synthesizer """

    # Define base path
    # basepath = (f"{synthesizer_data_dir}/input_files/popIII/"
    #             "Yggdrasil/")

    # Define output
    if not os.path.exists(f'{synthesizer_data_dir}/grids/'):
        os.makedirs(f'{synthesizer_data_dir}/grids/')

    model_name = F'popIII{ver}_fcov_{fcov}'
    fname = f'{synthesizer_data_dir}/grids/{model_name}.h5'

    # Get spectra and attributes
    out = convertPOPIII(synthesizer_data_dir, ver, fcov)

    metallicities = out[1]
    log10metallicities = np.log10(metallicities)

    ages = out[2] * 1e6  # since ages are quoted in Myr
    log10ages = np.log10(ages)

    lam = out[3]
    nu = 3E8/(lam*1E-10)

    spec = out[0]

    spec *= lam/nu  # erg s^-1 Hz^-1 Msol^-1

    na = len(ages)
    nZ = len(metallicities)

    log10Q = np.zeros((na, nZ))  # the ionising photon production rate

    for iZ, metallicity in enumerate(metallicities):
        for ia, log10age in enumerate(log10ages):

            # --- calcualte ionising photon luminosity
            log10Q[ia, iZ] = np.log10(calculate_Q(lam, spec[ia, iZ, :]))

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

    write_data_h5py(fname, 'log10metallicities', data=log10metallicities,
                    overwrite=True)
    write_attribute(fname, 'log10metallicities', 'Description',
                    'raw abundances in log10')
    write_attribute(fname, 'log10metallicities', 'Units',
                    'dimensionless [log10(Z)]')

    write_data_h5py(fname, 'log10Q', data=log10Q, overwrite=True)
    write_attribute(fname, 'log10Q', 'Description',
                    ("Two-dimensional ionising photon "
                     "production rate grid, [age,Z]"))

    write_data_h5py(fname, 'spectra/wavelength', data=lam, overwrite=True)
    write_attribute(fname, 'spectra/wavelength', 'Description',
                    'Wavelength of the spectra grid')
    write_attribute(fname, 'spectra/wavelength', 'Units', 'AA')

    if fcov == '0':
        write_data_h5py(fname, 'spectra/stellar', data=spec, overwrite=True)
        write_attribute(fname, 'spectra/stellar', 'Description',
                        """Three-dimensional spectra grid, [age, metallicity
                        , wavelength]""")
        write_attribute(fname, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')
    else:
        write_data_h5py(fname, 'spectra/nebular', data=spec, overwrite=True)
        write_attribute(fname, 'spectra/nebular', 'Description',
                        """Three-dimensional spectra grid, [age, metallicity
                        , wavelength]""")
        write_attribute(fname, 'spectra/nebular', 'Units', 'erg s^-1 Hz^-1')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Install the POPIII grid to the specified directory.')
    parser.add_argument("-dir", "--directory", type=str, required=True)
    args = parser.parse_args()

    synthesizer_data_dir = args.directory

    # Different forms of the IMFs
    vers = np.array(['.1', '.2', '_kroupa_IMF'])
    # different gas covering fractions for nebular emission model
    fcovs = np.array(['0', '0.5', '1'])

    for ver in vers:
        for fcov in fcovs:
            make_grid(synthesizer_data_dir, ver, fcov)
