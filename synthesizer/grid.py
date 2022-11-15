"""
Create a Grid object
"""

import os
import numpy as np
import h5py
import cmasher as cmr
import matplotlib.pyplot as plt

from . import __file__ as filepath
from .plt import mlabel
from .sed import Sed, convert_fnu_to_flam


def parse_grid_id(grid_id):
    """
    This is used for parsing a grid ID to return the SPS model, version, and IMF 
    """

    if len(grid_id.split('_')) == 2:
        sps_model_, imf_ = grid_id.split('_')
        cloudy = cloudy_model = ''

    if len(grid_id.split('_')) == 4:
        sps_model_, imf_, cloudy, cloudy_model = grid_id.split('_')

    if len(sps_model_.split('-')) == 1:
        sps_model = sps_model_.split('-')[0]
        sps_model_version = ''

    if len(sps_model_.split('-')) == 2:
        sps_model = sps_model_.split('-')[0]
        sps_model_version = sps_model_.split('-')[1]

    if len(sps_model_.split('-')) > 2:
        sps_model = sps_model_.split('-')[0]
        sps_model_version = '-'.join(sps_model_.split('-')[1:])

    if len(imf_.split('-')) == 1:
        imf = imf_.split('-')[0]
        imf_hmc = ''

    if len(imf_.split('-')) == 2:
        imf = imf_.split('-')[0]
        imf_hmc = imf_.split('-')[1]

    if imf in ['chab', 'chabrier03', 'Chabrier03']:
        imf = 'Chabrier (2003)'
    if imf in ['kroupa']:
        imf = 'Kroupa (2003)'
    if imf in ['salpeter', '135all']:
        imf = 'Salpeter (1955)'
    if imf.isnumeric():
        imf = rf'$\alpha={float(imf)/100}$'

    return {'sps_model': sps_model, 'sps_model_version': sps_model_version,
            'imf': imf, 'imf_hmc': imf_hmc}


class Grid:
    """
    Base class containing useful functions for operating on grids
    """

    def get_nearest_index(self, value, array):

        return (np.abs(array - value)).argmin()

    def get_nearest(self, value, array):

        idx = self.get_nearest_index(value, array)

        return idx, array[idx]

    def get_nearest_log10Z(self, log10metallicity):

        return self.get_nearest(log10metallicity, self.log10metallicities)

    def get_nearest_log10age(self, log10age):

        return self.get_nearest(log10age, self.log10ages)


class SpectralGrid(Grid):
    """
    This provides an object to hold the SPS / Cloudy grid
    for use by other parts of the code
    """

    def __init__(self, grid_name, grid_dir=None, verbose=False):

        if not grid_dir:

            grid_dir = os.path.join(os.path.dirname(filepath), 'data/grids')

        print(grid_dir)

        # if synthesizer_data_dir:
        #     grid_filename = f'{synthesizer_data_dir}/grids/{grid_name}.h5'
        # else:
        #     grid_filename = f'{grid_name}.h5'
        #     grid_name = grid_filename.split('/')[-1]

        self.grid_dir = grid_dir
        self.grid_name = grid_name

        with h5py.File(f'{self.grid_dir}/{self.grid_name}.h5', 'r') as hf:
            self.spec_names = list(hf['spectra'].keys())
            self.spec_names.remove('wavelength')

            self.lam = hf['spectra/wavelength'][:]
            self.nu = 3E8/(self.lam*1E-10)

            self.log10ages = hf['log10ages'][:]
            self.ages = 10**self.log10ages
            self.metallicities = hf['metallicities'][:]
            self.log10metallicities = hf['log10metallicities'][:]
            self.log10Zs = self.log10metallicities  # alias

            if 'log10Q' in hf.keys():
                self.log10Q = hf['log10Q'][:]
                self.log10Q[self.log10Q != self.log10Q] = -99.99

        if verbose:
            print(f'metallicities: {self.metallicities}')
            print(f'ages: {self.ages}')
            print(f'ages: {self.log10ages}')

        self.spectra = {}

        for spec_name in self.spec_names:

            with h5py.File(f'{self.grid_dir}/{self.grid_name}.h5', 'r') as hf:
                self.spectra[spec_name] = hf['spectra'][spec_name][:]

            if spec_name == 'incident':
                self.spectra['stellar'] = self.spectra[spec_name]

        """ if full cloudy grid available calculate
        some other spectra for convenience """
        if 'linecont' in self.spec_names:

            self.spectra['total'] = self.spectra['transmitted'] +\
                self.spectra['nebular']  #  assumes fesc = 0

            self.spectra['nebular_continuum'] = self.spectra['nebular'] -\
                self.spectra['linecont']

        if verbose:
            print('available spectra:', list(self.spectra.keys()))

    def get_sed(self, ia, iZ, spec_name='stellar'):

        return Sed(self.lam, lnu=self.spectra[spec_name][ia, iZ])
    
    # TODO: move to plotting script to remove cmasher dependency
    def plot_log10Q(self, hsize=3.5, vsize=2.5, cmap=cmr.sapphire,
                    vmin=42.5, vmax=47.5, max_log10age=9.):

        left = 0.2
        height = 0.6
        bottom = 0.15
        width = 0.75

        if not vsize:
            vsize = hsize*width/height

        fig = plt.figure(figsize=(hsize, vsize))

        ax = fig.add_axes((left, bottom, width, height))
        cax = fig.add_axes([left, bottom+height, width, 0.03])

        y = np.arange(len(self.metallicities))

        log10Q = self.log10Q

        if max_log10age:
            ia_max = self.get_nearest_index(max_log10age, self.log10ages)
            log10Q = log10Q[:ia_max, :]
        else:
            ia_max = -1

        """ this is technically incorrect because metallicity
        is not on an actual grid."""
        ax.imshow(log10Q.T, origin='lower', extent=[self.log10ages[0],
                  self.log10ages[ia_max], y[0]-0.5, y[-1]+0.5], cmap=cmap,
                  aspect='auto', vmin=vmin, vmax=vmax)

        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(r'$\rm log_{10}(\dot{n}_{LyC}/s^{-1})$')

        ax.set_yticks(y, self.metallicities)

        ax.minorticks_off()
        ax.set_xlabel(mlabel('log_{10}(age/yr)'))
        ax.set_ylabel(mlabel('Z'))

        return fig, ax


class LineGrid(Grid):

    """ This provides an object to hold the SPS / Cloudy grid for use by other parts of the code """

    def __init__(self, grid_name, verbose=False):

        if synthesizer_data_dir:
            grid_filename = f'{synthesizer_data_dir}/grids/{grid_name}.h5'
        else:
            grid_filename = f'{grid_name}.h5'
            grid_name = grid_filename.split('/')[-1]

        self.grid_name = grid_name

        with h5py.File(grid_filename, 'r') as hf:
            self.lines = hf['lines']
            self.line_list = self.lines.attrs['lines']
            self.log10ages = hf['log10ages'][()]
            self.ages = 10**self.log10ages
            self.metallicities = hf['metallicities'][()]
            self.log10metallicities = hf['log10metallicities'][()]
            self.log10Zs = self.log10metallicities  # alias

        self.na = len(self.ages)
        self.nZ = len(self.metallicities)

        if verbose:
            print(f'metallicities: {self.metallicities}')
            print(f'ages: {self.ages}')
            print(f'ages: {self.log10ages}')
            print(f'available lines: {self.line_list}')

    def get_line_info(self, line_id, ia, iZ):
        """ return the equivalent width of a given line and age, metalliciy """

        if type(line_id) is str:
            line_id = [line_id]

        line_luminosity = 0.0
        continuum_nu = []
        wv = []

        for line_id_ in line_id:
            line = self.lines[line_id_]
            wv.append(line.attrs['wavelength'])  # \AA
            line_luminosity += line['luminosity'][ia, iZ]  # line luminosity, erg/s
            #  continuum at line wavelength, erg/s/Hz
            continuum_nu.append(line['continuum'][ia, iZ])

        continuum_lam = convert_fnu_to_flam(np.mean(wv), np.mean(
            continuum_nu))  # continuum at line wavelength, erg/s/AA
        ew = line_luminosity / continuum_lam  # AA

        return np.mean(wv), line_luminosity, ew
