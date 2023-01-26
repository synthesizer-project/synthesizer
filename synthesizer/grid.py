"""
Create a Grid object
"""

import os
import numpy as np
import h5py
import cmasher as cmr
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from . import __file__ as filepath
from .plt import mlabel
from .sed import Sed, convert_fnu_to_flam


from collections.abc import Iterable


def get_available_lines(grid_name, grid_dir, include_wavelengths=False):
    """Get a list of the lines available to a grid

    Parameters
    ----------
    grid_name : str
        list containing lists and/or strings and integers

    grid_dir : str
        path to grid

    Returns
    -------
    list
        list of lines
    """

    grid_filename = f'{grid_dir}/{grid_name}.hdf5'
    with h5py.File(grid_filename, 'r') as hf:

        lines = list(hf['lines'].keys())

        if include_wavelengths:
            wavelengths = np.array([hf['lines'][line].attrs['wavelength'] for line in lines])
            return lines, wavelengths
        else:
            return lines


def flatten_linelist(list_to_flatten):
    """Flatten a mixed list of lists and strings and remove duplicates

    Flattens a mixed list of lists and strings. Used when converting a desired line list which may contain single lines and doublets.

    Parameters
    ----------
    list : list
        list containing lists and/or strings and integers


    Returns
    -------
    list
        flattend list
    """

    flattend_list = []
    for l in list_to_flatten:

        if isinstance(l, list) or isinstance(l, tuple):
            for ll in l:
                flattend_list.append(ll)

        elif isinstance(l, str):

            # --- if the line is a doublet resolve it and add each line individually
            if len(l.split(',')) > 1:
                flattend_list += l.split(',')
            else:
                flattend_list.append(l)

        else:
            # raise exception
            pass

    return list(set(flattend_list))


def parse_grid_id(grid_id):
    """
    This is used for parsing a grid ID to return the SPS model,
    version, and IMF
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


class Grid():
    """
    The Grid class, containing attributes and methods for reading and manipulating spectral grids

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, grid_name, grid_dir=None, verbose=False, read_spectra=True, read_lines=False):

        if not grid_dir:
            grid_dir = os.path.join(os.path.dirname(filepath), 'data/grids')

        self.grid_dir = grid_dir
        self.grid_name = grid_name
        self.grid_filename = f'{self.grid_dir}/{self.grid_name}.hdf5'

        self.spectra = None
        self.lines = None

        # convert line list into flattend list and remove duplicates
        if isinstance(read_lines, list):
            read_lines = flatten_linelist(read_lines)

        with h5py.File(self.grid_filename, 'r') as hf:
            self.spec_names = list(hf['spectra'].keys())
            self.spec_names.remove('wavelength')

            self.lam = hf['spectra/wavelength'][:]
            self.nu = 3E8/(self.lam*1E-10)

            self.log10ages = hf['log10ages'][:]
            self.ages = 10**self.log10ages

            self.metallicities = hf['metallicities'][:]
            self.log10metallicities = np.log10(self.metallicities)
            # TODO: why do we need this?
            self.log10Zs = self.log10metallicities  # alias

            if 'log10Q' in hf.keys():

                # backwards compatability
                if isinstance(hf['log10Q'], h5py.Dataset):
                    self.log10Q = hf['log10Q'][:]
                else:
                    self.log10Q = hf['log10Q/HI'][:]
                self.log10Q[self.log10Q != self.log10Q] = -99.99

            # self.units = {}
            # self.units['log10ages'] = hf['log10ages'].attrs['Units']
            # self.units['log10metallicities'] = hf['log10ages'].attrs['Units']
            # self.units['lam'] = hf['spectra/wavelength'].attrs['Units']

        if read_spectra:

            self.spectra = {}

            for spec_name in self.spec_names:

                with h5py.File(f'{self.grid_dir}/{self.grid_name}.hdf5', 'r') as hf:
                    self.spectra[spec_name] = hf['spectra'][spec_name][:]
                    # self.units[f'spectra/{spec_name}'] = hf['spectra'][spec_name].attrs['Units']

                if spec_name == 'incident':
                    self.spectra['stellar'] = self.spectra[spec_name]
                    # self.units[f'spectra/stellar'] = hf['spectra'][spec_name].attrs['Units']

            """ if full cloudy grid available calculate
            some other spectra for convenience """
            if 'linecont' in self.spec_names:

                self.spectra['total'] = self.spectra['transmitted'] +\
                    self.spectra['nebular']  #  assumes fesc = 0

                self.spectra['nebular_continuum'] = self.spectra['nebular'] -\
                    self.spectra['linecont']

            if verbose:
                print('available spectra:', list(self.spectra.keys()))

        if read_lines:

            self.lines = {}

            if isinstance(read_lines, list):
                self.line_list = read_lines
            else:
                self.line_list = hf['lines'].attrs['lines']  # apparently this doesn't exist

            with h5py.File(f'{self.grid_dir}/{self.grid_name}.hdf5', 'r') as hf:

                for line in self.line_list:

                    self.lines[line] = {}
                    self.lines[line]['wavelength'] = hf['lines'][line].attrs['wavelength']  # angstrom
                    self.lines[line]['luminosity'] = hf['lines'][line]['luminosity'][:]
                    self.lines[line]['continuum'] = hf['lines'][line]['continuum'][:]

    def get_nearest_index(self, value, array):
        """
        Simple function for calculating the closest index in an array for a given value

        Parameters
        ----------
        value : float
            The target value

        array : nn.ndarray
            The array to search

        Returns
        -------
        int
             The index of the closet point in the grid (array)
        """

        return (np.abs(array - value)).argmin()

    def get_nearest(self, value, array):
        """
        Simple function for calculating the closest index in an array for a given value

        Parameters
        ----------
        value : float
            The target value

        array : nn.ndarray
            The array to search

        Returns
        -------
        int
             The index of the closet point in the grid (array)
        """

        idx = self.get_nearest_index(value, array)

        return idx, array[idx]

    def get_nearest_log10Z(self, log10metallicity):

        return self.get_nearest(log10metallicity, self.log10metallicities)

    def get_nearest_log10age(self, log10age):

        return self.get_nearest(log10age, self.log10ages)

    def get_sed(self, ia, iZ, spec_name='stellar'):
        """
        Simple function for calculating the closest index in an array for a given value

        Parameters
        ----------
        ia : int
            the age grid point

        iZ : int
            the metallicity grid point

        Returns
        -------
        obj (Sed)
             An Sed object at the defined grid point
        """

        return Sed(self.lam, lnu=self.spectra[spec_name][ia, iZ])

    # TODO: move to plotting script to remove cmasher dependency
    def plot_log10Q(self, hsize=3.5, vsize=2.5, cmap=cmr.sapphire,
                    vmin=42.5, vmax=47.5, max_log10age=9.):

        left = 0.2
        height = 0.65
        bottom = 0.15
        width = 0.75

        if not vsize:
            vsize = hsize*width/height

        fig = plt.figure(figsize=(hsize, vsize))

        ax = fig.add_axes((left, bottom, width, height))
        cax = fig.add_axes([left, bottom+height+0.01, width, 0.05])

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

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        fig.colorbar(cmapper, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(r'$\rm log_{10}(\dot{n}_{LyC}/s^{-1}\ M_{\odot}^{-1})$')
        cax.set_yticks([])

        ax.set_yticks(y, self.metallicities)
        ax.minorticks_off()
        ax.set_xlabel(mlabel('log_{10}(age/yr)'))
        ax.set_ylabel(mlabel('Z'))

        return fig, ax

    # I'm not convinced this is necessary anymore

    def fetch_line(self, line_id, save=True):
        """
        Fetch line information from the grid HDF5 file
        Parameters:
        line_id (str): unique line identifier
        Returns:
        luminosity (ndarray)
        continuum (ndarray)
        wavelength (float)
        """

        with h5py.File(self.grid_filename, 'r') as hf:
            luminosity = hf[f'lines/{line_id}/luminosity'][:]
            continuum = hf[f'lines/{line_id}/continuum'][:]
            wavelength = hf[f'lines/{line_id}'].attrs['wavelength']

        if save:
            self.lines[line_id] = {}
            self.lines[line_id]['luminosity'] = luminosity
            self.lines[line_id]['continuum'] = continuum
            self.lines[line_id]['wavelength'] = wavelength

        return {'luminosity': luminosity,
                'continuum': continuum,
                'wavelength': wavelength}

    def get_line_info(self, line_id, ia, iZ):
        """
        return the equivalent width of a line (or line combination) for a given age and metalliciy

        Parameters:

        line_id (list or str): unique line identification string
        ia (int): age index
        iZ (int): metallicity index
        save_line_info (bool): if fetch_line required, determines whether
                               we save the line properties to the grid
                               object (True), or load them on the fly (False)

        Returns:
        wavelength (float)
        line_luminosity (float)
        ew (float): line equivalent width
        """

        if type(line_id) is str:
            line_id = [line_id]

        line_luminosity = 0.0
        continuum_nu = []
        wv = []

        for _lid in line_id:

            line = self.lines[_lid]

            wavelength = line['wavelength']
            luminosity = line['luminosity']
            continuum = line['continuum']

            wv.append(wavelength)  # \AA
            line_luminosity += luminosity[ia, iZ]  # line luminosity, erg/s

            #  continuum at line wavelength, erg/s/Hz
            continuum_nu.append(continuum[ia, iZ])

        continuum_lam = convert_fnu_to_flam(np.mean(wv), np.mean(
            continuum_nu))  # continuum at line wavelength, erg/s/AA
        ew = line_luminosity / continuum_lam  # AA

        return {'name': ','.join(line_id),
                'wavelength': np.mean(wv),
                'luminosity': line_luminosity,
                'equivalent_width': ew}
