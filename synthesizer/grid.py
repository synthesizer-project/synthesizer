"""
Create a Grid object
"""


import os
import numpy as np
import h5py

from synthesizer.utils import (load_arr, get_names_h5py)


# Get sythesizer_data_dir
sythesizer_data_dir = os.getenv('SYNTHESIZER_DATA')

if not sythesizer_data_dir:
    print('WARNING: SYNTHESIZER_DATA environment variable not set. SpectralGrid may not work properly unless you provide the full path to the grid file (excluding the extension)')


# --- note from Steve. I've superceded sps_grid with SpectralGrid below.

class sps_grid:

    def __init__(self, grid_file, verbose=False, lines=False, line_type='emergent', max_age=7):

        self.grid_file = grid_file
        self.max_age = max_age  # log10(yr)

        if verbose:
            print("Loading model from: \n\n%s\n" % (grid_file))

        self.spectra = load_arr('spectra', grid_file)
        self.ages = load_arr('ages', grid_file)
        self.metallicities = load_arr('metallicities', grid_file)
        self.wl = load_arr('wavelength', grid_file)

        if self.ages[0] > self.ages[1]:
            if verbose:
                print("Age array not sorted ascendingly. Sorting...\n")

            self.ages = self.ages[::-1]
            self.spectra = self.spectra[:, ::-1, :]

        if self.metallicities[0] > self.metallicities[1]:
            if verbose:
                print("Metallicity array not sorted ascendingly. Sorting...\n")

            self.metallicities = self.metallicities[::-1]
            self.spectra = self.spectra[::-1, :, :]

        if lines:
            self.load_lines_grid(line_type='emergent')

    def load_lines_grid(self, line_type='emergent'):
        self.lines = get_names_h5py(self.grid_file, f'lines/{line_type}')

        line_lum = [None] * len(self.lines)
        for i, line in enumerate(self.lines):
            line_lum[i] = load_arr(f'lines/{line_type}/{line}', self.grid_file)

        self.line_luminosities = np.stack(line_lum)


if __name__ == '__main__':
    grid = sps_grid('../grids/output/bc03.h5')
    print("Array shapes (spectra, ages, metallicities, wavelength):\n",
          grid.spectra.shape, grid.ages.shape,
          grid.metallicities.shape, grid.wl.shape)





class SpectralGrid:

    """ This provides an object to hold the SPS / Cloudy grid for use by other parts of the code """


    def __init__(self, grid_name):

        if sythesizer_data_dir:
            grid_filename = f'{sythesizer_data_dir}/grids/{grid_name}.h5'
        else:
            grid_filename = f'{grid_name}.h5'
            grid_name = grid_filename.split('/')[-1]

        hf = h5py.File(grid_filename,'r')

        spectra = hf['spectra']

        self.grid_name = grid_name

        self.lam = spectra['wavelength'][()]
        self.nu = 3E8/(self.lam*1E-10)
        self.log10ages = hf['log10ages'][()]
        self.ages = 10**self.log10ages
        self.metallicities = hf['metallicities'][()]
        self.log10metallicities = hf['log10metallicities'][()]
        self.log10Zs = self.log10metallicities # alias

        self.spectra = {}

        self.spec_names = list(spectra.keys())
        self.spec_names.remove('wavelength')

        for spec_name in self.spec_names:
            # self.spectra[spec_name] = np.swapaxes(hf['spectra/stellar'][()], 0, 1)
            self.spectra[spec_name] = spectra[spec_name][()]

            if spec_name == 'incident':
                self.spectra['stellar'] = self.spectra[spec_name]

        # --- if full cloudy grid available calculate some other spectra for convenience
        if 'linecont' in self.spec_names:
            self.spectra['total'] = self.spectra['transmitted'] + self.spectra['nebular'] # assumes fesc = 0
            self.spectra['nebular_continuum'] = self.spectra['nebular'] - self.spectra['linecont']

        print('available spectra:', list(self.spectra.keys()))


    def get_nearest_index(self, value, array):

        return (np.abs(array - value)).argmin()

    def get_nearest(self, value, array):

        idx = self.get_nearest_index(value, array)

        return idx, array[idx]

    def get_nearest_log10Z(self, log10metallicity):

        return self.get_nearest(log10metallicity, self.log10metallicities)

    def get_nearest_log10age(self, log10age):

        return self.get_nearest(log10age, self.log10ages)


    # def plot_seds(self, ):
    #
    #     """ makes a nice plot of the pure stellar """


# class LineGrid:
