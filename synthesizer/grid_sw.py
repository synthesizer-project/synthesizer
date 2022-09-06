"""
Create Grid object
"""

import numpy as np
import h5py

class Grid:

    def __init__(self, grid_name, path_to_SPS_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'):

        """ This provides an object to hold the SPS / Cloudy grid for use by other parts of the code """

        hf = h5py.File(f'{path_to_SPS_grid}/{grid_name}.h5','r')

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
