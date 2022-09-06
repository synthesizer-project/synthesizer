"""
Create grid in age / metallicity for a given SPS model and IMF
(+ other parameters?)
"""
import numpy as np

from synthesizer.utils import (load_arr, get_names_h5py)


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






# 
# class Binned:
#
#     def get_nearest_index(self, value, array):
#
#         return (np.abs(array - value)).argmin()
#
#     def get_nearest(self, value, array):
#
#         idx = self.get_nearest_index(value, array)
#
#         return idx, array[idx]
#
#     def get_nearest_log10Z(self, log10Z):
#
#         return self.get_nearest(log10Z, self.grid['log10Z'])


# class SPS(SPS_):
#
#     def __init__(self, grid, path_to_SPS_grid = '/data/SPS/nebular/3.0/'):
#
#         self.grid_name = grid.replace('/','-')
#
#         self.grid = pickle.load(open(flare.FLARE_dir + path_to_SPS_grid + grid + '/nebular.p','rb'), encoding='latin1')
#
#         self.lam = self.grid['lam']
#
#
# class synthesizer(SPS_):
#
#     def __init__(self, grid, path_to_SPS_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'):
#
#         hf = h5py.File(f'{path_to_SPS_grid}/{grid}.h5','r')
#
#         self.grid_name = grid
#
#         self.grid = {}
#         self.grid['lam'] = hf['wavelength'][()]
#         self.lam = self.grid['lam']
#         self.nu = 3E8/(self.lam*1E-10)
#         self.grid['log10age'] = hf['log10ages'][()]
#         self.grid['log10Z'] = hf['log10Zs'][()]
#         self.grid['stellar'] = np.swapaxes(hf['spectra/stellar'][()], 0, 1)
#
# class synthesizer_old(SPS_):
#
#     def __init__(self, grid, path_to_SPS_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/old'):
#
#         hf = h5py.File(f'{path_to_SPS_grid}/{grid}.h5','r')
#
#         self.grid_name = grid
#
#         self.grid = {}
#         self.grid['lam'] = hf['wavelength'][()]
#         self.lam = self.grid['lam']
#         self.nu = 3E8/(self.lam*1E-10)
#         self.grid['log10age'] = hf['ages'][()]
#         self.grid['log10Z'] = hf['metallicities'][()]
#         self.grid['stellar'] = np.swapaxes(hf['spectra'][()], 0, 1) * 1.1964952e40 * self.lam/self.nu # erg/s/Hz (1.1964952e40 is a magic number used by Chris)
