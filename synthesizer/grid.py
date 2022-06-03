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
