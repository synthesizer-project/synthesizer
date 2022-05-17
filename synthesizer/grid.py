"""
Create grid in age / metallicity for a given SPS model and IMF (+ other parameters?)
"""

from synthesizer.utils import load_arr


class sps_grid:

    def __init__(self, grid_file, verbose=False):

        if verbose: print("Loading model from: \n\n%s\n"%(grid_file))

        self.spectra = load_arr('spectra', grid_file)
        self.ages = load_arr('ages', grid_file)
        self.metallicities = load_arr('metallicities', grid_file)
        self.wl = load_arr('wavelength', grid_file)

        if self.ages[0] > self.ages[1]:
             if verbose: print("Age array not sorted ascendingly. Sorting...\n")
             self.ages = self.ages[::-1]
             self.spectra = self.spectra[:,::-1,:]


        if self.metallicities[0] > self.metallicities[1]:
            if verbose: print("Metallicity array not sorted ascendingly. Sorting...\n")
            self.metallicities = self.metallicities[::-1]
            self.spectra = self.spectra[::-1,:,:]



if __name__ == '__main__':
    grid = sps_grid('../grids/output/bc03.h5')
    print("Array shapes (spectra, ages, metallicities, wavelength):\n", 
          grid.spectra.shape, grid.ages.shape, grid.metallicities.shape, grid.wl.shape)


