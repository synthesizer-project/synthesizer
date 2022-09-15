
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import SpectralGrid
from synthesizer.cloudy import read_wavelength, read_continuum



spectra = 'total'

infile = 'test'

spec_dict = read_continuum(infile, return_dict = True)

print(spec_dict.keys())

lam = spec_dict['lam']
lnu = spec_dict[spectra]
lnu /= np.interp(5500., lam, lnu)

plt.plot(lam, np.log10(lnu))

# --- compare with original grids

grid = SpectralGrid('bpass-v2.2.1-bin_chab-100_cloudy-v17.0_logUref-2_OLD')

print(grid.metallicities)
print(len(grid.metallicities))
print(len(grid.ages))




ia = 0
iZ = 8

lam = grid.lam
lnu = grid.spectra[spectra][ia, iZ]
lnu /= np.interp(5500., lam, lnu)

plt.plot(lam, np.log10(lnu), alpha = 0.5)

plt.xlim([0, 10000])
plt.ylim([-1., 2.49])

# plt.figsave('test.pdf')
plt.show()
