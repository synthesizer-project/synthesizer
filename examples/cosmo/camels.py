import numpy as np
import matplotlib.pyplot as plt

from synthesizer import grid
from synthesizer.load_data import load_CAMELS_SIMBA

## first load a spectral grid
_grid = grid.SpectralGrid('bc03_chabrier03', grid_dir=f'/an/example/grid/synthesizer_data/grids/')

# now load some example CAMELS data using the dedicated data loader
gals = load_CAMELS_SIMBA('data/', snap='033')

# calculate the spectra for a single galaxy
_g = gals[0]
_spec = _g.integrated_stellar_spectrum(_grid)

plt.loglog(_grid.lam, _spec.lnu)
plt.show()

# multiple galaxies
_specs = np.vstack([_g.integrated_stellar_spectrum(_grid).lnu for _g in gals[:10]])

plt.loglog(_grid.lam, _specs.T)
plt.show()

# calculate broadband luminosities
from synthesizer.filters import UVJ

# first get rest frame 'flux'
_spec.get_fnu0()

# define a filter collection object (UVJ default)
fc = UVJ(new_lam=_grid.lam)

_UVJ = _spec.get_broadband_fluxes(fc)
print(_UVJ)

# do for multiple, plot UVJ diagram
_specs = [_g.integrated_stellar_spectrum(_grid) \
                for _g in gals]

[_s.get_fnu0() for _s in _specs]

_UVJ = [_s.get_broadband_fluxes(fc) for _s in _specs]

UV = [(_uvj['U'] / _uvj['V']).value for _uvj in _UVJ]
VJ = [(_uvj['V'] / _uvj['J']).value for _uvj in _UVJ]

plt.scatter(VJ, UV)
plt.xlabel('VJ')
plt.ylabel('UV')
plt.show()


