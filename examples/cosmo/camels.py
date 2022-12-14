import numpy as np
import matplotlib.pyplot as plt

from synthesizer import grid
from synthesizer.load_data import load_CAMELS_SIMBA

_grid = grid.SpectralGrid('bc03_chabrier03', grid_dir=f'../../synthesizer_data/grids/')

gals = load_CAMELS_SIMBA('data/', snap='033')

_g = gals[0]
_spec = _g.integrated_stellar_spectrum(_grid)

plt.loglog(_grid.lam, _spec)
plt.show()

_specs = np.vstack([_g.integrated_stellar_spectrum(_grid) for _g in gals[:10]])

plt.loglog(_grid.lam, _specs.T)
plt.show()

