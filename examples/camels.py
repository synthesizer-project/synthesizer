from synthesizer import grid
from synthesizer.load_data import load_CAMELS_SIMBA

_grid = grid.SpectralGrid(f'../../synthesizer_data/grids/bc03_chabrier03.h5')

gals = load_CAMELS_SIMBA('data/', snap='033')

_g = gals[0]
_spec = _g.integrated_stellar_spectrum(_grid)


import matplotlib.pyplot as plt

plt.loglog(_grid.lam, _spec)
plt.show()

