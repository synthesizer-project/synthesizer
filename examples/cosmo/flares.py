from synthesizer import grid
from synthesizer.load_data import load_FLARES

_grid = grid.Grid(f'../../synthesizer_data/grids/bc03_chabrier03.h5')

region = '00'
tag = '010_z005p000'
_f = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'
gals = load_FLARES(_f, region, tag)

_g = gals[0]
_spec = _g.integrated_stellar_spectrum(_grid)


import matplotlib.pyplot as plt

plt.loglog(_grid.lam, _spec)
plt.show()

