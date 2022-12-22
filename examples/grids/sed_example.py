import sys
import numpy as np

from synthesizer.sed import Sed
from synthesizer.grid import SpectralGrid
from synthesizer.filters import SVOFilterCollection


grid_dir = '/cosma7/data/dp004/dc-love2/codes/synthesizer_data/grids/'
# grid_dir = str(sys.argv[1])

model = 'bc03_chabrier03'
grid = SpectralGrid(model, grid_dir=grid_dir)

lam = grid.lam
spec = grid.spectra['stellar'][0,0]
spec_2d = grid.spectra['stellar'][0,:10]

_sed = Sed(lam=lam, lnu=spec)
_sed_2d = Sed(lam=lam, lnu=spec_2d)

print("Beta")
print("1D:", _sed.return_beta())
print("2D:", _sed_2d.return_beta())

print("Beta from spectra")
print("1D:", _sed.return_beta_spec())
print("2D:", _sed_2d.return_beta_spec())

print("Balmer break")
print("1D:", _sed.get_balmer_break())
print("2D:", _sed_2d.get_balmer_break())

print("Broadband luminosities")
fs = [f'JWST/NIRCam.{f}' for f in ['F200W', 'F356W']]
fc = SVOFilterCollection(fs, new_lam=_sed.lam)

print("1D:", _sed.get_broadband_luminosities(fc))
print("2D:", _sed_2d.get_broadband_luminosities(fc))

print("1D:", _sed.get_broadband_fluxes(fc))
print("2D:", _sed_2d.get_broadband_fluxes(fc))
