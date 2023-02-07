import sys
import numpy as np

from synthesizer.sed import Sed
from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection


if __name__ == '__main__':

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid = Grid(grid_name, grid_dir=grid_dir)

    lam = grid.lam
    spec = grid.spectra['stellar'][0, 0]
    spec_2d = grid.spectra['stellar'][0, :10]

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
    fc = FilterCollection(fs, new_lam=_sed.lam)

    print("1D:", _sed.get_broadband_luminosities(fc))
    print("2D:", _sed_2d.get_broadband_luminosities(fc))

    print("1D:", _sed.get_broadband_fluxes(fc))
    print("2D:", _sed_2d.get_broadband_fluxes(fc))
