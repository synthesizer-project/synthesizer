"""
Example for creating an Sed object straight from the Grid. This example will:
- extract an Sed from a grid point
- demonstrating adding two Sed objects together
"""
import os
import sys
import numpy as np

from synthesizer.sed import Sed
from synthesizer.grid import Grid


# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = script_path + "/../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)
grid_point = (5, 5)

sed1 = grid.get_sed(grid_point)  # get stellar SED at ia = 5, iZ = 5
sed2 = grid.get_sed((3, 5))  # get stellar SED at ia = 5, iZ = 5

# add together two Seds (MAGIC!)
sed = sed1 + sed2

print(sed)

# calculate UV continuum slope


# calcualte Balmer break

# calcualte fluxes


# print("Beta")
# print("1D:", _sed.return_beta())
# print("2D:", _sed_2d.return_beta())
#
# print("Beta from spectra")
# print("1D:", _sed.return_beta_spec())
# print("2D:", _sed_2d.return_beta_spec())
#
# print("Balmer break")
# print("1D:", _sed.get_balmer_break())
# print("2D:", _sed_2d.get_balmer_break())
#
# print("Broadband luminosities")
# fs = [f'JWST/NIRCam.{f}' for f in ['F200W', 'F356W']]
# fc = FilterCollection(fs, new_lam=_sed.lam)
#
# print("1D:", _sed.get_broadband_luminosities(fc))
# print("2D:", _sed_2d.get_broadband_luminosities(fc))
#
# print("1D:", _sed.get_broadband_fluxes(fc))
# print("2D:", _sed_2d.get_broadband_fluxes(fc))
