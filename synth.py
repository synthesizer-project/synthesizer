"""
Front end for handling data and parameter files, setting up runs
"""

import argparse

from synthesizer import sps_grid


parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

params = __import__(args.param_file)

print(params.sps_grid)

grid = sps_grid.sps_grid(params.sps_grid)
print("Array shapes (spec, ages, metallicity, wavelength):\n", grid.spec.shape, grid.ages.shape, grid.metallicity.shape, grid.wl.shape)



if params.dust:
    print("do some dust stuff here")


if params.nebular:
    print("do some nebular stuff here")


