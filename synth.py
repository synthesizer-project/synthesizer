"""
Front end for handling data and parameter files, setting up runs
"""

import argparse

import numpy as np

from synthesizer import grid

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

# Will: Could we bury this in a function in utils called "read_params" or similar
# would aide readability but could also let us hide other manipulations we had to do
# an example being if we have multiple levels of verbosity on different ranks 
# comes to mind but might be other stuff in the long run.
params = __import__(args.param_file)

print(params.sps_grid)


# ---- load SPS grid
grid = grid.sps_grid(params.sps_grid)

