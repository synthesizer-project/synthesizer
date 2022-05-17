"""
Front end for handling data and parameter files, setting up runs
"""

import argparse

import numpy as np

from synthesizer import grid

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="parameter file", type=str)
args = parser.parse_args()

params = __import__(args.param_file)

print(params.sps_grid)


# ---- load SPS grid
grid = grid.sps_grid(params.sps_grid)

