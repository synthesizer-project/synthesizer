

# Create a model SED


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os

from synthesizer.grid import LineGrid






if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    model = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
    target_Z = 0.01 # target metallicity

    line_id = 'HI6563'
    line_id = ('HI6563')
    line_id = ('HI4861', 'OIII4959', 'OIII5007')

    grid = LineGrid(model, verbose = True)

    iZ = grid.get_nearest_index(target_Z, grid.metallicities)

    print(iZ)


    for ia, log10age in enumerate(grid.log10ages):

        print(log10age, grid.get_line_info(line_id, ia, iZ))
