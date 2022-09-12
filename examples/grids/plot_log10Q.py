

# Create a model SED


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os

from synthesizer.grid import SpectralGrid



if __name__ == '__main__':


    sps_names = ['bpass-v2.3-bin-+00_chab300']

    for sps_name in sps_names:

        grid = SpectralGrid(sps_name)

        fig, ax = grid.plot_log10Q()

        plt.show()

        fig.savefig(f'figs/log10Q_{sps_name}.pdf')
