"""
Create a model SED
"""

import matplotlib.pyplot as plt

from synthesizer.grid import Grid

if __name__ == '__main__':

    sps_names = ['bc03_chabrier03','maraston-rhb_salpeter']

    for sps_name in sps_names:

        grid = Grid(sps_name, verbose = True)

        print(grid.log10Q)

        fig, ax = grid.plot_log10Q()

        plt.show()

        # fig.savefig(f'figs/log10Q_{sps_name}.pdf')
