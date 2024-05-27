"""
Demonstrate the dense basis approach for describing the SFZH
============================================================

This script demonstrates how to describe a SFZH in the dense basis formalism.
It includes the following steps:
- Builds a and plots a parametric galaxy using a dense basis representation
- Creates spectra from this object
"""

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = (9.0, -3, 2, 0.3, 0.8)
    redshift = 0.1

    Z_p = {
        "log10metallicity": -2.0
    }  # can also use linear metallicity e.g. {'Z': 0.01}

    stellar_mass = 1e8

    # define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.DenseBasis(db_tuple=sfh_p, redshift=0.1)
    print(sfh)  # print sfh summary

    sfh.plot()

    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS
    # grid.
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    stars.plot_sfzh()

    time = np.linspace(0, 10, 100)
    plt.plot(time, sfh._sfr(10**time))
    plt.show()

    # create a galaxy object
    galaxy = Galaxy(stars)

    # generate pure stellar spectra alone
    galaxy.stars.get_spectra_incident(grid)
    print("Pure stellar spectra")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )
