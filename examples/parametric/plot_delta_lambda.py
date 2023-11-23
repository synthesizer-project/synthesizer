"""
Plot delta_lambda for a grid.
===============================

This script demonstrates how to generate delta_lambda from a provided grid. 
It includes the following steps:
- Builds a parametric galaxy using make_sfzh.
- Retrieves delta_lambda for the galaxy using the grid.

"""

import numpy as np
import matplotlib.pyplot as plt

import synthesizer

import importlib

package_name = "synthesizer"

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist, Stars
from synthesizer.parametric.galaxy import Galaxy
from unyt import Myr

if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want the directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

       # define filters
    filter_codes = [
        f"JWST/NIRCam.{f}"
        for f in ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
    ]  # define a list of filter codes
    filter_codes += [f"JWST/MIRI.{f}" for f in ["F770W"]]
    fc = FilterCollection(filter_codes, new_lam=grid.lam)

    # define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"duration": 10 * Myr}
    Z_p = {"log10metallicity": -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8

    # define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(**sfh_p)  # constant star formation
    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS
    # grid. This is (age, Z).
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    # Define redshift
    z = 10.0

    # create a galaxy object
    galaxy = Galaxy(stars, redshift=z)
    
    # Delta lambda model for pure stellar spectra
    galaxy.stars.get_spectra_incident(grid)
    lam, delta_lam = Grid.get_delta_lambda(grid)
    print("Mean delta: ", np.mean(delta_lam))
    
    ylimits = ("peak", np.mean(delta_lam))
    figsize = (10, 5)
    
    fig = plt.figure(figsize=figsize)

    left = 0.15
    height = 0.6
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    xlim = [2.6, 4.2]

    ypeak = -100

    ax.plot(np.log10(lam)[:-1], delta_lam, lw=1, alpha=0.8, label=grid_name)

    if np.max(delta_lam) > ypeak:
        ypeak = np.min(delta_lam)

    ax.set_xlim(xlim)

    if ylimits[0] == "peak":
        if ypeak == ypeak:
            ylim = [ypeak - ylimits[1], ypeak + ylimits[1]]
        ax.set_ylim(ylim)

    ax.set_xlim(xlim)

    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    ax.set_ylabel(r"$\rm Δ(\lambda/\AA)$")

    plt.show()

    exit()
