"""
Plot delta_lambda for a grid.
===============================

This script demonstrates how to generate delta_lambda from a provided grid.
It includes the following steps:
- Builds a parametric galaxy using make_sfzh.
- Retrieves delta_lambda for the galaxy using the grid.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the emission model
    model = IncidentEmission(grid)

    # define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"max_age": 10 * Myr}
    Z_p = {
        "log10metallicity": -2.0
    }  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8 * Msun

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
    galaxy.stars.get_spectra(model)
    lam, delta_lam = Grid.get_delta_lambda(grid)
    print("Mean delta: ", np.mean(delta_lam))

    figsize = (10, 5)

    fig = plt.figure(figsize=figsize)

    left = 0.15
    height = 0.6
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    ypeak = np.min(delta_lam)
    ylimits = np.mean(delta_lam)

    ax.plot(np.log10(lam)[:-1], delta_lam, lw=1, alpha=0.8, label=grid_name)

    xlim = [2.6, 4.2]
    ylim = [ypeak - ylimits, ypeak + ylimits]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    ax.set_ylabel(r"$\rm Δ(\lambda/\AA)$")

    plt.show()
