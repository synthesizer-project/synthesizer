"""
Compare parametric and particle SEDs
=====================================

This example compares a sampled and binned (parametric) SED for different numbers of particles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfhz
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy


# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the SFH and metallicity distribution
Z_p = {"metallicity": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(**sfh_p)

# Define the parametric stars
sfzh = ParametricStars(
    grid.log10age, grid.metallicity, sf_hist=sfh, metal_dist=metal_dist
    initial_mass=1,
)

# Compute the parametric sed
parametric_galaxy = ParametricGalaxy(sfzh)
parametric_galaxy.stars.get_spectra_incident(grid)
sed = parametric_galaxy.stars.spectra["incident"]
plt.plot(
    np.log10(sed.lam), np.log10(sed.lnu), label="parametric", lw=4, c="k", alpha=0.3
)


# Compute the particle Sed for a range of particle samples
for nstar in [1, 10, 100, 1000]:
    # Get the stars object
    stars = sample_sfhz(
        sfzh.sfzh,
        sfzh.log10ages,
        sfzh.log10metallicities,
        nstar,
        initial_mass=1 / nstar,
    )

    # Get the particle galaxy
    particle_galaxy = ParticleGalaxy(stars=stars)

    # Calculate the stars SEDs using nearest grid point
    ngp_sed = particle_galaxy.stars.get_spectra_incident(
        grid, grid_assignment_method="ngp"
    )

    plt.plot(
        np.log10(ngp_sed.lam),
        np.log10(ngp_sed.lnu),
        label=f"particle (N={nstar})",
    )


plt.legend()
plt.xlim([2, 5])
plt.ylim([18, 22])
plt.show()
