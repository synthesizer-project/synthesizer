""" A sanity check example for a single star in both parametric and particle
"""
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.particle import Stars as ParticleStars
from synthesizer.parametric import Stars as ParametricStars


# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the parametric stars
sfzh = ParametricStars(
    grid.log10age,
    grid.metallicity,
    instant_sf=1e7,
    instant_metallicity=0.01,
    initial_mass=1,
)

# Compute the parametric sed
sed = stars.get_spectra_transmitted(grid)
plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label="parametric")


part_stars = ParticleStars(
    initial_masses=np.array([1.0]),
    ages=np.array([1e7]),
    metallicities=np.array([0.01]),
)
sed = part_stars.get_spectra_transmitted(grid)
plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label="particle")
plt.legend()
plt.xlim([2, 5])
plt.ylim([18, 22])
plt.show()
