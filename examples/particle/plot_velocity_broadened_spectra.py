"""
Plot velocity broadened spectra
===============================

This example shows how to compute line of sight dust surface densities,
and plots some diagnostics.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from unyt import km, rad, s

from synthesizer.emission_models import NebularEmission
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.sed import plot_spectra

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


# Set the seed
np.random.seed(42)

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the model
model = NebularEmission(grid)

# Create galaxy object
galaxy = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    physical=True,
)[0]

# Invent some fake velocities for the stars (with a bulk moving towards us)
galaxy.stars.velocities = (
    np.random.normal(100, 500, galaxy.stars.coordinates.shape) * km / s
)

# Get the spectra (this will automatically use the tau_vs we just calculated
# since the emission model has tau_v="tau_v")
start_with_shift = time.time()
galaxy.stars.get_spectra(model, vel_shift=True)
print(
    "Time to get spectra with velocity shift: "
    f"{time.time() - start_with_shift}"
)

# Unpack the spectra with the velocity broadening
with_shift = galaxy.stars.spectra["nebular"]

# Clear the spectra
galaxy.clear_all_emissions()

# Get the spectra without the velocity broadening
start_without_shift = time.time()
galaxy.stars.get_spectra(model, vel_shift=False)
print(
    "Time to get spectra without velocity shift: "
    f"{time.time() - start_without_shift}"
)

# Unpack the spectra without the velocity broadening
without_shift = galaxy.stars.spectra["nebular"]

# Plot the two spectra
plot_spectra(
    spectra={"with_shift": with_shift, "without_shift": without_shift},
    show=True,
)

# Exagerate the velocities for the next part
galaxy.stars.velocities *= 10

# Compute the velocity broadened spectra for a range of random rotations
# and plot the difference in the spectra
phis = np.linspace(0, 2 * np.pi, 10) * rad
thetas = np.linspace(0, np.pi, 10) * rad
spectra = {}
for phi, theta in zip(phis, thetas):
    # Rotate the stars
    galaxy.stars.rotate_particles(phi, theta)

    # Get the spectra
    spec = galaxy.stars.get_spectra(model, vel_shift=True)

    # Store it for plotting
    spectra[f"phi={phi:.2f}, theta={theta:.2f}"] = spec

# Plot the spectra
fig, ax = plot_spectra(
    spectra, xlimits=(10**3, 10**4), show=False, figsize=(10, 6)
)

# Modify the legend
ax.legend(ncol=2)

plt.show()
