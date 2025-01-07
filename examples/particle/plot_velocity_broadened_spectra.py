"""
Plot velocity broadened spectra
===============================

This example shows how to compute line of sight dust surface densities,
and plots some diagnostics.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from unyt import km, s

from synthesizer.emission_models import NebularEmission
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.sed import plot_spectra

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def calculate_smoothing_lengths(positions, num_neighbors=56):
    """Calculate the SPH smoothing lengths for a set of coordinates."""
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=num_neighbors + 1)

    # The k-th nearest neighbor distance (k = num_neighbors)
    kth_distances = distances[:, num_neighbors]

    # Set the smoothing length to the k-th nearest neighbor
    # distance divided by 2.0
    smoothing_lengths = kth_distances / 2.0

    return smoothing_lengths


# Set the seed
np.random.seed(42)

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the model
model = NebularEmission(
    grid,
    per_particle=True,
)

# Create galaxy object
galaxy = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    physical=True,
)[0]

# Invent some fake velocities for the stars (with a bulk moving towards us)
galaxy.stars.velocities = (
    np.random.normal(100, 200, galaxy.stars.coordinates.shape) * km / s
)

# Get the spectra (this will automatically use the tau_vs we just calculated
# since the emission model has tau_v="tau_v")
start_with_shift = time.time()
galaxy.stars.get_spectra(model, vel_shift=True)
print(
    "Time to get spectra with velocity shift: "
    f"{time.time() - start_with_shift}"
)

# Plot the Sed
galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

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

# Plot the difference between the two spectra
plt.figure()
plt.loglog(
    with_shift._lam, (with_shift.lnu - without_shift.lnu) / with_shift.lnu
)

plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Difference in Luminosity")
plt.title("Difference in Luminosity with and without Velocity Broadening")

plt.show()
