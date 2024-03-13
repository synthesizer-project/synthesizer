"""
Aperture Mask Example
=====================

Show how to implement fixed spherical apertures
when getting the emission from galaxy objects.
"""

import numpy as np
from unyt import kpc
import matplotlib.pyplot as plt

from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.grid import Grid


grid_dir = "../../tests/test_grid"
grid_name = "test_grid"
grid = Grid(grid_name, grid_dir=grid_dir)

gals = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="camels_snap.hdf5",
    fof_name="camels_subhalo.hdf5",
    fof_dir="../../tests/data/",
)

# Select a single galaxy
gal = gals[1]

# Test calculating the centre manually
print("Galaxy centre from file: gal.centre", gal.centre)

print(
    "Stars centre adopted from parent galaxy: gal.stars.centre",
    gal.stars.centre,
)

gal.stars.calculate_centre_of_mass()

print("Stars centre of mass: gal.stars.centre = ", gal.stars.centre)

print("Galaxy centre unchanged: gal.centre = ", gal.centre)


fig, ax = plt.subplots(1, 1)

for aperture_radius in np.array([30, 10, 5, 2, 1, 0.5]) * kpc:
    spec = gal.stars.get_spectra_incident(grid=grid, aperture=aperture_radius)

    ax.loglog(spec.lam, spec.lnu, label=f"Aperture: {aperture_radius.value}")

ax.set_ylim(1e20, 1e30)
ax.set_xlim(1e2, 2e4)
ax.legend()
ax.set_xlabel("$\\lambda \\,/\\, \\AA$")
ax.set_ylabel("$L_{\\lambda} / \\mathrm{erg / Hz / s}$")

plt.show()
