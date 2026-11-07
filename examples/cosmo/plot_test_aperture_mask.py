"""
Aperture Mask Example
=====================

Show how to implement fixed spherical apertures
when getting the emission from galaxy objects.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import kpc

from synthesizer import TEST_DATA_DIR
from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG

grid_name = "test_grid"
grid = Grid(grid_name)

gals = load_CAMELS_IllustrisTNG(
    TEST_DATA_DIR,
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    group_dir=TEST_DATA_DIR,
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

# Compute the radii of the star particles from this centre, these are what
# the aperture masks below select on
gal.stars.get_radii()

# Define the emission model, masked to each aperture in turn below
model = IncidentEmission(grid)

fig, ax = plt.subplots(1, 1)

for aperture_radius in np.array([30, 10, 5, 2, 1, 0.5]) * kpc:
    # Only include stars within the aperture
    model.clear_masks()
    model.add_mask(attr="radii", op="<", thresh=aperture_radius)

    spec = gal.stars.get_spectra(model)

    ax.loglog(spec.lam, spec.lnu, label=f"Aperture: {aperture_radius.value}")

ax.set_ylim(1e20, 1e30)
ax.set_xlim(1e2, 2e4)
ax.legend()
ax.set_xlabel("$\\lambda \\,/\\, \\AA$")
ax.set_ylabel("$L_{\\lambda} / \\mathrm{erg / Hz / s}$")

plt.show()
