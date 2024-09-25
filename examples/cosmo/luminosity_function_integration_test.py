"""
Camels luminosity function example
==================================

Use test cosmological simulation data (from the [CAMELS simulations](
https://www.camel-simulations.org/)) to generate SDSS photometry and
plot the luminosity functions.

Note: this is an integration test, and requires the full camels
snapshot and subhalo catalogue. These can be downloaded from the
tests/data/ folder using the `download_camels.sh` script (an internet
connection is required).
"""

import matplotlib.pyplot as plt
import numpy as np

from synthesizer.conversions import lnu_to_absolute_mag
from synthesizer.emission_models import IncidentEmission
from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG


def calc_df(x, volume, binLimits):
    hist, dummy = np.histogram(x, bins=binLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (binLimits[1] - binLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) / (
        binLimits[1] - binLimits[0]
    )  # Poisson errors

    return phi, phi_sigma, hist


h = 0.6711
grid_dir = "../../tests/test_grid"
grid_name = "test_grid"
grid = Grid(grid_name, grid_dir=grid_dir)
incident = IncidentEmission(grid)

filter_codes = [f"SLOAN/SDSS.{f}" for f in ["g"]]
fc = FilterCollection(filter_codes=filter_codes, new_lam=grid.lam)

gals = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="CV_0_snap_086.hdf5",
    group_name="CV_0_fof_subhalo_tab_086.hdf5",
)

# Filter by stellar mass
mstar_mask = np.array([np.sum(g.stars.masses) > 1e9 for g in gals])
gals = [g for g, _m in zip(gals, mstar_mask) if _m]

# Calculate g-band magnitudes
specs = np.vstack([g.stars.get_spectra(incident).lnu for g in gals])
phot = np.vstack(
    [g.stars.get_photo_lnu(fc)["incident"]["SLOAN/SDSS.g"] for g in gals]
)
mags = lnu_to_absolute_mag(phot)

# Calculate g-band luminosity function
binlims = np.linspace(-25, -16, 12)
bins = binlims[:-1] + (binlims[1] - binlims[0]) / 2
phi, phi_sigma, _ = calc_df(mags, (25 / h) ** 3, binlims)

# Plot luminosity function
fig, ax = plt.subplots(1, 1)

ax.plot(bins, np.log10(phi), label="Current version")

# Previous 'working' version LF for reference
# Commit hash:
# b683b81e5de1e8c6ab6938d047cab54e7c5a2fdf
phi_previous = [
    -np.inf,
    -np.inf,
    -4.63,
    -3.63,
    -3.23,
    -2.66,
    -2.59,
    -2.85,
    -2.73,
    -2.89,
    -3.33,
]

ax.plot(bins, phi_previous, label="Previous version", ls="dashed")

ax.legend()
ax.set_xlabel("$m_{g}^{AB}$")
ax.set_ylabel(r"$\phi \,/\, \mathrm{Mpc^{-3} \; dex^{-1}}$")

plt.show()
# plt.savefig('test_lf.png'); plt.close()
