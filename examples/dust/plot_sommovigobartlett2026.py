"""
Sommovigo & Bartlett 2026 attenuation
=====================================

Predict a SommovigoBartlett2026 attenuation curve from galaxy properties and
apply it through the standard ``get_spectra`` workflow.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Angstrom, Msun, degree, kpc, yr

from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import SommovigoBartlett2026
from synthesizer.grid import Grid
from synthesizer.particle import Galaxy, Gas, Stars

# SB26 is calibrated over the UV-optical range.
grid = Grid(
    "test_grid",
    lam_lims=(1000 * Angstrom, 10000 * Angstrom),
    ignore_lines=True,
)

# SFR100, stellar half-mass radius, stellar mass, and gas metallicity are
# inferred from this toy particle galaxy.
stars = Stars(
    initial_masses=np.array([0.3, 0.2, 4.0, 6.0]) * 1e9 * Msun,
    current_masses=np.array([0.28, 0.18, 3.5, 5.0]) * 1e9 * Msun,
    ages=np.array([10.0, 50.0, 200.0, 500.0]) * 1e6 * yr,
    metallicities=np.full(4, 0.02),
    coordinates=np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-8.0, 0.0, 0.0],
            [0.0, -12.0, 0.0],
        ]
    )
    * kpc,
    centre=np.zeros(3) * kpc,
)
gas = Gas(
    masses=np.array([3.0, 2.0, 1.0]) * 1e9 * Msun,
    metallicities=np.array([0.018, 0.022, 0.015]),
    dust_to_metal_ratio=0.3,
)
galaxy = Galaxy(stars=stars, gas=gas, centre=np.zeros(3) * kpc)

# Omitting inclination draws a random isotropic viewing angle. Every prediction
# input may instead be a direct value or a galaxy attribute name, for example
# inclination="observed_inclination". Supplying all five inputs also supports a
# parametric galaxy without gas because z_gas need not be inferred.
galaxy.get_dust_curve_params_sommovigobartlett2026(
    inclination=60 * degree,
    dust_model="MW",
)

# The no-argument curve reads tau_v and all four B parameters from galaxy.stars
# through the standard get_params machinery.
model = PacmanEmission(
    grid,
    dust_curve=SommovigoBartlett2026(),
    fesc=0.0,
    fesc_ly_alpha=1.0,
)
galaxy.stars.get_spectra(model)

fig, ax = plt.subplots(figsize=(8, 6))
for label in ("reprocessed", "attenuated"):
    sed = galaxy.stars.spectra[label]
    ax.loglog(sed.lam, sed.lnu, label=label)
ax.set_xlabel(r"$\lambda/(\AA)$")
ax.set_ylabel(r"$L_\nu/(\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{Hz}^{-1})$")
ax.set_title("Sommovigo & Bartlett (2026) attenuation")
ax.legend()
plt.tight_layout()
plt.show()
