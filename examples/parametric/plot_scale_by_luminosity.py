"""
An example showing how to scale a galaxy's mass by luminosity/flux.
===================================================================

Parametric galaxies scale their brightness based on their initial stellar
mass which is passed at instantiation. This example shows how a galaxy
can later be scaled in terms of mass to achieve a particular brightness in
a particular filter.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from matplotlib.lines import Line2D
from unyt import Hz, Msun, Myr, erg, nJy, s

from synthesizer import galaxy
from synthesizer.conversions import apparent_mag_to_fnu, fnu_to_lnu
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.filters import Filter
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist

# Set up a figure to plot on
fig = plt.figure()
ax_lum = fig.add_subplot(111)

# Set up the flux y axis
ax_flux = ax_lum.twinx()

# Enforce logscale
ax_lum.loglog()
ax_flux.loglog()

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the emission model
model = PacmanEmission(
    grid,
    fesc=0.5,
    fesc_ly_alpha=0.5,
    tau_v=0.1,
    dust_curve=PowerLaw(slope=-1),
)

# Set up the SFH
sfh = SFH.Constant(max_age=100 * Myr)

# Set up the metallicity distribution
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

# Get the stellar population
stars = Stars(
    grid.log10age,
    grid.metallicity,
    sf_hist=sfh,
    metal_dist=metal_dist,
    initial_mass=10**9 * Msun,
)

# And make a galaxy containing this stellar population
redshift = 12
gal = galaxy(stars=stars, redshift=redshift)

# Generate spectra using pacman model (complex)
gal.stars.get_spectra(model)

# Get the observed spectra
gal.get_observed_spectra(cosmo=cosmo)

# Plot the original spectra
ax_lum.plot(
    gal.stars.spectra["attenuated"]._lam,
    gal.stars.spectra["attenuated"]._lnu,
    color="m",
    linestyle="--",
    alpha=0.7,
)
ax_flux.plot(
    gal.stars.spectra["attenuated"]._obslam,
    gal.stars.spectra["attenuated"]._fnu,
    color="m",
    label=(
        r"Original ($\log_{10}(M_\star / M_\odot)"
        + f"={np.log10(gal.stars.initial_mass):.2f})$"
    ),
    alpha=0.7,
)

# Define a filter in which we will scale the galaxy
f = Filter(filter_code="JWST/NIRCam.F150W", new_lam=grid.lam)

# First lets scale by luminosity
scale_lum = 10**25.0 * erg / s / Hz
gal.stars.scale_mass_by_luminosity(
    lum=scale_lum,
    scale_filter=f,
    spectra_type="attenuated",
)

# Plot the luminosity scaled spectra
ax_lum.plot(
    gal.stars.spectra["attenuated"]._lam,
    gal.stars.spectra["attenuated"]._lnu,
    color="c",
    linestyle="--",
    alpha=0.7,
)
ax_flux.plot(
    gal.stars.spectra["attenuated"]._obslam,
    gal.stars.spectra["attenuated"]._fnu,
    color="c",
    label=(
        r"Luminosity scaled ($\log_{10}(M_\star / M_\odot)"
        + f"={np.log10(gal.stars.initial_mass):.2f})$"
    ),
    alpha=0.7,
)

# And scale by flux
scale_flux = apparent_mag_to_fnu(20)
print(scale_flux)
gal.stars.scale_mass_by_flux(
    flux=scale_flux,
    scale_filter=f,
    spectra_type="attenuated",
)

# Plot the luminosity scaled spectra
ax_lum.plot(
    gal.stars.spectra["attenuated"]._lam,
    gal.stars.spectra["attenuated"]._lnu,
    color="orange",
    linestyle="--",
    alpha=0.7,
)
ax_flux.plot(
    gal.stars.spectra["attenuated"]._obslam,
    gal.stars.spectra["attenuated"]._fnu,
    color="orange",
    label=(
        r"Flux scaled ($\log_{10}(M_\star / M_\odot)"
        + f"={np.log10(gal.stars.initial_mass):.2f})$"
    ),
    alpha=0.7,
)

# Label axes
x_units = str(gal.stars.spectra["attenuated"].lam.units)
y_units_lnu = str(gal.stars.spectra["attenuated"].lnu.units)
y_units_fnu = str(gal.stars.spectra["attenuated"].fnu.units)
ax_lum.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
ax_lum.set_ylabel(r"$L_{\nu}/[\mathrm{" + y_units_lnu + r"}]$")
ax_flux.set_ylabel(r"$F_{\nu}/[\mathrm{" + y_units_fnu + r"}]$")


# Make the legend
legend_handles = [
    Line2D([0], [0], color="k", linestyle="--"),
    Line2D([0], [0], color="k", linestyle="-"),
]
legend_labels = ["Luminosity", "Flux"]
ax_flux.legend(loc="upper right")
ax_lum.legend(handles=legend_handles, labels=legend_labels, loc="upper left")

ax_lum.set_ylim(
    fnu_to_lnu(10**-4 * nJy, cosmo, redshift=redshift),
    fnu_to_lnu(10**12.5 * nJy, cosmo, redshift=redshift),
)
ax_flux.set_ylim(10**-4, 10**12.5)
ax_lum.set_xlim(10**2, None)
ax_flux.set_xlim(10**2, None)
plt.show()
