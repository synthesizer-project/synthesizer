"""
Generating Rest Frame SEDs from a Parametric Galaxy
===================================================

Building on the `make_sfzh` documents we can now see how to generate a
galaxy and produce its rest-frame spectral energy distribution.
"""

import matplotlib.pyplot as plt
from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.sed import plot_spectra_as_rainbow
from unyt import Angstrom, Msun, Myr

# We begin by initialising a `Grid`:
grid_name = "test_grid"
grid_dir = "../../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)


# Next we can define the star formation and metal enrichment history:
# Define the functional form of the star formation and
# metal enrichment histories

# Constant star formation
sfh = SFH.Constant(duration=100 * Myr)

# Constant metallicity
metal_dist = ZDist.DeltaConstant(log10metallicity=-2.0)

print(sfh)  # print sfh summary

# Get the 2D star formation and metal enrichment history for
# the given SPS grid. This is (age, Z).
stars = Stars(
    grid.log10age,
    grid.metallicity,
    sf_hist=sfh,
    metal_dist=metal_dist,
    initial_mass=10**8,
)
print(stars)


# Create a `Galaxy` object using this SZFH:


galaxy = Galaxy(stars)


# When combined with a `Grid` we can now generate the spectral energy
# distribution of a galaxy. There are a range of options available to us
# here, most depending on whether we include nebular emission and/or dust.

# Let's star with just the pure stellar spectra. As you can see there is
# also a method on `Galaxy` objects that allows us to quickly plot spectra.


galaxy.stars.get_spectra_incident(grid)
galaxy.stars.plot_spectra()


# However, in most cases we might like to access spectra directly.
# Spectra are stored within each `Galaxy` in the `spectra` dictionary.


sed = galaxy.stars.spectra["incident"]


# Spectra are stored in `Sed` objects. There is a separate tutorial on
# these but the contain the spectra, wavelength grid, and have access to
# a range of other methods, e.g. for caclulating broadband photometry.


print(sed)


plt.loglog(sed.lam, sed.lnu)
plt.show()


fig, ax = plot_spectra_as_rainbow(sed)
plt.show()


# Next, we can generate spectra including nebular emission. In the
# parlance of `synthesizer` these are reprocessed spectra. This
# introduces a new free parameter, the Lyman-continuum escape fraction, `
# fesc`.


galaxy.stars.get_spectra_reprocessed(grid, fesc=0.5)
galaxy.stars.plot_spectra()


# `get_spectra_reprocessed()` actually generates more than just the
# reprocessed spectra, if also generates the `incident`, `transmitted`, `
# nebular`, and `intrinsic` `spectra`. If `fesc>0` it also generates `
# escaped`. The definitions of all of these are described in the spectra
# docs.

# At anytime we can get a list of the spectra associated with a galaxy using:


print(galaxy.stars.spectra.keys())


fig, ax = plot_spectra_as_rainbow(galaxy.stars.spectra["intrinsic"])
plt.show()

fig, ax = plot_spectra_as_rainbow(
    galaxy.stars.spectra["intrinsic"], logged=True
)
plt.show()


# `get_spectra_reprocessed()` also includes a parameter allowing us to
# suppress Lyman-alpha emission, the Lyman-alpha escape fraction `fesc_LyA`.


galaxy.spectra = {}  # reset spectra
galaxy.stars.get_spectra_reprocessed(grid, fesc=0.5, fesc_LyA=0.0)
galaxy.stars.plot_spectra()


# Dust attenuation in `synthesizer` is implemented via the flexible **
# Pacman** model. This model has a few features:
#
# - In this model the parameter `fesc` denotes the fraction of light that
# entirely escapes a galaxy with no reprocessing by gas or dust.
# - Like the `get_spectra_reprocessed()` you can also set the Lyman-alpha
# escape fraction `fesc_LyA` here.
# - It is possible to calculate spectra for both a young and old
# component each with different dust attenuation.
# - Various different dust attenuation (and emission) are provided. By
# default we use a simple power-law.
# - For dust attenuation the required free parameter here is `tau_v` the
# attenuation at 5500A. If an array is provided.

# First, let's add dust attenuation using a simple screen model with a V-
# band optical depth `tau_v=0.5` and a power-law attenuation curve with `
# alpha=-1`.


galaxy.spectra = {}  # reset spectra
galaxy.stars.get_spectra_pacman(grid, tau_v=0.5, alpha=-1)
galaxy.stars.plot_spectra(
    spectra_to_plot=["intrinsic", "attenuated", "emergent"]
)


# Next, let's allow `fesc` to vary. In the pacman model the fraction of
# light escaping reprocessing by gas also escape dust attenuation.


galaxy.spectra = {}  # reset spectra
galaxy.stars.get_spectra_pacman(grid, tau_v=0.5, alpha=-1, fesc=0.5)
galaxy.stars.plot_spectra(
    spectra_to_plot=["intrinsic", "attenuated", "emergent"]
)


# Note, that despite the same `tau_v` the actual attenuation is much less.
# Fortunately if we want to know the true attenuation there is a method `A()`
# on Galaxy for that which take the wavelength.


# FIX COMING SOON
# galaxy.A(5500*Angstrom)


# `get_spectra_pacman()` can also implement dust attenuation separately
# for both young and old components (where the threshold is set by `young_
# old_thresh` which is log10(threshold/yr)). In this case it is also
# necessary to provide `tau_v` and `alpha` as pairs of values describing
# the ISM and birth-cloud components. Note, young stellar populations
# feel attenuation from both the ISM and birth-cloud components.


galaxy.spectra = {}  # reset spectra
tau_v_ISM = 0.5  # ISM component of dust attenuation
tau_v_BC = 0.5  # birth-cloud componest of dust attenuation
tau_v = [tau_v_ISM, tau_v_BC]
alpha = [-0.7, -1.3]
galaxy.stars.get_spectra_pacman(
    grid, tau_v=tau_v, alpha=alpha, young_old_thresh=10 * Myr
)
galaxy.stars.plot_spectra(
    spectra_to_plot=["emergent", "young_emergent", "old_emergent"]
)


# For users more familiar with the Charlot and Fall (2000) two component
# dust model `synthesizer` also includes a `get_spectra_CharlotFall()`
# method, which is really a wrapper around the more generic `get_spectra_
# pacman()` method. The difference is that `fesc` is implicitly assumed
# to `0.0` and there is a more familiar way of setting the parameters.


galaxy.stars.get_spectra_CharlotFall(
    grid, tau_v_ISM=0.5, tau_v_BC=0.5, alpha_ISM=-0.7, alpha_BC=-1.3
)
galaxy.stars.plot_spectra(
    spectra_to_plot=["emergent", "young_emergent", "old_emergent"],
    quantity_to_plot="luminosity",
)


# Here we also demonstrate that the luminosity can be plotted instead of
# the spectral luminosity density by passing `quantity_to_plot="luminosity
# "` to the `plot_spectra` method. In fact, any quantity stored on an `Sed
# ` can be passed to this argument to plot the respective quantity. These
# options include "lnu", "luminosity" or "llam" for rest frame spectra or
# "fnu", "flam" or "flux" for observed spectra.
#
# ### Dust emission
#
# `synthesizer` can also be used to model emission through a simple
# energy balance approach. To do this we can supply a method that
# calculates an attenuated spectrum a `synthesizer.dust.emission.
# DustEmission` object. `synthesizer` has several built-in and these are
# described in **insert referenc**.


from synthesizer.dust.emission import Greybody, IR_templates
from unyt import K

# If we provide a single attenuation (and curve) we need to only provide
# a single dust_emission model:


# initialise a greybody dust emission model
dust_emission_model = Greybody(30 * K, 1.2)

galaxy.spectra = {}  # reset spectra
galaxy.stars.get_spectra_pacman(
    grid, tau_v=0.5, alpha=-1, dust_emission_model=dust_emission_model
)
galaxy.stars.plot_spectra(spectra_to_plot=["emergent", "dust", "total"])

print(
    "Dust luminosity =",
    galaxy.stars.spectra["total"].measure_window_luminosity(
        window=[1e4 * Angstrom, 1e7 * Angstrom]
    ),
)


# We can also specificy different dust emission models for the birth
# cloud and ISM (diffuse) dust separately:


galaxy.spectra = {}  # reset spectra
tau_v_ISM = 0.5  # ISM component of dust attenuation
tau_v_BC = 0.5  # birth-cloud componest of dust attenuation
tau_v = [tau_v_ISM, tau_v_BC]
alpha = [-0.7, -1.3]
dust_emission_ISM = Greybody(20 * K, 1.2)
dust_emission_BC = Greybody(50 * K, 1.2)
dust_emission_model = [dust_emission_ISM, dust_emission_BC]

galaxy.stars.get_spectra_pacman(
    grid,
    tau_v=tau_v,
    alpha=alpha,
    young_old_thresh=10 * Myr,
    dust_emission_model=dust_emission_model,
)


galaxy.stars.plot_spectra(
    spectra_to_plot=[
        "old_dust",
        "young_dust_BC",
        "young_dust_ISM",
        "young_dust",
        "dust",
    ]
)
galaxy.stars.plot_spectra(
    spectra_to_plot=["old_total", "young_total", "total"]
)
plt.xlim(1e4, 5e7)

print(
    "Dust luminosity =",
    galaxy.stars.spectra["total"].measure_window_luminosity(
        window=[1e4 * Angstrom, 1e7 * Angstrom]
    ),
)


# #### We can instead specify IR template spectra as well


galaxy.spectra = {}  # reset spectra
grid_name_ir = "MW3.1"
grid_dir_ir = "../../../tests/test_grid/"
grid_ir = Grid(
    grid_name_ir, grid_dir=grid_dir_ir, read_spectra=True, read_lines=False
)


tau_v_ISM = 0.5  # ISM component of dust attenuation
tau_v_BC = 0.5  # birth-cloud componest of dust attenuation
tau_v = [tau_v_ISM, tau_v_BC]
alpha = [-0.7, -1.3]
mdust = 5e9 * Msun
dust_emission_model = IR_templates(grid_ir, mdust=mdust, gamma=0.05)

galaxy.stars.get_spectra_pacman(
    grid,
    tau_v=tau_v,
    alpha=alpha,
    young_old_thresh=1e7 * Myr,
    dust_emission_model=dust_emission_model,
)


galaxy.stars.plot_spectra(spectra_to_plot=["old_dust", "young_dust", "dust"])
galaxy.stars.plot_spectra(
    spectra_to_plot=["old_total", "young_total", "total"]
)
plt.xlim(1e4, 5e7)

print(
    "Dust luminosity =",
    galaxy.stars.spectra["total"].measure_window_luminosity(
        window=[1e4 * Angstrom, 1e7 * Angstrom]
    ),
)


# ### Galaxy summary

# Like other objects in `synthesizer` we can also get a useful summary of
# the `Galaxy` object just using the `print` function:


print(galaxy)


# We can also extract an spectra and generate broadband photometry. See
# the `Sed` and `Filter` tutorials:


sed = galaxy.stars.spectra["emergent"]

tophats = {
    "U": {"lam_eff": 3650, "lam_fwhm": 660},
    "V": {"lam_eff": 5510, "lam_fwhm": 880},
    "J": {"lam_eff": 12200, "lam_fwhm": 2130},
}
fc = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)
bb_lnu = sed.get_photo_luminosities(fc)

print(bb_lnu)

# Plot the photometry
fig, ax = bb_lnu.plot_photometry(show=True)
