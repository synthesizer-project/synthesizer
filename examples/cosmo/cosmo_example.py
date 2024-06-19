"""
# Camels example

Use test cosmological simulation data (from the [CAMELS simulations](https://www.camel-simulations.org/)) to generate spectra and calculate photometry.
"""

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.filters import UVJ
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.sed import Sed
from unyt import Myr


grid_dir = "../../../tests/test_grid"
grid_name = "test_grid"
grid = Grid(grid_name, grid_dir=grid_dir)

"""
We then need to load our galaxy data. There are custom data loading script for different simulation codes in `synthesizer.load_data`. For CAMELS-IllutrisTNG there is the `load_CAMELS_IllutrisTNG` method

If your simulation does not have its own front end, please use the templates in `synthesizer/load_data.py` to create your own.
"""

gals = load_CAMELS_IllustrisTNG(
    "../../../tests/data/",
    snap_name="camels_snap.hdf5",
    fof_name="camels_subhalo.hdf5",
)

len(gals)

"""
this creates `gals`, which is a list containing a `synthesizer.Galaxy` object for each structure in the subfind file. These `Galaxy` objects contain lots of useful methods for acting on galaxies, in addition to the component parts of a galaxy. These components include `Stars`, `Gas`, and `BlackHoles`. To generate the intrinsic spectrum of the stellar component we can do the following.
"""

g = gals[1]
spec = g.stars.get_spectra_incident(grid)

"""
Here we grab a single galaxy, and call `stars.get_spectra_incident`, providing our grid object as the first argument. This returns the spectra as an `Sed` object (see the [SED docs](../sed.ipynb)).
"""

spec.lam[:10], spec.lnu[:10]


"""
To access the luminosity and wavelength for `_spec` we can now do
"""

spec.lam[:10], spec.lnu[:10]

"""Notice that these are unyt arrays with associated units. To plot the spectra manually we can do the following.
"""

plt.loglog(spec.lam, spec.lnu)
plt.xlabel("$\\lambda \\,/\\, \\AA$")
plt.ylabel("$L_{\\nu} \\,/\\, \\mathrm{erg \\; s^{-1} \\; Hz^{-1}}$")

"""
However, we can also use the `stars.plot_spectra` method in the stars object (and, indeed, all other components) to plot all of the spectra associated with a galaxy at once.
"""

g.stars.plot_spectra()
plt.show()


"""
Why might you want to create an `Sed` object? This class contains a lot of useful functionality for working with SED's. For example, we can calculate the broadband luminosities.

First, get rest frame 'flux' from the `Sed`.
"""

spec.get_fnu0()

# To get broadband luminosity we first need to define a filter collection object (UVJ default).
fc = UVJ(new_lam=grid.lam)

# And then we can apply it using the `Sed` helper function.

_UVJ = spec.get_photo_fluxes(fc)

print(_UVJ)

"""
## Young and old stellar populations

We restrict the age of star particles used for calculating the spectra. The age is specified by the `young` and `old` parameters; these default to `None`, but if set to a value of age (in units of Myrs) they will filter the star particles above or below this value. If both `young` and `old` are set, the code will raise an error.
"""

young_spec = g.stars.get_spectra_incident(grid, young=100 * Myr)
old_spec = g.stars.get_spectra_incident(grid, old=100 * Myr)

plt.loglog(young_spec.lam, young_spec.lnu, label="young")
plt.loglog(old_spec.lam, old_spec.lnu, label="old")
plt.legend()
plt.xlabel("$\\lambda \\,/\\, \\AA$")
plt.ylabel("$L_{\\nu} \\,/\\, \\mathrm{erg \\; s^{-1} \\; Hz^{-1}}$")

"""
### Replacing young star particles with Parametric Star Formation Histories

For simulations with low mass resolution, the sampling of the star formation history can be affected by Poisson scatter. This is particularly the case for young star particles; a single massive particle that happens to form close to the time of observation can significantly alter the integrated colour of the entire galaxy.

To mitigate this, we provide a method for smoothing the recent star formation history of a particle galaxy by replacing each young star particle with a parametric SFH. An example is provided [here](../sed.ipynb).

This functionality can be enabled by setting the argument `parametric_young_stars` on any `get_spectra_*` methods. This should be set to the age at which you wish to smooth the SFH. The default form of the SFH is constant, but this can also be modified by providing a parametric SFH object to the `parametric_sfh` argument.
"""

parametric_spec = g.stars.get_spectra_incident(
    grid=grid, parametric_young_stars=500 * Myr
)

plt.loglog(young_spec.lam, young_spec.lnu + old_spec.lnu, label="Particle")
plt.loglog(
    parametric_spec.lam, parametric_spec.lnu, label="Parametric + Particle"
)
plt.legend()
plt.xlabel("$\\lambda \\,/\\, \\AA$")
plt.ylabel("$L_{\\nu} \\,/\\, \\mathrm{erg \\; s^{-1} \\; Hz^{-1}}$")

"""
## Nebular emission

If our grid file has been post-processed with CLOUDY we can produce the nebular emission for our camels galaxies. `get_spectra_nebular` produces the pure nebular emission
"""

spec = g.stars.get_spectra_nebular(grid)

fig, ax = g.stars.plot_spectra()
plt.show()

"""
`get_spectra_reprocessed` calculates the gas reprocessed spectra, which we refer to as the *reprocessed* spectra, assuming some escape fraction `fesc`. The combination of the  *reprocessed* and  *escaped* spectra is the *intrinsic* spectra. 
"""

spec = g.stars.get_spectra_reprocessed(grid, fesc=0.1)
fig, ax = g.stars.plot_spectra()
plt.show()

"""
## Dust attenuation

We can apply a range of different dust models to our intrinsic spectra. `get_spectra_screen` applies a simple dust screen to all stellar particles, assuming a V band optical depth $\tau_V$
"""

spec = g.stars.get_spectra_screen(grid, tau_v=0.33)

fig, ax = g.stars.plot_spectra(spectra_to_plot=["intrinsic", "emergent"])
plt.show()

"""
`get_spectra_CharlotFall` applies the [Charlot & Fall+00](https://ui.adsabs.harvard.edu/abs/2000ApJ...539..718C/abstract) two component dust screen model, with V band attenuation for young and old populations defined by the $\tau_V^{BC}$ and $\tau_V^{ISM}$
"""

spec = g.stars.get_spectra_CharlotFall(grid, tau_v_ISM=0.33, tau_v_BC=0.67)

fig, ax = g.stars.plot_spectra(spectra_to_plot=["intrinsic", "emergent"])
plt.show()

"""
### Multiple galaxies 
If we want to create spectra for multiple galaxies we can use a list comprehension. Here we grab the luminosity `lnu` of each galaxy into a list, and combine into a single sed object afterwards using the grid wavelength.
"""

specs = np.vstack([g.stars.get_spectra_incident(grid).lnu for g in gals])
specs = Sed(lam=grid.lam, lnu=specs)

"""
Importantly here, we don't create an SED object for each galaxy spectra. We instead create the 2D array of spectra, and then create an `Sed` object for the whole collection.
"""

fig, ax = plt.subplots(1, 1)
ax.loglog(grid.lam, specs.lnu.T)
ax.set_ylim(
    1e22,
)
ax.set_xlim(1e2, 2e4)
ax.set_xlabel("$\\lambda \\,/\\, \\AA$")
ax.set_ylabel("$L_{\\nu} \\,/\\, \\mathrm{erg \\; s^{-1} \\; Hz^{-1}}$")

"""
### Calculate broadband luminosities

We can then use the `Sed` methods on the whole collection. This is much faster than calling the method for each spectra individually, since we can take advantage of vectorisation. For example, we can calculate UVJ colours of all the selected galaxies in just a couple of lines.

First get rest frame 'flux'
"""

spec.get_fnu0()

# Define a filter collection object (UVJ default) and calculate the photometry.

fc = UVJ(new_lam=grid.lam)

_UVJ = spec.get_photo_fluxes(fc)
print(_UVJ)

_UVJ.plot_photometry(show=True)

# Do for multiple, plot UVJ diagram, coloured by $M_{\star}$

mstar = np.log10(
    np.array([np.sum(g.stars.initial_masses) for g in gals]) * 1e10
)
specs.get_fnu0()
_UVJ = specs.get_photo_fluxes(fc)

UV = _UVJ["U"] / _UVJ["V"]
VJ = _UVJ["V"] / _UVJ["J"]

plt.scatter(VJ, UV, c=mstar, s=40)
plt.xlabel("VJ")
plt.ylabel("UV")
plt.colorbar(
    label=r"$\mathrm{log_{10}} \, M_{\star} \,/\, \mathrm{M_{\odot}}$"
)

"""
## Collective operations

It is often useful to collectively do different operations on a `Galaxy`. Synthesizer enables this via some wrapper methods on a galaxy which will operate on it's components to, for instance, get the observed spectra or photometry for all spectra nested in a `Galaxy`. 
"""

from astropy.cosmology import Planck18 as cosmo

# Get observed spectra for all spectra
g.get_observed_spectra(cosmo=cosmo)

# Get UVJ photometry for all spectra
g.get_photo_luminosities(fc)
g.get_photo_fluxes(fc)

print(
    "Stellar luminosities available:", list(g.stars.photo_luminosities.keys())
)
print("Stellar fluxes available:", list(g.stars.photo_fluxes.keys()))