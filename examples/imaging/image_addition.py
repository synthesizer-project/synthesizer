"""
Image addition example
======================

An example of how to do image addition and testing error functionality.
"""

import os
import numpy as np
from unyt import kpc

from synthesizer.grid import Grid
from synthesizer.exceptions import InconsistentAddition
from synthesizer.filters import FilterCollection as Filters
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.imaging.images import ParametricImage, ParticleImage
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.parametric import Stars

# First set up some stuff so we can make demonstration images.

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want the directory
# script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define an arbitrary morphology to feed the imaging
morph = Sersic2D(r_eff_kpc=1.0 * kpc, n=1.0, ellip=0.5, theta=35.0)

# Define the parameters of the star formation and metal enrichment histories
stellar_mass = 1e10
stars = Stars(
    grid.log10age,
    grid.metallicity,
    sf_dist=10.0,
    metal_dist=0.01,
    initial_mass=stellar_mass,
    morphology=morph,
)

# Get galaxy object
galaxy = Galaxy(stars=stars)

# Get specrtra
sed = galaxy.stars.get_spectra_incident(grid)

# Create a filter collection
filter_codes1 = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W", "JWST/NIRCam.F200W"]
filter_codes2 = filter_codes1[:-1]
filters1 = Filters(filter_codes1)
filters2 = Filters(filter_codes2)

fake_img = np.zeros((100, 100))

# Create fake image objects
img1 = ParticleImage(
    resolution=0.5 * kpc,
    npix=100,
    fov=None,
    filters=(),
    positions=np.zeros((100, 3)) * kpc,
    pixel_values=np.ones(100),
)
img2 = ParticleImage(
    resolution=0.4 * kpc,
    npix=100,
    fov=None,
    filters=(),
    positions=np.zeros((100, 3)) * kpc,
    pixel_values=np.ones(100),
)
img_with_filters1 = ParametricImage(
    morphology=morph,
    sed=sed,
    resolution=0.5 * kpc,
    npix=100,
    fov=None,
    filters=filters1,
)
img_with_filters2 = ParametricImage(
    morphology=morph,
    sed=sed,
    resolution=0.5 * kpc,
    npix=100,
    fov=None,
    filters=filters2,
)
img1.img = fake_img
img2.img = fake_img
for f in filter_codes1:
    img_with_filters1.imgs[f] = fake_img
for f in filter_codes2:
    img_with_filters2.imgs[f] = fake_img

# Add images
composite = img1 + img1 + img1
composite_with_filters = img_with_filters1 + img_with_filters1
composite_mixed1 = img1 + img_with_filters1
composite_mixed2 = img_with_filters1 + img1

# Added images preserve a history of what objects were added
print("Example of image nesting from img1 + img2 + img3:", composite.combined_imgs)

print("Error Demonstration:")

# Demonstrate errors
try:
    broken = img1 + img2
except InconsistentAddition as e:
    print(e)
try:
    broken = img_with_filters1 + img_with_filters2
except InconsistentAddition as e:
    print(e)
