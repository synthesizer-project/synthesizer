"""
Image addition example
======================

An example of how to do image addition and testing error functionality.
"""

import numpy as np
from unyt import Msun, Myr, kpc

from synthesizer.emission_models import IncidentEmission
from synthesizer.exceptions import InconsistentAddition
from synthesizer.filters import FilterCollection as Filters
from synthesizer.grid import Grid
from synthesizer.imaging import Image, ImageCollection
from synthesizer.parametric import Stars
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.parametric.morphology import Sersic2D

if __name__ == "__main__":
    # First set up some stuff so we can make demonstration images.

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want the directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define an arbitrary morphology to feed the imaging
    morph = Sersic2D(
        r_eff=1.0 * kpc, sersic_index=1.0, ellipticity=0.5, theta=35.0
    )

    # Define the parameters of the star formation and metal enrichment
    # histories
    stellar_mass = 1e10 * Msun
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=10.0 * Myr,
        metal_dist=0.01,
        initial_mass=stellar_mass,
        morphology=morph,
    )

    # Define the emission model
    model = IncidentEmission(grid)

    # Get galaxy object
    galaxy = Galaxy(stars=stars)

    # Get specrtra
    sed = galaxy.stars.get_spectra(model)

    # Create a filter collection
    filter_codes1 = [
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F200W",
    ]
    filter_codes2 = filter_codes1[:-1]
    filters1 = Filters(filter_codes1)
    filters2 = Filters(filter_codes2)

    fake_img = np.zeros((100, 100))

    # Create fake dicts of images we'll put in image collections
    fakes_imgs1 = {f: fake_img for f in filter_codes1}
    fakes_imgs2 = {f: fake_img for f in filter_codes2}

    # Create fake image objects
    res1 = 0.5 * kpc
    res2 = 0.4 * kpc
    img1 = Image(resolution=res1, fov=100 * res1, img=fake_img)
    img2 = Image(resolution=res2, fov=100 * res2, img=fake_img)
    img_with_filters1 = ImageCollection(
        resolution=res1,
        fov=100 * res1,
        imgs=fakes_imgs1,
    )
    img_with_filters2 = ImageCollection(
        resolution=res1,
        fov=100 * res1,
        imgs=fakes_imgs2,
    )

    # Add images
    composite = img1 + img1 + img1
    composite_with_filters = img_with_filters1 + img_with_filters1

    print("Error Demonstration:")

    # Demonstrate errors
    try:
        broken = img1 + img2
    except InconsistentAddition as e:
        print(e)
