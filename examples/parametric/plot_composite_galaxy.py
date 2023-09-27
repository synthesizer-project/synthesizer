"""
Generate composite image example
================================

Example for generating a composite galaxy
photometry. This example will:
- build two parametric "galaxies" (see make_sfzh)
- calculate spectral luminosity density of each
- make images of both
"""


from unyt import Myr, kpc
import matplotlib.pyplot as plt
import numpy as np

from synthesizer.grid import Grid
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.parametric.sfzh import (SFH,
                                         ZH,
                                         generate_sfzh,
                                         generate_instant_sfzh)
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.filters import UVJ


if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Get a UVJ filter collection
    filters = UVJ()

    # Define geometry of the images
    resolution = 0.05 * kpc  # resolution in kpc
    npix = 50  # number of pixels wide
    fov = resolution.value * npix * kpc  # field-of-view

    # ===================== Make Disk =====================

    # Define morphology
    morph = Sersic2D(r_eff_kpc=1.0 * kpc, n=1.0, ellip=0.5, theta=35.0)

    # Define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"duration": 10 * Myr}
    Z_p = {"log10Z": -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 10**8.5

    # Define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # Get the 2D star formation and metal enrichment history for the given
    # SPS grid. This is (age, Z).
    sfzh = generate_sfzh(
        grid.log10age, grid.metallicity, sfh, Zh, stellar_mass=stellar_mass
    )

    # Initialise Galaxy object
    disk = Galaxy(morph=morph, sfzh=sfzh)

    # Generate stellar spectra
    disk.get_spectra_incident(grid)

    # Make images
    disk_img = disk.make_images(
        resolution=resolution,
        filters=filters,
        sed=disk.spectra["incident"],
        fov=fov,
    )

    print(disk)

    # Make and plot an rgb image
    disk_img.make_rgb_image(rgb_filters={"R": "J", "G": "V", "B": "U"})
    fig, ax, _ = disk_img.plot_rgb_image(show=True)

    # ===================== Make Bulge =====================

    # Define bulge morphology
    morph = Sersic2D(r_eff_kpc=1.0 * kpc, n=4.0)

    # Define the parameters of the star formation and metal enrichment 
    # histories
    stellar_mass = 2e10
    sfzh = generate_instant_sfzh(
        grid.log10age, grid.metallicity, 10.0, 0.01, stellar_mass=stellar_mass
    )

    # Get galaxy object
    bulge = Galaxy(morph=morph, sfzh=sfzh)

    # Get specrtra
    bulge.get_spectra_incident(grid)

    # make images
    bulge_img = bulge.make_images(
        resolution=resolution,
        filters=filters,
        sed=bulge.spectra["incident"],
        fov=fov,
    )

    print(bulge)

    # Make and plot an rgb image
    bulge_img.make_rgb_image(rgb_filters={"R": "J", "G": "V", "B": "U"})
    fig, ax, _ = bulge_img.plot_rgb_image(show=True)

    # ===================== Make Composite =====================

    # Combine galaxy objects
    combined = disk + bulge

    print(combined)

    # Combine images
    total = disk_img + bulge_img

    # Make and plot an rgb image
    total.make_rgb_image(rgb_filters={"R": "J", "G": "V", "B": "U"})
    fig, ax, _ = total.plot_rgb_image(show=True)

    # Plot the spectra of both components

    sed = disk.spectra["incident"]
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, c="b", 
             label="disk")

    sed = bulge.spectra["incident"]
    plt.plot(
        np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, c="r", 
        label="bulge"
    )

    sed = combined.spectra["incident"]
    plt.plot(
        np.log10(sed.lam), np.log10(sed.lnu), lw=2, alpha=0.8, c="k", 
        label="combined"
    )

    plt.xlim(3.0, 4.3)
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r"$\rm log_{10}(\lambda_{obs}/\AA)$")
    plt.ylabel(r"$\rm log_{10}(f_{\nu}/nJy)$")

    plt.show()
