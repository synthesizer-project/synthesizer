"""
SC-SAM example
==============

Load SC-SAM example data into a list of galaxy objects.
"""

import matplotlib.pyplot as plt
import numpy as np

from synthesizer import TEST_DATA_DIR
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.grid import Grid
from synthesizer.load_data.load_scsam import load_SCSAM

if __name__ == "__main__":
    # Define the grid
    grid_name = "test_grid.hdf5"
    grid = Grid(grid_name)

    # Define the emission model
    model = PacmanEmission(
        grid,
        tau_v=0.33,
        dust_curve=PowerLaw(slope=-1),
        fesc=0.1,
        fesc_ly_alpha=0.5,
    )

    # Load example SC-SAM SF history (just contains 10 galaxies) with
    # both methods
    test_data = f"{TEST_DATA_DIR}/sc-sam_sfhist.dat"
    galaxies = {
        method: load_SCSAM(test_data, method, grid)[0]
        for method in ("particle", "parametric")
    }

    # Spectrum that we want
    # (e.g. incident, nebular, intrinsic, emergent)
    spectrum = "emergent"

    # Plot the SEDs from each method
    for method, gals in galaxies.items():
        for galaxy in gals:
            if galaxy.stars is None:
                continue
            galaxy.stars.get_spectra(model)
            sed = galaxy.stars.spectra[spectrum]
            plt.plot(np.log10(sed.lam), np.log10(sed.lnu))
        plt.xlabel(r"$\log_{10}(\lambda/\rm{\AA})$")
        plt.ylabel(r"$\log_{10}(L_\nu/\rm{erg\,s^{-1}\,Hz^{-1}})$")
        plt.xlim(0, 8)
        plt.ylim(10, 35)
        plt.title(f"{method} method - {spectrum}")
        plt.grid(color="whitesmoke")
        plt.show()
