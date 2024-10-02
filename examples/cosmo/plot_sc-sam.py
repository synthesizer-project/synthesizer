"""
SC-SAM example
==============

Load SC-SAM example data into a list of galaxy objects.
"""

import matplotlib.pyplot as plt
import numpy as np

from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.grid import Grid
from synthesizer.load_data.load_scsam import load_SCSAM

if __name__ == "__main__":
    # Define the grid
    grid_name = "test_grid.hdf5"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the emission model
    model = PacmanEmission(
        grid,
        tau_v=0.33,
        dust_curve=PowerLaw(slope=-1),
        fesc=0.1,
        fesc_ly_alpha=0.5,
    )

    # Load example SC-SAM SF history (just contains 10 galaxies)
    test_data = "../../tests/data/sc-sam_sfhist.dat"
    # Obtain galaxy objects using different methods:
    # Particle method
    particle_galaxies, _, _ = load_SCSAM(test_data, "particle")
    # Paramteric method, interpolating the grid using scipy's
    # nearest ND interpolator
    parametric_NNI_galaxies, _, _ = load_SCSAM(
        test_data, "parametric_NNI", grid
    )
    # Paramteric method, interpolating the grid using scipy's
    # regular grid interpolator
    parametric_RGI_galaxies, _, _ = load_SCSAM(
        test_data, "parametric_RGI", grid
    )

    # Set up arrays to store galaxy SEDs
    particle_SEDs = []
    parametric_NNI_SEDs = []
    parametric_RGI_SEDs = []

    # Spectrum that we want
    # (e.g. incident, nebular, intrinsic, emergent)
    spectrum = "emergent"

    # Loop over each galaxy object
    for i in range(len(particle_galaxies)):
        # Get SEDs for the particle galaxy object
        particle_galaxy = particle_galaxies[i]
        particle_galaxy.stars.get_spectra(model)
        particle_sed = particle_galaxy.stars.spectra[spectrum]
        particle_SEDs.append(particle_sed.lnu)

        # Get SEDs for the parametric NNI galaxy object
        parametric_NNI_galaxy = parametric_NNI_galaxies[i]
        parametric_NNI_galaxy.stars.get_spectra(model)
        parametric_sed = parametric_NNI_galaxy.stars.spectra[spectrum]
        parametric_NNI_SEDs.append(parametric_sed.lnu)

        # Get SEDs for the parametric RGI galaxy object
        parametric_RGI_galaxy = parametric_RGI_galaxies[i]
        parametric_RGI_galaxy.stars.get_spectra(model)
        parametric_sed = parametric_RGI_galaxy.stars.spectra[spectrum]
        parametric_RGI_SEDs.append(parametric_sed.lnu)

    # Plot SEDs
    for lnu in particle_SEDs:
        plt.plot(np.log10(particle_sed.lam), np.log10(lnu))
        plt.xlabel(r"$\log_{10}(\lambda/\rm{\AA})$")
        plt.ylabel(
            r"$\log_{10}(L_\nu/\rm{erg\,s^{-1}\,Hz^{-1}\,M_{\odot}^{-1}})$"
        )
        plt.xlim(0, 8)
        plt.ylim(10, 35)
        plt.title(f"simple particle method - {spectrum}")
        plt.grid(color="whitesmoke")
    plt.show()

    for lnu in parametric_NNI_SEDs:
        plt.plot(np.log10(parametric_sed.lam), np.log10(lnu))
        plt.xlabel(r"$\log_{10}(\lambda/\rm{\AA})$")
        plt.ylabel(
            r"$\log_{10}(L_\nu/\rm{erg\,s^{-1}\,Hz^{-1}\,M_{\odot}^{-1}})$"
        )
        plt.xlim(0, 8)
        plt.ylim(10, 35)
        plt.title(f"NNI parametric method - {spectrum}")
        plt.grid(color="whitesmoke")
    plt.show()

    for lnu in parametric_RGI_SEDs:
        plt.plot(np.log10(parametric_sed.lam), np.log10(lnu))
        plt.xlabel(r"$\log_{10}(\lambda/\rm{\AA})$")
        plt.ylabel(
            r"$\log_{10}(L_\nu/\rm{erg\,s^{-1}\,Hz^{-1}\,M_{\odot}^{-1}})$"
        )
        plt.xlim(0, 8)
        plt.ylim(10, 35)
        plt.title(f"RGI parametric method - {spectrum}")
        plt.grid(color="whitesmoke")
    plt.show()
