"""
Plot emission model diagnostics
===============================

This example compares the spectra produced by the traditional methods, emission
models, and a method utilising vectorisation
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.emission_models import (
    EmissionModel,
    IncidentEmission,
    ReprocessedEmission,
)
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.stars import sample_sfhz
from synthesizer.sed import combine_list_of_seds
from unyt import Myr

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def make_screen_emission(grid):
    """
    Makes an emission model to match the traditional methods.

    Args:
        grid (Grid)
            The grid object to use for the emission model.
    """
    # Define the old incident model
    old_incident = IncidentEmission(
        grid=grid,
        label="old_incident",
        mask_attr="ages",
        mask_thresh=10.0 * Myr,
        mask_op=">",
    )

    # Define the young reprocessed model
    young_reprocessed = ReprocessedEmission(
        grid=grid,
        label="young_reprocessed",
        mask_attr="ages",
        mask_thresh=10.0 * Myr,
        mask_op="<=",
    )

    # Define the intrinsic model
    intrinsic = EmissionModel(
        label="intrinsic",
        combine=[old_incident, young_reprocessed],
    )

    # Define the screen model
    screen = EmissionModel(
        grid=grid,
        label="screen",
        apply_dust_to=intrinsic,
        dust_curve=PowerLaw(slope=-1),
        tau_v=0.33,
    )

    return screen


def make_intrinsic_emission(grid):
    """
    Makes an emission model to match the singel sed methods.

    Args:
        grid (Grid)
            The grid object to use for the emission model.
    """
    # Define the old incident model
    old_incident = IncidentEmission(
        grid=grid,
        label="old_incident",
        mask_attr="ages",
        mask_thresh=10.0 * Myr,
        mask_op=">",
    )

    # Define the young reprocessed model
    young_reprocessed = ReprocessedEmission(
        grid=grid,
        label="young_reprocessed",
        mask_attr="ages",
        mask_thresh=10.0 * Myr,
        mask_op="<=",
    )

    # Define the intrinsic model
    intrinsic = EmissionModel(
        grid=grid,
        label="intrinsic",
        combine=[old_incident, young_reprocessed],
    )

    return intrinsic


def make_galaxies(ngal, sfh, metal_dist, log10ages, metallicities):
    """
    Make a list of galaxies.

    Args:
        ngal (int)
            The number of galaxies to make.

        sfh (SFH)
            The star formation history to use.

        metal_dist (ZDist)
            The metallicity distribution to use.

        log10ages (np.array)
            The log10 ages to use.

        metallicities (np.array)
            The metallicities to use.
    """
    # Define a list to hold all the galaxies we'll make
    galaxies = []

    make_start = time.time()

    # Loop over the number of galaxies
    for i in range(ngal):
        # How many stars will we make?
        nstars = 1000

        # Generate the star formation metallicity history
        mass = 10**6 * nstars
        param_stars = ParametricStars(
            log10ages,
            metallicities,
            sf_hist=sfh,
            metal_dist=metal_dist,
            initial_mass=mass,
        )

        # Sample the SFZH, producing a Stars object
        # we will also pass some keyword arguments for attributes
        # we will need for imaging
        stars = sample_sfhz(
            param_stars.sfzh,
            param_stars.log10ages,
            param_stars.log10metallicities,
            nstars,
            redshift=1,
        )

        # Create galaxy object
        galaxies.append(Galaxy("Galaxy", stars=stars, redshift=1))

    print(f"Made {ngal} galaxies in {time.time() - make_start:.2f} seconds")

    return galaxies


def get_spectra_traditional(gals, grid, age_pivot=10.0 * Myr):
    """
    Get the spectra for a galaxy.

    This function uses the "traditional" method of calculating the spectra
    for individual galaxies using methods on a component.

    Args:
        gals (list)
            A list of galaxy objects.
        grid (Grid)
            The grid object to use for the emission model.
        age_pivot (unyt_quantity)
            The age to pivot the spectra at for the traditional method.
    """
    for gal in gals:
        spec = {}

        # Get pure stellar spectra
        old_spec = gal.stars.get_spectra_incident(grid, old=age_pivot)

        # Get nebular spectra for each star particle
        young_reprocessed_spec = gal.stars.get_spectra_reprocessed(
            grid, young=age_pivot
        )

        # Save intrinsic stellar spectra
        spec["intrinsic"] = young_reprocessed_spec + old_spec

        # Simple screen model
        spec["screen"] = spec["intrinsic"].apply_attenuation(
            dust_curve=PowerLaw(slope=-1), tau_v=0.33
        )


def get_spectra_single_sed(gals, grid, age_pivot=10.0 * Myr):
    """
    Get the spectra for a galaxy.

    This function uses the "traditional" method of calculating the spectra
    for individual galaxies but only applies attenuation .

    Args:
        gals (list)
            A list of galaxy objects.
        grid (Grid)
            The grid object to use for the emission model.
        age_pivot (unyt_quantity)
            The age to pivot the spectra at for the traditional method.
    """
    seds = []
    for gal in gals:
        spec = {}

        # Get young pure stellar spectra
        young_spec = gal.stars.get_spectra_incident(grid, young=age_pivot)

        # Get pure stellar spectra
        old_spec = gal.stars.get_spectra_incident(grid, old=age_pivot)

        spec["stellar"] = old_spec + young_spec

        # Get nebular spectra for each star particle
        young_reprocessed_spec = gal.stars.get_spectra_reprocessed(
            grid, young=age_pivot
        )

        # Simple screen model
        seds.append(young_reprocessed_spec + old_spec)

    # Combine seds
    sed = combine_list_of_seds(seds)

    sed.apply_attenuation(dust_curve=PowerLaw(slope=-1), tau_v=0.33)


def get_spectra_emission_model(gals, model):
    """
    Get the spectra for a galaxy using an emission model.

    Args:
        gals (list)
            A list of galaxy objects.
        model (EmissionModel)
            The emission model to use.
    """
    for gal in gals:
        gal.stars.get_spectra(model)


def get_spectra_emission_model_single_sed(gals, model):
    """
    Get the spectra for a galaxy combining an emission model with a single sed.

    Args:
        gals (list)
            A list of galaxy objects.
        model (EmissionModel)
            The emission model to use.
    """
    seds = []
    for gal in gals:
        gal.stars.get_spectra(model)
        seds.append(gal.stars.spectra["intrinsic"])

    # Combine seds
    sed = combine_list_of_seds(seds)

    sed.apply_attenuation(dust_curve=PowerLaw(slope=-1), tau_v=0.33)


# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Make the emission models
screen_model = make_screen_emission(grid)
intrinsic_model = make_intrinsic_emission(grid)

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
Z_p = {"metallicity": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(**sfh_p)  # constant star formation

# Define the number of galaxies
ngals = np.logspace(1, 4, 10, dtype=int)

# Define lists to hold the runtimes
traditional_times = []
single_sed_times = []
screen_model_times = []
single_sed_model_times = []


# Loop over different numbers of galaxies
for ngal in ngals:
    # Make the galaxies
    gals = make_galaxies(ngal, sfh, metal_dist, log10ages, metallicities)

    # Time each method
    start_time = time.time()
    get_spectra_traditional(gals, grid)
    traditional_times.append(time.time() - start_time)
    print(f"Traditional method took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    get_spectra_single_sed(gals, grid)
    single_sed_times.append(time.time() - start_time)
    print(f"Single sed method took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    get_spectra_emission_model(gals, screen_model)
    screen_model_times.append(time.time() - start_time)
    print(f"Screen model method took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    get_spectra_emission_model_single_sed(gals, intrinsic_model)
    single_sed_model_times.append(time.time() - start_time)
    print(
        f"Single sed model method took {time.time() - start_time:.2f} seconds"
    )

    print("===========================================")

# Plot the timing results
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

ax.plot(ngals, traditional_times, label="Traditional")
ax.plot(ngals, single_sed_times, label="Single SED")
ax.plot(ngals, screen_model_times, label="Screen Model", linestyle="--")
ax.plot(
    ngals, single_sed_model_times, label="Single SED Model", linestyle="--"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of Galaxies")
ax.set_ylabel("Time (s)")

ax.legend()

plt.tight_layout()
plt.show()
