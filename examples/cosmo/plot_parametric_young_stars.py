"""
Parametric Young Stars Example
==============================

Test the effect on the intrinsic emission of assuming a
parametric SFH for young star particles.

"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Myr

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.parametric import SFH, Stars
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    grid_dir = "../../tests/test_grid"
    grid_name = "test_grid"
    grid = Grid(grid_name, grid_dir=grid_dir)

    gals = load_CAMELS_IllustrisTNG(
        "../../tests/data/",
        snap_name="camels_snap.hdf5",
        group_name="camels_subhalo.hdf5",
        group_dir="../../tests/data/",
    )

    # Define the emission model
    model = IncidentEmission(grid)

    # Select a single galaxy
    gal = gals[2]

    # Age limit at which we replace star particles
    age_lim = 500 * Myr

    """
    We first demonstrate the process *manually*.
    This also allows us to obtain the SFH of each parametric
    model for plotting purposes.
    """
    # First, filter for star particles
    pmask = gal.stars.ages < age_lim

    stars = []
    # Loop through each young star particle
    for _pmask in np.where(pmask)[0]:
        # Initialise SFH object
        sfh = SFH.Constant(max_age=age_lim)

        # Create a parametric stars object
        stars.append(
            Stars(
                grid.log10age,
                grid.metallicity,
                sf_hist=sfh,
                metal_dist=gal.stars.metallicities[_pmask],
                initial_mass=gal.stars.initial_masses[_pmask],
            )
        )

    # Sum each individual Stars object
    stars = sum(stars[1:], stars[0])

    # Create a parametric galaxy
    para_gal = Galaxy(stars)

    para_spec = para_gal.stars.get_spectra(model)
    part_spec = gal.stars.get_spectra(model)
    part_spec_old = gal.stars.get_spectra(
        model,
        mask={"incident": {"attr": "ages", "op": ">", "thresh": age_lim}},
    )

    """
    We can also do this directly by calling the parametric_young_stars
    method on the Stars to resample in place.
    """
    gal.stars.parametric_young_stars(
        age=age_lim,
        parametric_sfh="constant",
        grid=grid,
    )

    combined_spec = gal.stars.get_spectra(model)

    """
    Plot intrinsic emission from pure particle, parametric
    and parametric + particle models
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.loglog(
        para_spec.lam, para_spec.lnu, label="Parametric young", color="C0"
    )
    ax1.loglog(
        part_spec_old.lam, part_spec_old.lnu, label="Particle old", color="C3"
    )
    ax1.loglog(
        part_spec.lam,
        part_spec.lnu,
        label="Particle all",
        color="C1",
        linestyle="dashed",
    )
    ax1.loglog(
        part_spec.lam,
        part_spec_old.lnu + para_spec.lnu,
        label="Para + Part",
        color="C2",
    )
    ax1.set_ylim(1e20, 1e30)
    ax1.set_xlim(1e2, 2e4)
    ax1.legend()
    ax1.set_xlabel("$\\lambda \\,/\\, \\AA$")
    ax1.set_ylabel("$L_{\\lambda} / \\mathrm{erg / Hz / s}$")

    """
    Plot SFH from particles and parametric
    """
    binLimits = np.linspace(5, 10, 30)

    ax2.hist(
        np.log10(gal.stars.ages),
        histtype="step",
        weights=gal.stars.initial_masses.value,
        bins=binLimits,
        log=True,
        label="All Particles",
        color="C1",
        linestyle="dashed",
        linewidth=3,
    )
    ax2.hist(
        np.log10(stars.ages.value),
        histtype="step",
        weights=stars.sf_hist,
        bins=binLimits,
        log=True,
        label="Young Parametric",
        color="C0",
        linewidth=3,
        linestyle="dashed",
    )
    ax2.legend()
    # plt.show()
    ax2.set_xlabel("$\\mathrm{log_{10} Age \\,/\\, yr}$")
    ax2.set_ylabel("$\\mathrm{log_{10} (Mass \\,/\\, M_{\\odot})}$")

    plt.show()
