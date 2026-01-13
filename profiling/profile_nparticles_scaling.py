"""Profile runtime scaling with the number of particles.

This script generates three separate plots (Spectra, Photometry, Imaging)
showing how various operations scale from 10^3 to 10^6 particles.
"""

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, kpc

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh

# Set style
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def profile_nparticles():
    """Run the profiling."""
    print("Initializing Grid and Models...")
    grid = Grid("test_grid")

    # --- Setup Models ---
    # Particle, No Shift
    model_part = IncidentEmission(grid, per_particle=True, label="part")
    # Particle, With Shift
    model_part_shift = IncidentEmission(
        grid, per_particle=True, label="part_shift", vel_shift=True
    )
    # Integrated, No Shift
    model_int = IncidentEmission(grid, per_particle=False, label="int")
    # Integrated, With Shift
    model_int_shift = IncidentEmission(
        grid, per_particle=False, label="int_shift", vel_shift=True
    )

    # --- Setup Filters ---
    # Small set (3 filters)
    filters_3 = FilterCollection(
        filter_codes=[
            "JWST/NIRCam.F150W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F444W",
        ],
        new_lam=grid.lam,
    )
    # Large set (10 filters) - Mixing JWST and HST
    filters_10 = FilterCollection(
        filter_codes=[
            "JWST/NIRCam.F070W",
            "JWST/NIRCam.F090W",
            "JWST/NIRCam.F115W",
            "JWST/NIRCam.F150W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F277W",
            "JWST/NIRCam.F356W",
            "JWST/NIRCam.F444W",
            "HST/ACS_WFC.F435W",
            "HST/ACS_WFC.F814W",
        ],
        new_lam=grid.lam,
    )

    # --- Setup Imaging ---
    kernel = Kernel().get_kernel()
    res_low = 0.1 * kpc
    res_high = 0.01 * kpc
    fov = 30 * kpc

    # Particle counts to test
    n_particles = np.logspace(3, 5, 5).astype(int)

    # Storage for results
    times = {
        "spectra": {
            "Particle": [],
            "Particle (Doppler)": [],
            "Integrated": [],
            "Integrated (Doppler)": [],
        },
        "photometry": {
            "Particle (3 filters)": [],
            "Particle (10 filters)": [],
            "Integrated (3 filters)": [],
            "Integrated (10 filters)": [],
        },
        "imaging": {
            "Smoothed (0.1 kpc)": [],
            "Smoothed (0.01 kpc)": [],
            "Histogram (0.1 kpc)": [],
            "Histogram (0.01 kpc)": [],
        },
    }

    # Standard parametric setup for sampling
    mass = 10**10 * Msun
    param_stars = ParametricStars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=mass,
    )

    for n in n_particles:
        print(f"Profiling n={n}...")

        # --- 1. Spectra Profiling ---
        # Re-sample for each iteration to keep environment clean
        stars = sample_sfzh(
            param_stars.sfzh,
            param_stars.log10ages,
            param_stars.log10metallicities,
            n,
            redshift=1,
        )
        stars.velocities = (
            np.random.randn(n, 3) * 100 * (kpc / Myr)
        )  # Needs velocities for shift

        # Particle
        start = time.perf_counter()
        stars.get_spectra(model_part)
        times["spectra"]["Particle"].append(time.perf_counter() - start)

        # Particle Shift
        start = time.perf_counter()
        stars.get_spectra(model_part_shift)
        times["spectra"]["Particle (Doppler)"].append(
            time.perf_counter() - start
        )

        # Integrated
        start = time.perf_counter()
        stars.get_spectra(model_int)
        times["spectra"]["Integrated"].append(time.perf_counter() - start)

        # Integrated Shift
        start = time.perf_counter()
        stars.get_spectra(model_int_shift)
        times["spectra"]["Integrated (Doppler)"].append(
            time.perf_counter() - start
        )

        # The previous step generated 'part' and 'int' spectra, so we can use
        # those.
        start = time.perf_counter()
        stars.get_particle_photo_lnu(filters_3)
        times["photometry"]["Particle (3 filters)"].append(
            time.perf_counter() - start
        )

        # Particle (10 filters)
        start = time.perf_counter()
        stars.get_particle_photo_lnu(filters_10)
        times["photometry"]["Particle (10 filters)"].append(
            time.perf_counter() - start
        )

        # Note: stars.get_photo_lnu usually works on cached spectra.
        # We need to target the integrated model label 'int'.
        # Assuming stars object has a method for this or we access the
        # spectra directly.
        # But `Stars` (the particle object) has `get_particle_photo_lnu`.
        # It DOES NOT usually have a direct method for integrated
        # photometry of its integrated spectra other than interacting with
        # the Sed object directly.
        # However, `StarsComponent` (parent) might have `get_photo_lnu`?
        # Let's check `get_photo_lnu` on the integrated spectra object
        # directly for fair timing.

        sed_int = stars.spectra["int"]

        start = time.perf_counter()
        sed_int.get_photo_lnu(filters_3)
        times["photometry"]["Integrated (3 filters)"].append(
            time.perf_counter() - start
        )

        start = time.perf_counter()
        sed_int.get_photo_lnu(filters_10)
        times["photometry"]["Integrated (10 filters)"].append(
            time.perf_counter() - start
        )

        # --- 3. Imaging Profiling ---
        # Setup coordinates and smoothing lengths
        stars.coordinates = np.random.randn(n, 3) * kpc
        stars.centre = np.array([0, 0, 0]) * kpc
        stars.calculate_smoothing_lengths(num_neighbours=50)

        # We use the 'part' (no shift) model for imaging as standard

        # Smoothed (0.1 kpc)
        start = time.perf_counter()
        stars.get_images_luminosity(
            model_part,
            resolution=res_low,
            fov=fov,
            kernel=kernel,
            img_type="smoothed",
        )
        times["imaging"]["Smoothed (0.1 kpc)"].append(
            time.perf_counter() - start
        )

        # Smoothed (0.01 kpc)
        start = time.perf_counter()
        stars.get_images_luminosity(
            model_part,
            resolution=res_high,
            fov=fov,
            kernel=kernel,
            img_type="smoothed",
        )
        times["imaging"]["Smoothed (0.01 kpc)"].append(
            time.perf_counter() - start
        )

        # Histogram (0.1 kpc)
        start = time.perf_counter()
        stars.get_images_luminosity(
            model_part, resolution=res_low, fov=fov, img_type="hist"
        )
        times["imaging"]["Histogram (0.1 kpc)"].append(
            time.perf_counter() - start
        )

        # Histogram (0.01 kpc)
        start = time.perf_counter()
        stars.get_images_luminosity(
            model_part, resolution=res_high, fov=fov, img_type="hist"
        )
        times["imaging"]["Histogram (0.01 kpc)"].append(
            time.perf_counter() - start
        )

        # Force garbage collection
        del stars
        gc.collect()

    # --- Plotting ---
    output_dir = Path("profiling/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_plot(category_name, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = times[category_name]

        # Style cycle
        markers = ["o", "s", "d", "v", "^", "<", ">"]

        for i, (label, values) in enumerate(data.items()):
            ax.loglog(
                n_particles,
                values,
                marker=markers[i % len(markers)],
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("Number of Particles")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    make_plot("spectra", "scaling_nparticles_spectra.png")
    make_plot("photometry", "scaling_nparticles_photometry.png")
    make_plot("imaging", "scaling_nparticles_imaging.png")


if __name__ == "__main__":
    profile_nparticles()
