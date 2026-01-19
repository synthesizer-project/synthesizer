"""Profile runtime scaling with the number of particles.

This script generates three separate plots (Spectra, Photometry, Imaging)
showing how various operations scale from 10^3 to 10^5 particles.
"""

import argparse
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
from synthesizer.utils.profiling_utils import (
    get_instrument_profile,
)

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def profile_nparticles(nthreads=1, n_averages=3):
    """Run the profiling."""
    print(
        f"Initializing Grid and Models (nthreads={nthreads}, "
        f"n_averages={n_averages})..."
    )
    grid = Grid("test_grid")
    n_lam = grid.nlam

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
    fov = 30 * kpc
    npix_low = 100
    npix_high = 1000
    res_low = fov / npix_low
    res_high = fov / npix_high

    # Create instruments for imaging
    inst_dir = Path("profiling/instruments")
    inst_dir.mkdir(parents=True, exist_ok=True)

    inst_low = get_instrument_profile(
        label="low_res",
        filepath=str(inst_dir / "low_res.hdf5"),
        filters=filters_3,
        resolution=res_low,
    )
    inst_high = get_instrument_profile(
        label="high_res",
        filepath=str(inst_dir / "high_res.hdf5"),
        filters=filters_3,
        resolution=res_high,
    )

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
            f"Smoothed ({npix_low}x{npix_low})": [],
            f"Smoothed ({npix_high}x{npix_high})": [],
            f"Histogram ({npix_low}x{npix_low})": [],
            f"Histogram ({npix_high}x{npix_high})": [],
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

        # Local storage for averages
        iter_times = {
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
                f"Smoothed ({npix_low}x{npix_low})": [],
                f"Smoothed ({npix_high}x{npix_high})": [],
                f"Histogram ({npix_low}x{npix_low})": [],
                f"Histogram ({npix_high}x{npix_high})": [],
            },
        }

        for i in range(n_averages):
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
            stars.get_spectra(model_part, nthreads=nthreads)
            iter_times["spectra"]["Particle"].append(
                time.perf_counter() - start
            )

            # Particle Shift
            start = time.perf_counter()
            stars.get_spectra(model_part_shift, nthreads=nthreads)
            iter_times["spectra"]["Particle (Doppler)"].append(
                time.perf_counter() - start
            )

            # Integrated
            start = time.perf_counter()
            stars.get_spectra(model_int, nthreads=nthreads)
            iter_times["spectra"]["Integrated"].append(
                time.perf_counter() - start
            )

            # Integrated Shift
            start = time.perf_counter()
            stars.get_spectra(model_int_shift, nthreads=nthreads)
            iter_times["spectra"]["Integrated (Doppler)"].append(
                time.perf_counter() - start
            )

            # The previous step generated 'part' and 'int' spectra, so we can
            # use those.
            start = time.perf_counter()
            stars.get_particle_photo_lnu(filters_3)
            iter_times["photometry"]["Particle (3 filters)"].append(
                time.perf_counter() - start
            )

            # Particle (10 filters)
            start = time.perf_counter()
            stars.get_particle_photo_lnu(filters_10)
            iter_times["photometry"]["Particle (10 filters)"].append(
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
            iter_times["photometry"]["Integrated (3 filters)"].append(
                time.perf_counter() - start
            )

            start = time.perf_counter()
            sed_int.get_photo_lnu(filters_10)
            iter_times["photometry"]["Integrated (10 filters)"].append(
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
                "part",
                fov=fov,
                instrument=inst_low,
                kernel=kernel,
                img_type="smoothed",
                nthreads=nthreads,
            )
            iter_times["imaging"][f"Smoothed ({npix_low}x{npix_low})"].append(
                time.perf_counter() - start
            )

            # Smoothed (0.01 kpc)
            start = time.perf_counter()
            stars.get_images_luminosity(
                "part",
                fov=fov,
                instrument=inst_high,
                kernel=kernel,
                img_type="smoothed",
                nthreads=nthreads,
            )
            iter_times["imaging"][
                f"Smoothed ({npix_high}x{npix_high})"
            ].append(time.perf_counter() - start)

            # Histogram (0.1 kpc)
            start = time.perf_counter()
            stars.get_images_luminosity(
                "part",
                fov=fov,
                instrument=inst_low,
                img_type="hist",
                nthreads=nthreads,
            )
            iter_times["imaging"][f"Histogram ({npix_low}x{npix_low})"].append(
                time.perf_counter() - start
            )

            # Histogram (0.01 kpc)
            start = time.perf_counter()
            stars.get_images_luminosity(
                "part",
                fov=fov,
                instrument=inst_high,
                img_type="hist",
                nthreads=nthreads,
            )
            iter_times["imaging"][
                f"Histogram ({npix_high}x{npix_high})"
            ].append(time.perf_counter() - start)

            # Force garbage collection
            del stars
            gc.collect()

        # Store averages
        for cat in iter_times:
            for label in iter_times[cat]:
                times[cat][label].append(np.mean(iter_times[cat][label]))

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
        ax.grid(True, alpha=0.3, which="major")
        ax.legend()
        ax.set_title(
            f"Particle Performance (n_lam={n_lam}, nthreads={nthreads})"
        )

        plt.tight_layout()
        filename = (
            f"nparticles_performance_{category_name}_"
            f"nlam{n_lam}_nt{nthreads}.png"
        )
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    make_plot("spectra", "scaling_nparticles_spectra.png")
    make_plot("photometry", "scaling_nparticles_photometry.png")
    make_plot("imaging", "scaling_nparticles_imaging.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    args = parser.parse_args()

    profile_nparticles(nthreads=args.nthreads, n_averages=args.n_averages)
