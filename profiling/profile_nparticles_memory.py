"""Profile memory scaling with the number of particles.

This script generates three separate plots (Spectra, Photometry, Imaging)
showing how the peak memory usage of various operations scales from 10^3 to
10^6 particles.
"""

import argparse
import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from memory_profiler import memory_usage
from unyt import Msun, Myr, kpc

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh
from synthesizer.utils.profiling_utils import get_instrument_profile

# Set style
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def get_current_mem():
    """Return the current process memory in MiB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_and_measure_memory(func, *args, interval=0.01, **kwargs):
    """Run a function and return its peak memory usage INCREASE in GB.

    This subtracts the baseline memory usage before the call to isolate
    the memory cost of the operation itself.
    """
    gc.collect()
    baseline = get_current_mem()
    peak = memory_usage(
        (func, args, kwargs), interval=interval, max_usage=True
    )
    return max(0, peak - baseline) / 1024  # Convert MiB to GB


def profile_nparticles_memory(nthreads=1, n_averages=3, mem_interval=0.01):
    """Run the profiling."""
    print(
        f"Initializing Grid and Models (nthreads={nthreads}, "
        f"n_averages={n_averages}, mem_interval={mem_interval})..."
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
    # Reduced max to 10^5 for memory safety/speed in this context,
    # or keep 10^3 to 10^5 as in original file logic (actually logspace(3,5,5))
    n_particles = np.logspace(3, 5, 5).astype(int)

    # Storage for results
    mems = {
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
        print(f"Profiling Memory n={n}...")

        # Local storage for averages
        iter_mems = {
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
            # Re-sample for each iteration
            stars = sample_sfzh(
                param_stars.sfzh,
                param_stars.log10ages,
                param_stars.log10metallicities,
                n,
                redshift=1,
            )
            stars.velocities = np.random.randn(n, 3) * 100 * (kpc / Myr)

            # Particle
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_part,
                interval=mem_interval,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Particle"].append(mem)
            del stars.spectra["part"]

            # Particle Shift
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_part_shift,
                interval=mem_interval,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Particle (Doppler)"].append(mem)
            del stars.spectra["part_shift"]

            # Integrated
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_int,
                interval=mem_interval,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Integrated"].append(mem)
            del stars.spectra["int"]

            # Integrated Shift
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_int_shift,
                interval=mem_interval,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Integrated (Doppler)"].append(mem)

            # Re-generate necessary spectra for photometry
            stars.get_spectra(model_part, nthreads=nthreads)
            stars.get_spectra(model_int, nthreads=nthreads)

            # Particle (3 filters)
            mem = run_and_measure_memory(
                stars.get_particle_photo_lnu, filters_3, interval=mem_interval
            )
            iter_mems["photometry"]["Particle (3 filters)"].append(mem)

            # Particle (10 filters)
            mem = run_and_measure_memory(
                stars.get_particle_photo_lnu, filters_10, interval=mem_interval
            )
            iter_mems["photometry"]["Particle (10 filters)"].append(mem)

            # Integrated Photometry
            sed_int = stars.spectra["int"]

            mem = run_and_measure_memory(
                sed_int.get_photo_lnu, filters_3, interval=mem_interval
            )
            iter_mems["photometry"]["Integrated (3 filters)"].append(mem)

            mem = run_and_measure_memory(
                sed_int.get_photo_lnu, filters_10, interval=mem_interval
            )
            iter_mems["photometry"]["Integrated (10 filters)"].append(mem)

            # --- 3. Imaging Profiling ---
            # Setup coordinates and smoothing lengths
            stars.coordinates = np.random.randn(n, 3) * kpc
            stars.centre = np.array([0, 0, 0]) * kpc
            stars.calculate_smoothing_lengths(num_neighbours=50)

            # Smoothed (0.1 kpc)
            mem = run_and_measure_memory(
                stars.get_images_luminosity,
                "part",
                interval=mem_interval,
                fov=fov,
                instrument=inst_low,
                kernel=kernel,
                img_type="smoothed",
                nthreads=nthreads,
            )
            iter_mems["imaging"][f"Smoothed ({npix_low}x{npix_low})"].append(
                mem
            )

            # Smoothed (0.01 kpc)
            mem = run_and_measure_memory(
                stars.get_images_luminosity,
                "part",
                interval=mem_interval,
                fov=fov,
                instrument=inst_high,
                kernel=kernel,
                img_type="smoothed",
                nthreads=nthreads,
            )
            iter_mems["imaging"][f"Smoothed ({npix_high}x{npix_high})"].append(
                mem
            )

            # Histogram (0.1 kpc)
            mem = run_and_measure_memory(
                stars.get_images_luminosity,
                "part",
                interval=mem_interval,
                fov=fov,
                instrument=inst_low,
                img_type="hist",
                nthreads=nthreads,
            )
            iter_mems["imaging"][f"Histogram ({npix_low}x{npix_low})"].append(
                mem
            )

            # Histogram (0.01 kpc)
            mem = run_and_measure_memory(
                stars.get_images_luminosity,
                "part",
                interval=mem_interval,
                fov=fov,
                instrument=inst_high,
                img_type="hist",
                nthreads=nthreads,
            )
            iter_mems["imaging"][
                f"Histogram ({npix_high}x{npix_high})"
            ].append(mem)

            # Force garbage collection
            del stars
            gc.collect()

        # Store averages
        for cat in iter_mems:
            for label in iter_mems[cat]:
                mems[cat][label].append(np.mean(iter_mems[cat][label]))

    # --- Plotting ---
    output_dir = Path("profiling/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_plot(category_name, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = mems[category_name]

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
        ax.set_ylabel("Peak Memory Increase (GB)")
        ax.grid(True, alpha=0.3, which="major")
        ax.legend()
        ax.set_title(
            f"Particle Performance (n_lam={n_lam}, nthreads={nthreads})"
        )

        plt.tight_layout()
        filename = (
            f"nparticles_performance_memory_{category_name}_"
            f"nlam{n_lam}_nt{nthreads}.png"
        )
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    make_plot("spectra", "scaling_nparticles_memory_spectra.png")
    make_plot("photometry", "scaling_nparticles_memory_photometry.png")
    make_plot("imaging", "scaling_nparticles_memory_imaging.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    parser.add_argument("--mem_interval", type=float, default=0.01)
    args = parser.parse_args()

    profile_nparticles_memory(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        mem_interval=args.mem_interval,
    )
