"""Profile thread scaling for various operations.

This script generates scaling plots showing performance from 1 to N threads
for integrated spectra, particle spectra, and smoothed imaging.
"""

import argparse
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
    run_scaling_test,
)

# Set style
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def profile_threads(max_threads=8, nstars=10**5, average_over=3):
    """Run the thread scaling profiling."""
    print(f"Initializing Grid and Models (nstars={nstars})...")
    grid = Grid("test_grid")

    # Models
    model_part = IncidentEmission(grid, per_particle=True, label="particle")
    model_int = IncidentEmission(grid, per_particle=False, label="integrated")

    # Kernel for imaging
    kernel = Kernel().get_kernel()

    # --- Setup Filters ---
    filters = FilterCollection(
        filter_codes=[
            "JWST/NIRCam.F150W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F444W",
        ],
        new_lam=grid.lam,
    )

    # Create instrument for imaging
    inst_dir = Path("profiling/instruments")
    inst_dir.mkdir(parents=True, exist_ok=True)

    fov = 30 * kpc
    npix = 1000
    res = fov / npix
    inst = get_instrument_profile(
        label="test",
        filepath=str(inst_dir / "thread_test.hdf5"),
        filters=filters,
        resolution=res,
    )

    # Standard parametric setup for sampling
    mass = 10**10 * Msun
    param_stars = ParametricStars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=mass,
    )

    # Sample stars
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        redshift=1,
    )
    # Setup for imaging
    stars.coordinates = np.random.randn(nstars, 3) * kpc
    stars.centre = np.array([0, 0, 0]) * kpc
    stars.calculate_smoothing_lengths(num_neighbours=50)

    # Generate spectra and photometry for imaging
    # (imaging requires photometry to be present for the labels)
    stars.get_spectra(model_part)
    stars.get_particle_photo_lnu(filters)

    output_dir = Path("profiling/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Integrated Spectra Scaling
    print("\nProfiling Thread Scaling: Integrated Spectra...")
    plot_path = output_dir / "integrated_performance_threads.png"
    log_path = plot_path.with_suffix(".log")
    run_scaling_test(
        max_threads,
        average_over,
        str(log_path),
        str(plot_path),
        stars.get_spectra,
        {"emission_model": model_int},
        total_msg="Integrated Spectra",
        low_thresh=0.1,
        paper_style=True,
    )

    # 2. Particle Spectra Scaling
    print("\nProfiling Thread Scaling: Particle Spectra...")
    plot_path = output_dir / "particle_performance_threads.png"
    log_path = plot_path.with_suffix(".log")
    run_scaling_test(
        max_threads,
        average_over,
        str(log_path),
        str(plot_path),
        stars.get_spectra,
        {"emission_model": model_part},
        total_msg="Particle Spectra",
        low_thresh=0.1,
        paper_style=True,
    )

    # 3. Smoothed Imaging Scaling
    print("\nProfiling Thread Scaling: Smoothed Imaging...")
    plot_path = output_dir / "particle_imaging_performance_threads.png"
    log_path = plot_path.with_suffix(".log")

    def get_images_wrapper(nthreads, **kwargs):
        model_label = kwargs.pop("model_label")
        stars.get_images_luminosity(model_label, nthreads=nthreads, **kwargs)

    run_scaling_test(
        max_threads,
        average_over,
        str(log_path),
        str(plot_path),
        get_images_wrapper,
        {
            "model_label": "particle",
            "instrument": inst,
            "fov": 30 * kpc,
            "kernel": kernel,
        },
        total_msg="Smoothed Imaging",
        low_thresh=0.1,
        paper_style=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_threads", type=int, default=8)
    parser.add_argument("--nstars", type=int, default=10**5)
    parser.add_argument("--average_over", type=int, default=3)
    args = parser.parse_args()

    profile_threads(args.max_threads, args.nstars, args.average_over)
