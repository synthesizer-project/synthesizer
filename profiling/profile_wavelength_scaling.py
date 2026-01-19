"""Profile runtime scaling with the number of wavelength elements.

This script generates a plot showing how spectra generation (Particle vs
Integrated) scales with the number of wavelength elements in the grid.
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
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def profile_wavelength_scaling(nthreads=1, n_averages=3):
    """Run the profiling."""
    print(
        f"Initializing Base Grid (nthreads={nthreads}, "
        f"n_averages={n_averages})..."
    )
    # Load the base grid once to get the range
    base_grid = Grid("test_grid")
    lam_min = base_grid.lam.min()
    lam_max = base_grid.lam.max()

    # Wavelength counts to test (log space)
    # e.g., from 100 to 100,000 points
    n_lambdas = np.logspace(2, 5, 10).astype(int)

    # Fixed number of particles
    n_particles = 10**4

    # Storage for results
    times = {
        "spectra": {
            "Particle": [],
            "Particle (Doppler)": [],
            "Integrated": [],
            "Integrated (Doppler)": [],
        },
    }

    # Standard parametric setup for sampling
    mass = 10**10 * Msun
    param_stars = ParametricStars(
        base_grid.log10ages,
        base_grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=mass,
    )

    # Pre-generate stars (we use the same stars for all grid resolutions)
    # We just need to make sure we clean up the spectra attached to them
    print(f"Generating {n_particles} stars...")
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        n_particles,
        redshift=1,
    )
    stars.velocities = (
        np.random.randn(n_particles, 3) * 100 * (kpc / Myr)
    )  # Needs velocities for shift

    for n_lam in n_lambdas:
        print(f"Profiling n_lam={n_lam}...")

        # 1. Create a new grid with n_lam points
        # We start fresh from the base grid to avoid accumulating interpolation
        # errors (though strictly we are just interpolating from the file each
        # time if we re-load). But Grid object modifies itself in place with
        # interp_spectra. So we should re-load or deep copy.
        grid = Grid("test_grid")

        # Create new wavelength array
        new_lam = np.linspace(lam_min, lam_max, n_lam)

        # Interpolate grid
        grid.interp_spectra(new_lam)

        # 2. Setup Models with this new grid
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

        # 3. Profile

        # Local storage for averages
        iter_times = {
            "Particle": [],
            "Particle (Doppler)": [],
            "Integrated": [],
            "Integrated (Doppler)": [],
        }

        for i in range(n_averages):
            # Clear previous spectra if any
            stars.spectra = {}

            # Particle
            start = time.perf_counter()
            stars.get_spectra(model_part, nthreads=nthreads)
            iter_times["Particle"].append(time.perf_counter() - start)

            # Particle Shift
            start = time.perf_counter()
            stars.get_spectra(model_part_shift, nthreads=nthreads)
            iter_times["Particle (Doppler)"].append(
                time.perf_counter() - start
            )

            # Integrated
            start = time.perf_counter()
            stars.get_spectra(model_int, nthreads=nthreads)
            iter_times["Integrated"].append(time.perf_counter() - start)

            # Integrated Shift
            start = time.perf_counter()
            stars.get_spectra(model_int_shift, nthreads=nthreads)
            iter_times["Integrated (Doppler)"].append(
                time.perf_counter() - start
            )

        # Store averages
        for key in iter_times:
            times["spectra"][key].append(np.mean(iter_times[key]))

        # Force garbage collection
        del grid
        del model_part
        del model_part_shift
        del model_int
        del model_int_shift
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
                n_lambdas,
                values,
                marker=markers[i % len(markers)],
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("Number of Wavelength Elements")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3, which="major")
        ax.legend()
        ax.set_title(
            f"Wavelength Performance (n_particles={n_particles}, "
            f"nthreads={nthreads})"
        )

        plt.tight_layout()
        filename = (
            f"wavelength_performance_{category_name}_"
            f"npart{n_particles}_nt{nthreads}.png"
        )
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    make_plot("spectra", "scaling_wavelength_spectra.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    args = parser.parse_args()

    profile_wavelength_scaling(
        nthreads=args.nthreads, n_averages=args.n_averages
    )
