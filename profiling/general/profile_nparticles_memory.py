"""Profile memory scaling with the number of particles.

This script generates two separate plots (Spectra, Photometry)
showing how the peak memory usage of various operations scales from 10^3 to
10^5 particles.
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, kpc, unyt_array

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from pipeline_test_data import get_test_instrument

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def get_obj_size_manual(obj):
    """Estimate object size in bytes, handling nested dicts and numpy arrays.

    This avoids bugs in pympler.asizeof with certain numpy versions.
    """
    size = 0
    if isinstance(obj, dict):
        for v in obj.values():
            size += get_obj_size_manual(v)
    elif isinstance(obj, (np.ndarray, unyt_array)):
        size += obj.nbytes
    elif hasattr(obj, "__dict__"):
        # For simple objects, try to sum their attributes
        for v in vars(obj).values():
            size += get_obj_size_manual(v)
    else:
        # Fallback for small objects
        import sys

        size += sys.getsizeof(obj)
    return size


def run_and_measure_memory(func, *args, obj_to_measure=None, **kwargs):
    """Run a function and return the size of the result object in GB.

    Uses a manual size estimation targeting large arrays.
    """
    # Run the function
    func(*args, **kwargs)

    # Measure the size of the specified object
    if obj_to_measure is not None:
        size = get_obj_size_manual(obj_to_measure)
    else:
        return 0.0

    return size / 1024 / 1024 / 1024  # Convert Bytes to GB


def profile_nparticles_memory(nthreads=1, n_averages=3):
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
    # Get cached instrument from pipeline_test_data (no network access)
    # JWSTNIRCamWide has 8 filters total
    instrument = get_test_instrument(grid)

    # Extract the FilterCollection from the instrument
    # Small set (3 filters) - select first 3 filters
    filter_codes_3 = instrument.available_filters[:3]
    filters_3 = instrument.filters.select(*filter_codes_3)

    # Large set (8 filters) - use all available filters
    filters_10 = instrument.filters.select(*instrument.available_filters)

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
            stars.particle_spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_part,
                obj_to_measure=stars.particle_spectra,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Particle"].append(mem)
            del stars.particle_spectra["part"]

            # Particle Shift
            stars.spectra = {}
            stars.particle_spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_part_shift,
                obj_to_measure=stars.particle_spectra,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Particle (Doppler)"].append(mem)
            del stars.particle_spectra["part_shift"]

            # Integrated
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_int,
                obj_to_measure=stars.spectra,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Integrated"].append(mem)
            del stars.spectra["int"]

            # Integrated Shift
            stars.spectra = {}
            mem = run_and_measure_memory(
                stars.get_spectra,
                model_int_shift,
                obj_to_measure=stars.spectra,
                nthreads=nthreads,
            )
            iter_mems["spectra"]["Integrated (Doppler)"].append(mem)

            # Re-generate necessary spectra for photometry
            stars.get_spectra(model_part, nthreads=nthreads)
            stars.get_spectra(model_int, nthreads=nthreads)

            # Particle (3 filters)
            mem = run_and_measure_memory(
                stars.get_particle_photo_lnu,
                filters_3,
                obj_to_measure=stars.particle_photo_lnu,
            )
            iter_mems["photometry"]["Particle (3 filters)"].append(mem)
            # Clear to isolate next measurement
            stars.particle_photo_lnu = {}

            # Particle (10 filters)
            mem = run_and_measure_memory(
                stars.get_particle_photo_lnu,
                filters_10,
                obj_to_measure=stars.particle_photo_lnu,
            )
            iter_mems["photometry"]["Particle (10 filters)"].append(mem)

            # Integrated Photometry
            sed_int = stars.spectra["int"]

            mem = run_and_measure_memory(
                sed_int.get_photo_lnu,
                filters_3,
                obj_to_measure=sed_int.photo_lnu,
            )
            iter_mems["photometry"]["Integrated (3 filters)"].append(mem)
            sed_int.photo_lnu = {}

            mem = run_and_measure_memory(
                sed_int.get_photo_lnu,
                filters_10,
                obj_to_measure=sed_int.photo_lnu,
            )
            iter_mems["photometry"]["Integrated (10 filters)"].append(mem)

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

    def make_plot(category_name):
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
        ax.set_ylabel("Result Object Size (GB)")
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

    make_plot("spectra")
    make_plot("photometry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    args = parser.parse_args()

    profile_nparticles_memory(
        nthreads=args.nthreads, n_averages=args.n_averages
    )
