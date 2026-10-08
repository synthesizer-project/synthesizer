"""Profile parametric memory scaling with the number of populations.

This script generates two separate plots (Spectra, Photometry) showing how
the size of the result objects scales from a single population to 100
populations, both with a separate Stars per population and with a single
Stars holding every population.
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthesizer import set_default_out_dtype
from synthesizer.emission_models import IncidentEmission, PacmanEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.utils.operation_timers import OperationTimers

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from parametric_test_data import (
    NPOPS,
    combine_populations,
    get_obj_size_manual,
    make_populations,
)
from pipeline_test_data import get_test_instrument

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def measure_pops(pops, func, attr):
    """Run func on every population and return the result size in GB.

    Args:
        pops (list of Stars):
            The populations.
        func (callable):
            Called with each population, populating attr on it.
        attr (callable):
            Returns the object to measure from each population.
    """
    size = 0
    for p in pops:
        func(p)
        size += get_obj_size_manual(attr(p))
    return size / 1024 / 1024 / 1024  # Convert Bytes to GB


def profile_npops_memory(
    nthreads=1,
    n_averages=3,
    output_dir=Path("profiling/plots"),
    grid_precision="float64",
):
    """Run the profiling."""
    print(
        f"Initializing Grid and Models (nthreads={nthreads}, "
        f"n_averages={n_averages})..."
    )
    timers = OperationTimers()
    timers.reset()

    grid = Grid("test_grid", use_precision=np.dtype(grid_precision))
    n_lam = grid.nlam

    # --- Setup Models ---
    model_incident = IncidentEmission(grid, label="int")
    model_pacman = PacmanEmission(
        grid,
        tau_v=0.3,
        dust_curve=Calzetti2000(),
        fesc=0.1,
        fesc_ly_alpha=0.1,
    )
    model_per_pop = IncidentEmission(grid, label="int", per_particle=True)

    # --- Setup Filters ---
    # Get cached instrument from pipeline_test_data (no network access)
    instrument = get_test_instrument(grid)
    filters_small = instrument.filters.select(
        *instrument.filters.filter_codes[:3]
    )
    filters_large = instrument.filters
    nfilt_small = len(filters_small.filter_codes)
    nfilt_large = len(filters_large.filter_codes)

    # Storage for results
    labels = {
        "spectra": [
            "Incident, separate",
            "Incident, combined",
            "Pacman, separate",
            "Pacman, combined",
            "Incident per population, combined",
        ],
        "photometry": [
            f"Separate ({nfilt_small} filters)",
            f"Separate ({nfilt_large} filters)",
            f"Per population, combined ({nfilt_small} filters)",
            f"Per population, combined ({nfilt_large} filters)",
        ],
    }
    mems = {cat: {lab: [] for lab in labs} for cat, labs in labels.items()}

    for n in NPOPS:
        print(f"Profiling Memory npops={n}...")

        # Local storage for averages
        iter_mems = {
            cat: {lab: [] for lab in labs} for cat, labs in labels.items()
        }

        for i in range(n_averages):
            pops = make_populations(grid, n, seed=i)
            combined = combine_populations(pops)

            # --- 1. Spectra Profiling ---
            for name, model in (
                ("Incident", model_incident),
                ("Pacman", model_pacman),
            ):
                for p in pops:
                    p.spectra = {}
                iter_mems["spectra"][f"{name}, separate"].append(
                    measure_pops(
                        pops,
                        lambda p, m=model: p.get_spectra(m, nthreads=nthreads),
                        lambda p: p.spectra,
                    )
                )
                combined.spectra = {}
                iter_mems["spectra"][f"{name}, combined"].append(
                    measure_pops(
                        [combined],
                        lambda p, m=model: p.get_spectra(m, nthreads=nthreads),
                        lambda p: p.spectra,
                    )
                )

            # The emission of each population (kept like per particle
            # emission, alongside the integrated emission)
            for p in pops:
                p.spectra = {}
            combined.spectra = {}
            for p in pops:
                p.get_spectra(model_incident, nthreads=nthreads)
            iter_mems["spectra"]["Incident per population, combined"].append(
                measure_pops(
                    [combined],
                    lambda p: p.get_spectra(model_per_pop, nthreads=nthreads),
                    lambda p: (p.spectra, p.particle_spectra),
                )
            )

            # --- 2. Photometry Profiling ---
            for filters, nfilt in (
                (filters_small, nfilt_small),
                (filters_large, nfilt_large),
            ):
                for p in pops:
                    p.spectra["int"].photo_lnu = {}
                iter_mems["photometry"][f"Separate ({nfilt} filters)"].append(
                    measure_pops(
                        pops,
                        lambda p, f=filters: p.spectra["int"].get_photo_lnu(f),
                        lambda p: p.spectra["int"].photo_lnu,
                    )
                )
                sed = combined.particle_spectra["int"]
                sed.photo_lnu = {}
                iter_mems["photometry"][
                    f"Per population, combined ({nfilt} filters)"
                ].append(
                    measure_pops(
                        [combined],
                        lambda p, f=filters: sed.get_photo_lnu(f),
                        lambda p: sed.photo_lnu,
                    )
                )

            # Force garbage collection
            del pops, combined
            gc.collect()

        # Store averages
        for cat in iter_mems:
            for label in iter_mems[cat]:
                mems[cat][label].append(np.mean(iter_mems[cat][label]))

    # --- Plotting ---
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_plot(category_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = mems[category_name]

        # Style cycle
        markers = ["o", "s", "d", "v", "^", "<", ">"]

        for i, (label, values) in enumerate(data.items()):
            ax.loglog(
                NPOPS,
                values,
                marker=markers[i % len(markers)],
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("Number of Populations")
        ax.set_ylabel("Result Object Size (GB)")
        ax.grid(True, alpha=0.3, which="major")
        ax.legend()

        plt.tight_layout()
        filename = (
            f"parametric_npops_performance_memory_{category_name}_"
            f"nlam{n_lam}_nt{nthreads}.png"
        )
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    for category_name in labels:
        make_plot(category_name)
    print("Operation timing table:")
    OperationTimers.print_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("profiling/plots"),
    )
    parser.add_argument(
        "--grid-precision",
        choices=("float32", "float64"),
        default="float64",
        help="Precision to load the grid arrays at.",
    )
    parser.add_argument(
        "--out-dtype",
        choices=("float32", "float64"),
        default=None,
        help="Requested output precision. Defaults to the global default.",
    )
    args = parser.parse_args()

    # Set the global output precision if one was requested
    if args.out_dtype is not None:
        set_default_out_dtype(np.dtype(args.out_dtype))

    profile_npops_memory(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        output_dir=args.output_dir,
        grid_precision=args.grid_precision,
    )
