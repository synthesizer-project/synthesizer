"""Profile parametric runtime scaling with the number of wavelength elements.

This script generates a plot showing how parametric spectra generation
(Incident vs Pacman) scales with the number of wavelength elements in the
grid for a fixed number of populations.
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parametric_test_data import make_populations

from synthesizer import set_default_out_dtype
from synthesizer.emission_models import IncidentEmission, PacmanEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.utils.operation_timers import OperationTimers

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def profile_wavelength_scaling(
    nthreads=1,
    n_averages=3,
    output_dir=Path("profiling/plots"),
    grid_precision="float64",
):
    """Run the profiling."""
    print(
        f"Initializing Base Grid (nthreads={nthreads}, "
        f"n_averages={n_averages})..."
    )
    timers = OperationTimers()
    timers.reset()

    # Load the base grid once to get the range
    base_grid = Grid("test_grid", use_precision=np.dtype(grid_precision))
    lam_min = base_grid.lam.min()
    lam_max = base_grid.lam.max()

    # Wavelength counts to test (log space)
    n_lambdas = np.logspace(2, 5, 10).astype(int)

    # Fixed number of populations
    n_pops = 10

    # Storage for results
    times = {
        "spectra": {
            "Incident": [],
            "Pacman": [],
        },
    }

    # Pre-generate the populations (the SFZHs only depend on the grid axes
    # so the same populations are used for all wavelength resolutions)
    print(f"Generating {n_pops} populations...")
    pops = make_populations(base_grid, n_pops)

    for n_lam in n_lambdas:
        print(f"Profiling n_lam={n_lam}...")

        # 1. Create a new grid with n_lam points
        grid = Grid("test_grid", use_precision=np.dtype(grid_precision))
        new_lam = np.linspace(lam_min, lam_max, n_lam)
        grid.interp_spectra(new_lam)

        # 2. Setup Models with this new grid
        model_incident = IncidentEmission(grid, label="int")
        model_pacman = PacmanEmission(
            grid,
            tau_v=0.3,
            dust_curve=Calzetti2000(),
            fesc=0.1,
            fesc_ly_alpha=0.1,
        )

        # 3. Profile

        # Local storage for averages
        iter_times = {
            "Incident": [],
            "Pacman": [],
        }

        for i in range(n_averages):
            # Clear previous spectra
            for p in pops:
                p.spectra = {}

            start = time.perf_counter()
            for p in pops:
                p.get_spectra(model_incident, nthreads=nthreads)
            iter_times["Incident"].append(time.perf_counter() - start)

            start = time.perf_counter()
            for p in pops:
                p.get_spectra(model_pacman, nthreads=nthreads)
            iter_times["Pacman"].append(time.perf_counter() - start)

        # Store averages
        for key in iter_times:
            times["spectra"][key].append(np.mean(iter_times[key]))

        # Force garbage collection
        del grid
        del model_incident
        del model_pacman
        gc.collect()

    # --- Plotting ---
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_plot(category_name):
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

        plt.tight_layout()
        filename = (
            f"parametric_wavelength_performance_{category_name}_"
            f"npops{n_pops}_nt{nthreads}.png"
        )
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.close()

    make_plot("spectra")
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

    profile_wavelength_scaling(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        output_dir=args.output_dir,
        grid_precision=args.grid_precision,
    )
