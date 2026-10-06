"""Profile parametric runtime scaling with the number of populations.

This script generates four separate plots (Construction, Spectra, Photometry,
Imaging) showing how parametric operations scale from a single population to
100 populations (e.g. a bulge+disk galaxy up to a galaxy resolved into 100
annuli).
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import kpc

from synthesizer import set_default_out_dtype
from synthesizer.emission_models import IncidentEmission, PacmanEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import Instrument
from synthesizer.parametric import Stars
from synthesizer.utils.operation_timers import OperationTimers

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from parametric_test_data import NPOPS, make_populations
from pipeline_test_data import get_test_instrument

# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 0  # Force no titles

# Set the seed
np.random.seed(42)


def profile_npops(
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

    # --- Setup Instrument and Filters ---
    # Get cached instrument from pipeline_test_data (no network access)
    instrument = get_test_instrument(grid)
    filters_small = instrument.filters.select(
        *instrument.filters.filter_codes[:3]
    )
    filters_large = instrument.filters
    nfilt_small = len(filters_small.filter_codes)
    nfilt_large = len(filters_large.filter_codes)

    # --- Setup Imaging ---
    fov = 30 * kpc
    npix_low = 100
    npix_high = 1000
    inst_low = Instrument(
        "low_res",
        filters=filters_small,
        resolution=fov / npix_low,
    )
    inst_high = Instrument(
        "high_res",
        filters=filters_small,
        resolution=fov / npix_high,
    )

    # Storage for results
    labels = {
        "construction": ["Functions (SFH, ZDist)", "Binned (SFZH array)"],
        "spectra": ["Incident", "Pacman"],
        "photometry": [
            f"Integrated ({nfilt_small} filters)",
            f"Integrated ({nfilt_large} filters)",
        ],
        "imaging": [
            f"Smoothed ({npix_low}x{npix_low})",
            f"Smoothed ({npix_high}x{npix_high})",
        ],
    }
    times = {cat: {lab: [] for lab in labs} for cat, labs in labels.items()}

    for n in NPOPS:
        print(f"Profiling npops={n}...")

        # Local storage for averages
        iter_times = {
            cat: {lab: [] for lab in labs} for cat, labs in labels.items()
        }

        for i in range(n_averages):
            # --- 1. Construction Profiling ---
            # From SFH and ZDist functions
            start = time.perf_counter()
            pops = make_populations(grid, n, seed=i)
            iter_times["construction"]["Functions (SFH, ZDist)"].append(
                time.perf_counter() - start
            )

            # From binned SFZH arrays (the route taken by SAM outputs)
            sfzhs = [p.sfzh for p in pops]
            start = time.perf_counter()
            for sfzh in sfzhs:
                Stars(grid.log10ages, grid.metallicities, sfzh=sfzh)
            iter_times["construction"]["Binned (SFZH array)"].append(
                time.perf_counter() - start
            )

            # --- 2. Spectra Profiling ---
            start = time.perf_counter()
            for p in pops:
                p.get_spectra(model_pacman, nthreads=nthreads)
            iter_times["spectra"]["Pacman"].append(time.perf_counter() - start)

            start = time.perf_counter()
            for p in pops:
                p.get_spectra(model_incident, nthreads=nthreads)
            iter_times["spectra"]["Incident"].append(
                time.perf_counter() - start
            )

            # --- 3. Photometry Profiling ---
            for filters, nfilt in (
                (filters_small, nfilt_small),
                (filters_large, nfilt_large),
            ):
                seds = [p.spectra["int"] for p in pops]
                start = time.perf_counter()
                for sed in seds:
                    sed.get_photo_lnu(filters)
                iter_times["photometry"][
                    f"Integrated ({nfilt} filters)"
                ].append(time.perf_counter() - start)

            # --- 4. Imaging Profiling ---
            # Imaging needs the component photometry for the label
            for p in pops:
                p.get_photo_lnu(filters_small)

            for inst, npix in ((inst_low, npix_low), (inst_high, npix_high)):
                start = time.perf_counter()
                for p in pops:
                    p.get_images_luminosity(
                        "int",
                        fov=fov,
                        instrument=inst,
                        img_type="smoothed",
                        nthreads=nthreads,
                    )
                iter_times["imaging"][f"Smoothed ({npix}x{npix})"].append(
                    time.perf_counter() - start
                )

            # Force garbage collection
            del pops
            gc.collect()

        # Store averages
        for cat in iter_times:
            for label in iter_times[cat]:
                times[cat][label].append(np.mean(iter_times[cat][label]))

    # --- Plotting ---
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_plot(category_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = times[category_name]

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
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3, which="major")
        ax.legend()

        plt.tight_layout()
        filename = (
            f"parametric_npops_performance_{category_name}_"
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

    profile_npops(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        output_dir=args.output_dir,
        grid_precision=args.grid_precision,
    )
