"""Profile executing emission models concurrently (nr_model_threads).

This builds a PacmanEmission carrying tau_v x fesc parameter variations, which
expands into a large model tree, and times get_spectra for every combination
of model threads and OpenMP threads requested. Each run is checked against the
serial result so any threading bug shows up as a mismatch.

Model threads are only used on a free-threaded build of Python with the GIL
disabled. At the time of writing astropy re-enables the GIL on import, so run
with PYTHON_GIL=0:

    PYTHON_GIL=0 python3.14t profiling/scaling/profile_model_threads.py \
        --ntau 40 --nfesc 10 --model_threads 1,2,4,8 --nthreads 1,2

Usage:
    python profile_model_threads.py --basename test --ntau 20 --nfesc 5
        --per_particle --nstars 1000
"""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr

from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.model_queue import gil_enabled
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def _peak_rss_gb():
    """Return the peak resident set size of this process in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is in bytes on macOS and kilobytes on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def _time_spectra(
    stars,
    model,
    nr_model_threads,
    nthreads,
    average_over,
    reference=None,
):
    """Time one configuration and compare it to the reference spectra.

    Returns the best wall time and, if a reference is given, whether the
    spectra match it, otherwise the spectra themselves (to use as the
    reference). The spectra are cleared before returning so that per-particle
    runs don't hold several sets of outputs in memory at once.
    """
    best = np.inf
    for _ in range(average_over):
        stars.clear_all_emissions()
        start = time.perf_counter()
        stars.get_spectra(
            model,
            nthreads=nthreads,
            nr_model_threads=nr_model_threads,
        )
        best = min(best, time.perf_counter() - start)

    emissions = stars.particle_spectra if model.per_particle else stars.spectra
    spectra = {label: sed.lnu.value for label, sed in emissions.items()}
    stars.clear_all_emissions()

    if reference is None:
        return best, spectra

    match = spectra.keys() == reference.keys() and all(
        np.array_equal(spectra[k], reference[k], equal_nan=True)
        for k in reference
    )
    return best, match


def profile_model_threads(
    basename,
    out_dir,
    ntau,
    nfesc,
    per_particle,
    nstars,
    model_threads,
    nthreads,
    average_over,
):
    """Profile get_spectra over model and OpenMP thread counts."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grid = Grid("test_grid")

    # Build and expand the varied model
    model = PacmanEmission(
        grid,
        tau_v=ParameterList(
            list(np.linspace(0.05, 2.0, ntau)),
            label_modifier="tauv%.3f",
        ),
        fesc=ParameterList(
            list(np.linspace(0.0, 0.9, nfesc)),
            label_modifier="fesc%.3f",
        ),
        per_particle=per_particle,
    ).expand_models()

    # Sample the SFZH, producing a Stars object
    param_stars = ParametricStars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=10**10 * Msun,
    )
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        redshift=1,
    )

    print(
        f"Python {sys.version.split()[0]}, GIL enabled: {gil_enabled()}, "
        f"models: {len(model._models)}, nstars: {nstars}, "
        f"per_particle: {per_particle}"
    )
    if gil_enabled() and max(model_threads) > 1:
        print(
            "WARNING: the GIL is enabled so every run will execute the "
            "models serially."
        )

    # Warm up (and get the serial reference)
    _time_spectra(stars, model, 1, 1, 1)
    serial_time, reference = _time_spectra(stars, model, 1, 1, average_over)

    rows = []
    for nt in nthreads:
        for mt in model_threads:
            if mt == 1 and nt == 1:
                wall, match = serial_time, True
            else:
                wall, match = _time_spectra(
                    stars,
                    model,
                    mt,
                    nt,
                    average_over,
                    reference=reference,
                )
            rows.append(
                {
                    "nr_model_threads": mt,
                    "nthreads": nt,
                    "wall_s": wall,
                    "speedup": serial_time / wall,
                    "match": match,
                }
            )
            print(
                f"  nr_model_threads={mt:<3d} nthreads={nt:<3d} "
                f"{wall:8.3f}s  speedup={serial_time / wall:5.2f}x  "
                f"match={match}"
            )

    print(f"Peak RSS: {_peak_rss_gb():.2f} GB")

    # Write the results
    kind = "part" if per_particle else "int"
    stem = (
        f"{out_dir}/{basename}_model_threads_{kind}_"
        f"models{len(model._models)}_nstars{nstars}"
    )
    with open(f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot speedup against model threads for each OpenMP thread count
    fig, ax = plt.subplots()
    for nt in nthreads:
        these = [r for r in rows if r["nthreads"] == nt]
        ax.plot(
            [r["nr_model_threads"] for r in these],
            [r["speedup"] for r in these],
            marker="o",
            label=f"nthreads={nt}",
        )
    ax.plot(model_threads, model_threads, "k--", alpha=0.5, label="Ideal")
    ax.set_xlabel("nr_model_threads")
    ax.set_ylabel("Speedup over serial")
    ax.set_title(
        f"{len(model._models)} models, {nstars} stars "
        f"({'per-particle' if per_particle else 'integrated'})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {stem}.csv and {stem}.png")


def _int_list(value):
    """Parse a comma separated list of integers."""
    return [int(v) for v in value.split(",")]


if __name__ == "__main__":
    # Get the command line args
    args = argparse.ArgumentParser()

    args.add_argument(
        "--basename",
        type=str,
        default="test",
        help="The basename of the output files.",
    )

    args.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="The output directory for the csv and plot files."
        " Defaults to the current directory.",
    )

    args.add_argument(
        "--ntau",
        type=int,
        default=40,
        help="The number of tau_v values to vary over.",
    )

    args.add_argument(
        "--nfesc",
        type=int,
        default=10,
        help="The number of fesc values to vary over.",
    )

    args.add_argument(
        "--per_particle",
        action="store_true",
        help="Generate per-particle spectra rather than integrated spectra.",
    )

    args.add_argument(
        "--nstars",
        type=int,
        default=1000,
        help="The number of stars to use.",
    )

    args.add_argument(
        "--model_threads",
        type=_int_list,
        default=[1, 2, 4, 8],
        help="Comma separated model thread counts to test.",
    )

    args.add_argument(
        "--nthreads",
        type=_int_list,
        default=[1],
        help="Comma separated OpenMP thread counts to test.",
    )

    args.add_argument(
        "--average_over",
        type=int,
        default=3,
        help="The number of repeats (the best time is reported).",
    )

    args = args.parse_args()

    profile_model_threads(
        args.basename,
        args.out_dir,
        args.ntau,
        args.nfesc,
        args.per_particle,
        args.nstars,
        args.model_threads,
        args.nthreads,
        args.average_over,
    )
