"""Profile pipeline execution timing."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from unyt import Msun, Myr

from synthesizer.emission_models import PacmanEmission
from synthesizer.grid import Grid
from synthesizer.particle import Galaxy, Stars


def run_pipeline(nparticles: int, ngalaxies: int, seed: int = 42) -> dict:
    """Run full pipeline and return stage timings.

    Parameters
    ----------
    nparticles : int
        Number of particles per galaxy
    ngalaxies : int
        Number of galaxies
    seed : int
        Random seed for reproducibility

    Returns:
    -------
    dict
        Dictionary with stage names and timings in seconds
    """
    timings = {}

    # Setup - load grid
    t_start = time.perf_counter()
    grid = Grid("test_grid")
    timings["grid_load"] = time.perf_counter() - t_start

    # Build galaxies
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    galaxies = []
    for i in range(ngalaxies):
        initial_masses = rng.uniform(1e4, 1e6, nparticles) * Msun
        ages = rng.uniform(1e6, 1e10, nparticles) * Myr
        metallicities = rng.uniform(0.001, 0.02, nparticles)

        stars = Stars(
            initial_masses=initial_masses,
            ages=ages,
            metallicities=metallicities,
            redshift=0.1,
        )
        gal = Galaxy(stars=stars, redshift=0.1)
        galaxies.append(gal)

    timings["build"] = time.perf_counter() - t_start

    # Create emission model
    t_start = time.perf_counter()
    model = PacmanEmission(
        grid,
        tau_v=0.1,
        fesc=0.0,
        fesc_ly_alpha=1.0,
    )
    timings["model_setup"] = time.perf_counter() - t_start

    # Spectra
    t_start = time.perf_counter()
    for gal in galaxies:
        gal.stars.get_spectra(model)
    timings["spectra"] = time.perf_counter() - t_start

    # Photometry
    t_start = time.perf_counter()
    for gal in galaxies:
        # Placeholder - just aggregate existing spectra
        _ = gal.stars.spectra
    timings["photometry"] = time.perf_counter() - t_start

    # Imaging
    t_start = time.perf_counter()
    for gal in galaxies:
        # Placeholder - just access the built spectra
        _ = gal.stars.spectra
    timings["imaging"] = time.perf_counter() - t_start

    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return timings


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile pipeline execution timing"
    )
    parser.add_argument(
        "--basename", type=str, required=True, help="Basename for output"
    )
    parser.add_argument(
        "--nparticles", type=int, default=1000, help="Number of particles"
    )
    parser.add_argument(
        "--ngalaxies", type=int, default=10, help="Number of galaxies"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path("profiling/outputs/timing") / args.basename
    output_dir.mkdir(parents=True, exist_ok=True)

    particles_str = f"particles={args.nparticles}, galaxies={args.ngalaxies}"
    print(f"Profiling pipeline timing ({particles_str})...")

    # Run pipeline
    timings = run_pipeline(args.nparticles, args.ngalaxies, args.seed)

    # Write CSV
    csv_file = output_dir / "timing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "seconds"])
        for operation, seconds in timings.items():
            writer.writerow([operation, f"{seconds:.6f}"])

    print(f"✓ Timing profile saved: {csv_file}")
    for op, t in timings.items():
        print(f"  {op}: {t:.3f}s")


if __name__ == "__main__":
    main()
