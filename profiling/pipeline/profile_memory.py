"""Profile pipeline memory usage with continuous sampling."""

from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from unyt import Msun, Myr

from synthesizer.emission_models import PacmanEmission
from synthesizer.grid import Grid
from synthesizer.particle import Galaxy, Stars


def run_pipeline_with_memory(
    nparticles: int, ngalaxies: int, seed: int = 42
) -> tuple[list, float]:
    """Run full pipeline and collect memory samples at 1000 Hz.

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
    tuple[list, float]
        List of (timestamp_ms, rss_mb) tuples and total runtime in seconds
    """
    samples = []
    stop_sampling = False

    def sample_memory():
        """Background thread to sample memory."""
        timestamp_ms = 0
        while not stop_sampling:
            rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
            samples.append((timestamp_ms, rss_mb))
            timestamp_ms += 1
            time.sleep(0.001)  # 1000 Hz

    # Start sampling thread
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    start_time = time.perf_counter()

    # Setup
    rng = np.random.default_rng(seed)
    grid = Grid("test_grid")

    # Create emission model
    model = PacmanEmission(
        grid,
        tau_v=0.1,
        fesc=0.0,
        fesc_ly_alpha=1.0,
    )

    # Build galaxies
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

    # Spectra
    for gal in galaxies:
        gal.stars.get_spectra(model)

    # Photometry
    for gal in galaxies:
        # Placeholder - just aggregate existing spectra
        _ = gal.stars.spectra

    # Imaging
    for gal in galaxies:
        # Placeholder - just access the built spectra
        _ = gal.stars.spectra

    total_time = time.perf_counter() - start_time
    stop_sampling = True
    sampler.join()

    return samples, total_time


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile pipeline memory usage (1000 Hz sampling)"
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
    output_dir = Path("profiling/outputs/memory") / args.basename
    output_dir.mkdir(parents=True, exist_ok=True)

    particles_str = f"particles={args.nparticles}, galaxies={args.ngalaxies}"
    print(f"Profiling pipeline memory ({particles_str})...")

    # Run pipeline with memory sampling
    samples, total_time = run_pipeline_with_memory(
        args.nparticles, args.ngalaxies, args.seed
    )

    # Write CSV with all samples
    csv_file = output_dir / "memory.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "rss_mb"])
        for timestamp_ms, rss_mb in samples:
            writer.writerow([timestamp_ms, f"{rss_mb:.2f}"])

    # Print summary
    peak_mb = max(s[1] for s in samples) if samples else 0
    mean_mb = sum(s[1] for s in samples) / len(samples) if samples else 0
    min_mb = min(s[1] for s in samples) if samples else 0

    print(f"✓ Memory profile saved: {csv_file}")
    print(f"  Samples: {len(samples)}")
    print(f"  Peak: {peak_mb:.2f} MB")
    print(f"  Mean: {mean_mb:.2f} MB")
    print(f"  Min: {min_mb:.2f} MB")
    print(f"  Duration: {total_time:.3f}s")


if __name__ == "__main__":
    main()
