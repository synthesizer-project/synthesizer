"""Analyze memory scaling as a function of particle count."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyse_memory import load_memory


def main() -> None:
    """Main entry point for the memory scaling analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyse memory scaling from multiple profiling runs"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Memory CSV files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        required=True,
        help="Particle-count labels for each run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Output directory for plots (default: current directory)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all memory data and collect peak values for the scaling plot.
    peak_values = []
    for filepath in args.inputs:
        _, mem = load_memory(filepath)
        peak_values.append(max(mem) if mem else 0.0)

    # Create the peak-memory scaling plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    nparticles = np.array(args.labels)

    ax.plot(
        nparticles,
        peak_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="Peak Memory",
    )
    ax.plot(
        nparticles,
        peak_values[0] * (nparticles / nparticles[0]),
        "k--",
        alpha=0.5,
        linewidth=2,
        label="O(n)",
    )

    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    plot_file = args.output_dir / "memory_comparison_scaling.png"
    fig.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")


if __name__ == "__main__":
    main()
