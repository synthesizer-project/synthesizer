"""Analyze memory scaling as a function of particle count.

This script is the scaling-focused companion to ``analyse_memory.py``. It is
intended for runs where each input corresponds to a different particle count,
and therefore requires numeric labels that can be plotted on a logarithmic
x-axis. The script reduces each memory trace to its peak RSS value and then
plots that peak as a function of particle count.

Example:
    Compare peak memory scaling across multiple particle counts::

        python analyse_memory_scaling.py --inputs npart_100/memory.csv \
            npart_1000/memory.csv npart_10000/memory.csv --labels 100 1000 \
            10000 --output-dir memory_scaling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyse_memory import load_memory


def main() -> None:
    """Run the memory scaling analysis workflow.

    This entry point parses the CLI arguments, loads the memory traces,
    extracts peak RSS values, and produces a log-log scaling plot with an
    ``O(n)`` reference line anchored at the first data point.

    Args:
        None

    Returns:
        None
    """
    # Define the command-line interface for the scaling analysis workflow.
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

    # Parse the arguments and ensure the output directory exists.
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure each input file has a matching particle-count label.
    if len(args.inputs) != len(args.labels):
        raise ValueError(
            "analyse_memory_scaling.py requires the same number of "
            "--inputs and --labels entries."
        )

    # Load all memory data and collect peak values for the scaling plot.
    peak_values = []
    for filepath in args.inputs:
        _, mem = load_memory(filepath)
        peak_values.append(max(mem) if mem else 0.0)

    # Create the peak-memory scaling plot against particle count.
    fig, ax = plt.subplots(figsize=(10, 6))
    nparticles = np.array(args.labels)

    # Plot the measured peak memory values for each particle count.
    ax.plot(
        nparticles,
        peak_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="Peak Memory",
    )

    # Add a linear reference curve anchored at the first measured point.
    ax.plot(
        nparticles,
        peak_values[0] * (nparticles / nparticles[0]),
        "k--",
        alpha=0.5,
        linewidth=2,
        label="O(n)",
    )

    # Format the axes for a standard scaling plot presentation.
    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    # Save the finished scaling plot to the requested output directory.
    plot_file = args.output_dir / "memory_comparison_scaling.png"
    fig.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")


if __name__ == "__main__":
    main()
