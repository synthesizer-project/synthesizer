"""Analyze timing scaling as a function of particle count.

This script is the scaling-focused companion to ``analyse_timing.py``. It is
intended for profiling suites where each timing CSV corresponds to a different
particle count, and therefore requires numeric labels that can be plotted on a
logarithmic x-axis. The script filters the operation list down to timings that
make a meaningful contribution in at least one run and then compares how those
costs scale with particle count.

Example:
    Compare timing scaling across multiple particle counts::

        python analyse_timing_scaling.py --inputs npart_100/timing.csv \
            npart_1000/timing.csv npart_10000/timing.csv --labels 100 1000 \
            10000 --output-dir timing_scaling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyse_timing import load_timing


def main() -> None:
    """Run the timing scaling analysis workflow.

    This entry point parses the CLI arguments, loads the timing CSV files,
    filters low-contribution operations, and produces a log-log scaling plot
    comparing both per-operation and total runtime trends.

    Args:
        None

    Returns:
        None
    """
    # Define the command-line interface for the scaling analysis workflow.
    parser = argparse.ArgumentParser(
        description="Analyse timing scaling from multiple profiling runs"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Timing CSV files to compare",
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
            "analyse_timing_scaling.py requires the same number of "
            "--inputs and --labels entries."
        )

    # Load all timing data keyed by numeric particle count labels.
    timing_data = {
        str(label): load_timing(filepath)
        for label, filepath in zip(args.labels, args.inputs)
    }
    labels = [str(label) for label in args.labels]
    operations = list(next(iter(timing_data.values())).keys())

    # Filter operations that contribute at least 5% in one run so the scaling
    # plot stays readable and focuses on meaningful contributors.
    filtered_ops = []
    for op in operations:
        for label in labels:
            op_time = timing_data[label].get(op, {}).get("time", 0)
            label_total = sum(
                timing_data[label].get(o, {}).get("time", 0)
                for o in operations
            )
            contribution = (
                (op_time / label_total * 100) if label_total > 0 else 0
            )
            if contribution >= 5.0:
                filtered_ops.append(op)
                break

    # Create the scaling plot with particle count on the x-axis.
    fig, ax = plt.subplots(figsize=(12, 8))
    nparticles = np.array(args.labels)

    # Split operations by source so the line style can communicate whether the
    # timing originated from Python or a C extension.
    c_ops = [
        op
        for op in filtered_ops
        if timing_data[labels[0]].get(op, {}).get("source") == "C"
    ]
    py_ops = [
        op
        for op in filtered_ops
        if timing_data[labels[0]].get(op, {}).get("source") != "C"
    ]

    # Plot the retained C-extension operations using solid lines.
    colors_c = plt.cm.Blues(np.linspace(0.4, 0.8, len(c_ops) + 1))
    for i, op in enumerate(c_ops):
        values = [
            timing_data[label].get(op, {}).get("time", 0) for label in labels
        ]
        ax.plot(
            nparticles,
            values,
            marker="o",
            linewidth=2,
            label=op,
            color=colors_c[i],
            markersize=6,
            linestyle="-",
        )

    # Plot the retained Python operations using dashed lines.
    colors_py = plt.cm.Oranges(np.linspace(0.4, 0.8, len(py_ops) + 1))
    for i, op in enumerate(py_ops):
        values = [
            timing_data[label].get(op, {}).get("time", 0) for label in labels
        ]
        ax.plot(
            nparticles,
            values,
            marker="s",
            linewidth=2,
            label=op,
            color=colors_py[i],
            markersize=6,
            linestyle="--",
        )

    # Plot the total runtime across all operations for each particle count.
    total_values = [
        sum(timing_data[label].get(op, {}).get("time", 0) for op in operations)
        for label in labels
    ]
    ax.plot(
        nparticles,
        total_values,
        marker="D",
        linewidth=3,
        label="Total",
        color="black",
        markersize=8,
        linestyle="-",
        alpha=0.7,
    )

    # Add a linear reference trend anchored at the first measured total time.
    ax.plot(
        nparticles,
        total_values[0] * (nparticles / nparticles[0]),
        "k:",
        alpha=0.5,
        linewidth=1.5,
        label="O(n)",
    )

    # Format the axes for a standard scaling plot presentation.
    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Draw the main legend listing all plotted operations.
    legend1 = ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.9)

    # Build a second legend describing the line-style convention.
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            marker="o",
            markersize=6,
            label="C Extension",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="s",
            markersize=6,
            label="Python",
        ),
    ]
    ax.add_artist(legend1)
    ax.legend(
        handles=style_handles, loc="lower right", fontsize=9, framealpha=0.9
    )

    # Finalise the figure layout before saving it to disk.
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    # Save the finished scaling plot to the requested output directory.
    plot_file = args.output_dir / "timing_comparison.png"
    fig.savefig(plot_file, dpi=200)
    print(f"✓ Saved: {plot_file}")


if __name__ == "__main__":
    main()
