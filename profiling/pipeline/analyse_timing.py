"""Analyze and compare timing profiles from multiple runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_timing(filepath: Path) -> dict:
    """Loads timing data from a CSV file.

    Args:
        filepath (Path): The path to the timing CSV file.

    Returns:
        dict: A dictionary where keys are operation names and values are dicts
            containing 'time' (float) and 'source' (str).
    """
    timings = {}
    with open(filepath) as f:
        lines = f.readlines()
        if not lines:
            return timings

        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(",")
            # Format: operation,seconds,count,source
            operation, seconds, _, source = (
                parts[0],
                float(parts[1]),
                int(parts[2]),  # count (unused here)
                parts[3],
            )
            timings[operation] = {"time": seconds, "source": source}
    return timings


def main() -> None:
    """Main entry point for the timing analysis script."""
    parser = argparse.ArgumentParser(
        description="Compare timing results from multiple profiling runs"
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
        type=str,
        help="Labels for each run (default: use filenames)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Output directory for plots (default: current directory)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all timing data
    timing_data = {}
    for i, filepath in enumerate(args.inputs):
        label = (
            args.labels[i]
            if args.labels and i < len(args.labels)
            else filepath.stem
        )
        timing_data[label] = load_timing(filepath)

    print(f"Loaded {len(timing_data)} timing profiles")

    # Get operations (from first profile)
    operations = list(next(iter(timing_data.values())).keys())

    # Filter operations: keep only those that contribute >=5% to one run
    filtered_ops = []
    labels = list(timing_data.keys())
    for op in operations:
        for label in labels:
            op_data = timing_data[label].get(op, {})
            op_time = op_data.get("time", 0)
            # Use the label's total time (sum of its operations)
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

    # Convert labels to integers (particle counts)
    nparticles = [int(label) for label in labels]

    # Create line plot showing scaling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group operations by source for legend
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

    # Plot C++ operations (solid lines)
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

    # Plot Python operations (dashed lines)
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

    # Plot total time
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

    # Add reference scaling line (O(n)) anchored at first data point
    npart_ref = np.array(nparticles)
    ax.plot(
        npart_ref,
        total_values[0] * (npart_ref / nparticles[0]),
        "k:",
        alpha=0.5,
        linewidth=1.5,
        label="O(n)",
    )

    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Create main legend for operations
    legend1 = ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.9)

    # Add second legend for line styles (C vs Python)
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
    ax.add_artist(legend1)  # Keep first legend
    ax.legend(
        handles=style_handles, loc="lower right", fontsize=9, framealpha=0.9
    )

    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    plot_file = args.output_dir / "timing_comparison.png"
    fig.savefig(plot_file, dpi=200)
    print(f"✓ Saved: {plot_file}")

    # Create summary text
    summary_file = args.output_dir / "timing_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Timing Summary\n")
        f.write("=" * 60 + "\n\n")

        for label in labels:
            f.write(f"{label}:\n")
            for op in operations:
                val = timing_data[label].get(op, {}).get("time", 0)
                src = timing_data[label].get(op, {}).get("source", "?")
                f.write(f"  {op:40s}: {val:8.3f}s [{src}]\n")
            f.write("\n")

    print(f"✓ Saved: {summary_file}")


if __name__ == "__main__":
    main()
