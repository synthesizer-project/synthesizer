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
        dict: A dictionary where keys are operation names and values are their
              execution times in seconds.
    """
    timings = {}
    with open(filepath) as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:
                operation, seconds = parts
                timings[operation] = float(seconds)
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
        default=Path("profiling/outputs/timing_analysis"),
        help="Output directory for plots",
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
    operations = [op for op in operations if op != "total"]

    # Filter operations: keep only those that contribute >=5% to at
    # least one run
    filtered_ops = []
    labels = list(timing_data.keys())
    for op in operations:
        for label in labels:
            total_time = timing_data[label].get("total", 1)
            op_time = timing_data[label].get(op, 0)
            contribution = (
                (op_time / total_time * 100) if total_time > 0 else 0
            )
            if contribution >= 5.0:
                filtered_ops.append(op)
                break

    # Convert labels to integers (particle counts)
    nparticles = [int(label) for label in labels]

    # Create line plot showing scaling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot filtered operations
    colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_ops) + 1))
    for i, op in enumerate(filtered_ops):
        values = [timing_data[label].get(op, 0) for label in labels]
        display_name = op.replace("op_", "") if op.startswith("op_") else op
        ax.plot(
            nparticles,
            values,
            marker="o",
            linewidth=2,
            label=display_name,
            color=colors[i],
            markersize=6,
        )

    # Plot total time (dashed, thicker)
    total_values = [timing_data[label].get("total", 0) for label in labels]
    ax.plot(
        nparticles,
        total_values,
        marker="s",
        linewidth=3,
        linestyle="--",
        label="Total",
        color="black",
        markersize=8,
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
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()

    plot_file = args.output_dir / "timing_comparison.png"
    fig.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")

    # Create summary text
    summary_file = args.output_dir / "timing_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Timing Summary\n")
        f.write("=" * 60 + "\n\n")

        for label in labels:
            f.write(f"{label}:\n")
            for op in operations + ["total"]:
                val = timing_data[label].get(op, 0)
                f.write(f"  {op:20s}:   {val:8.3f}s\n")
            f.write("\n")

        # Relative performance
        if len(labels) > 1:
            f.write("Relative Performance:\n")
            baseline_label = labels[0]
            baseline = timing_data[baseline_label]
            for label in labels[1:]:
                f.write(f"  {label} vs {baseline_label}:\n")
                for op in operations:
                    baseline_val = baseline.get(op, 1e-10)
                    val = timing_data[label].get(op, 1e-10)
                    ratio = val / baseline_val if baseline_val > 0 else 0
                    f.write(f"    {op:20s}:   {ratio:.2f}x\n")
                f.write("\n")

    print(f"✓ Saved: {summary_file}")


if __name__ == "__main__":
    main()
