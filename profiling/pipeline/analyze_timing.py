"""Analyze and compare timing profiles from multiple runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_timing(filepath: Path) -> dict:
    """Load timing CSV file."""
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
    """Main entry point."""
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

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(timing_data.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(operations)

    colors = plt.cm.Set3(np.linspace(0, 1, len(operations)))
    bottom = np.zeros(len(labels))

    for i, op in enumerate(operations):
        values = [timing_data[label].get(op, 0) for label in labels]
        ax.bar(
            x + i * width - width * len(operations) / 2,
            values,
            width,
            label=op,
            color=colors[i],
            bottom=bottom,
        )
        bottom += values

    ax.set_xlabel("Run")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
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
