"""Analyze and compare memory profiles from multiple runs.

This script is the generic memory-comparison tool. It compares memory traces
over normalised execution progress and writes a summary table, but it does not
assume the run labels are numeric particle counts.

Example:
    Compare branch-level memory traces with arbitrary string labels::

        python analyse_memory.py --inputs main_full/memory.csv \
            queue_full/memory.csv --labels main_full queue_full \
            --output-dir memory_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_memory(filepath: Path) -> tuple[list, list]:
    """Loads memory usage data from a CSV file.

    Args:
        filepath (Path): The path to the memory CSV file.

    Returns:
        tuple[list, list]: A tuple containing two lists: timestamps and memory
                           values (in MB).
    """
    timestamps = []
    memory = []
    with open(filepath) as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:
                ts, mem = parts
                timestamps.append(float(ts))
                memory.append(float(mem))
    return timestamps, memory


def main() -> None:
    """Main entry point for the memory analysis script."""
    parser = argparse.ArgumentParser(
        description="Compare memory profiles from multiple profiling runs"
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

    # Load all memory data
    memory_data = {}
    for i, filepath in enumerate(args.inputs):
        label = (
            args.labels[i]
            if args.labels and i < len(args.labels)
            else filepath.stem
        )
        ts, mem = load_memory(filepath)
        memory_data[label] = (ts, mem)

    print(f"Loaded {len(memory_data)} memory profiles")

    # Create the normalized progress plot for all provided memory traces.
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(memory_data)))

    for (label, (ts, mem)), color in zip(memory_data.items(), colors):
        # Normalize time to 0-100%
        if len(ts) > 1:
            t_min, t_max = ts[0], ts[-1]
            t_normalized = [(t - t_min) / (t_max - t_min) * 100 for t in ts]
        else:
            t_normalized = [0]

        ax1.plot(
            t_normalized,
            mem,
            label=label,
            color=color,
            linewidth=2,
            alpha=1.0,
        )

        # Mark peak memory with a marker
        if mem:
            peak_idx = mem.index(max(mem))
            ax1.plot(
                t_normalized[peak_idx],
                mem[peak_idx],
                "o",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

    ax1.set_xlabel("Progress (%)", fontsize=12)
    ax1.set_ylabel("RSS Memory (MB)", fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(alpha=0.3, which="major")
    fig1.tight_layout()

    plot_file_normalized = args.output_dir / "memory_comparison_normalized.png"
    fig1.savefig(plot_file_normalized, dpi=150)
    print(f"✓ Saved: {plot_file_normalized}")

    # Create summary text
    summary_file = args.output_dir / "memory_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Memory Profile Summary\n")
        f.write("=" * 60 + "\n\n")

        for label, (ts, mem) in memory_data.items():
            if mem:
                peak = max(mem)
                mean = sum(mem) / len(mem)
                minimum = min(mem)
                duration = (ts[-1] - ts[0]) / 1000 if ts else 0

                f.write(f"{label}:\n")
                f.write(f"  Peak:       {peak:8.2f} MB\n")
                f.write(f"  Mean:       {mean:8.2f} MB\n")
                f.write(f"  Min:        {minimum:8.2f} MB\n")
                f.write(f"  Duration:   {duration:8.1f} ms\n")
                f.write(f"  Samples:    {len(mem):8d}\n\n")

    print(f"✓ Saved: {summary_file}")


if __name__ == "__main__":
    main()
