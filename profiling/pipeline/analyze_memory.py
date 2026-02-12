"""Analyze and compare memory profiles from multiple runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_memory(filepath: Path) -> tuple[list, list]:
    """Load memory CSV file, return timestamps and memory values."""
    timestamps = []
    memory = []
    with open(filepath) as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:
                ts, mem = parts
                timestamps.append(int(ts))
                memory.append(float(mem))
    return timestamps, memory


def main() -> None:
    """Main entry point."""
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
        default=Path("profiling/outputs/memory_analysis"),
        help="Output directory for plots",
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

    # Collect peak values for scaling plot
    peak_values = []
    labels_list = []
    for label in memory_data.keys():
        ts, mem = memory_data[label]
        if mem:
            peak_values.append(max(mem))
            labels_list.append(int(label))

    # Create Plot 1: Normalized time (0-100% progress)
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
            label=f"{label} particles",
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
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(alpha=0.3, which="major")
    fig1.tight_layout()

    plot_file_normalized = args.output_dir / "memory_comparison_normalized.png"
    fig1.savefig(plot_file_normalized, dpi=150)
    print(f"✓ Saved: {plot_file_normalized}")

    # Create Plot 2: Peak memory vs particle count
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(
        labels_list,
        peak_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="Peak Memory",
    )

    # Add reference scaling line (linear) anchored at first data point
    npart_ref = np.array(labels_list)
    ax2.plot(
        npart_ref,
        peak_values[0] * (npart_ref / labels_list[0]),
        "k--",
        alpha=0.5,
        linewidth=2,
        label="O(n)",
    )

    ax2.set_xlabel("Number of Particles", fontsize=12)
    ax2.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(alpha=0.3, which="major")
    fig2.tight_layout()

    plot_file_scaling = args.output_dir / "memory_comparison_scaling.png"
    fig2.savefig(plot_file_scaling, dpi=150)
    print(f"✓ Saved: {plot_file_scaling}")

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
