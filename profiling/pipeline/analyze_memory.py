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

    # Create comparison plot (overlay)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(memory_data)))

    for (label, (ts, mem)), color in zip(memory_data.items(), colors):
        ax.plot(
            [t / 1000 for t in ts], mem, label=label, color=color, linewidth=2
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("RSS Memory (MB)")
    ax.set_title("Memory Profile Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_file = args.output_dir / "memory_comparison.png"
    fig.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")

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
