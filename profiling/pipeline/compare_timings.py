"""Compare one or more timing profiles with named labels.

This script is intended for direct named comparisons (e.g. main vs one or
more candidate branches) rather than particle-scaling plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_timing(filepath: Path) -> dict[str, dict[str, float | str]]:
    """Load operation timings from a CSV file.

    Args:
        filepath (Path): Path to a timing CSV file.

    Returns:
        dict[str, dict[str, float | str]]:
            Operation map with keys 'time' and 'source'.
    """
    timings: dict[str, dict[str, float | str]] = {}
    with open(filepath) as f:
        lines = f.readlines()
        if not lines:
            return timings

        for line in lines[1:]:
            parts = line.strip().split(",")
            operation, seconds, _, source = (
                parts[0],
                float(parts[1]),
                int(parts[2]),
                parts[3],
            )
            timings[operation] = {"time": seconds, "source": source}

    return timings


def compare_many(
    timing_data: dict[str, dict[str, dict[str, float | str]]],
    labels: list[str],
    output_dir: Path,
    top_n: int,
    min_fraction: float,
) -> None:
    """Generate a named timing comparison plot and summary.

    Args:
        timing_data (dict[str, dict[str, dict[str, float | str]]]):
            Mapping of label to operation timing entries.
        labels (list[str]): Ordered run labels; first entry is reference.
        output_dir (Path): Output directory for plot and summary text.
        top_n (int): Maximum number of operations to include.
        min_fraction (float): Minimum percent of total runtime to keep an op.
    """
    if len(labels) < 2:
        raise ValueError("Need at least two timing profiles to compare.")

    operations: set[str] = set()
    for label in labels:
        operations.update(timing_data[label].keys())

    totals = {
        label: sum(float(v["time"]) for v in timing_data[label].values())
        for label in labels
    }

    kept_ops: list[str] = []
    for op in sorted(operations):
        max_fraction = 0.0
        for label in labels:
            op_time = float(timing_data[label].get(op, {}).get("time", 0.0))
            total = totals[label]
            frac = 100.0 * op_time / total if total > 0 else 0.0
            max_fraction = max(max_fraction, frac)
        if max_fraction >= min_fraction:
            kept_ops.append(op)

    # Sort by the largest runtime across all runs, so candidate-only hotspots
    # are retained.
    ref_label = labels[0]
    kept_ops.sort(
        key=lambda op: max(
            float(timing_data[label].get(op, {}).get("time", 0.0))
            for label in labels
        ),
        reverse=True,
    )
    kept_ops = kept_ops[:top_n]
    kept_ops.reverse()

    # Always include a total-runtime bar for each run.
    plot_ops = ["TOTAL"] + kept_ops

    y = np.arange(len(plot_ops))
    n_runs = len(labels)
    group_height = 0.8
    bar_height = group_height / n_runs

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 3)))

    for i, label in enumerate(labels):
        values = []
        for op in plot_ops:
            if op == "TOTAL":
                values.append(totals[label])
            else:
                values.append(
                    float(timing_data[label].get(op, {}).get("time", 0.0))
                )
        offsets = y - group_height / 2 + (i + 0.5) * bar_height
        ax.barh(
            offsets,
            values,
            bar_height,
            label=label,
            color=cmap[i],
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Operation", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_ops, fontsize=9)
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="major", axis="x")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    plot_file = output_dir / "timing_comparison.png"
    fig.savefig(plot_file, dpi=200)
    print(f"✓ Saved: {plot_file}")

    summary_file = output_dir / "timing_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Timing Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Reference: {ref_label}\n")
        f.write(f"Compared runs: {', '.join(labels[1:])}\n\n")

        f.write("Totals:\n")
        for label in labels:
            f.write(f"  {label:20s}: {totals[label]:10.3f} s\n")

        if totals[ref_label] > 0:
            f.write("\nRelative to reference:\n")
            for label in labels[1:]:
                speedup = (
                    totals[ref_label] / totals[label]
                    if totals[label] > 0
                    else np.inf
                )
                delta = (
                    (totals[label] - totals[ref_label])
                    / totals[ref_label]
                    * 100.0
                )
                f.write(
                    f"  {label:20s}: delta {delta:8.2f}%, "
                    f"speedup {speedup:6.2f}x\n"
                )

        f.write("\nTop operation deltas (by reference runtime):\n")
        for op in ["TOTAL"] + kept_ops[::-1]:
            if op == "TOTAL":
                ref_time = totals[ref_label]
            else:
                ref_time = float(
                    timing_data[ref_label].get(op, {}).get("time", 0.0)
                )
            f.write(f"  {op:40s}: {ref_label}={ref_time:9.3f}s")

            for label in labels[1:]:
                if op == "TOTAL":
                    run_time = totals[label]
                else:
                    run_time = float(
                        timing_data[label].get(op, {}).get("time", 0.0)
                    )
                if ref_time > 0:
                    delta = (run_time - ref_time) / ref_time * 100.0
                    speedup = ref_time / run_time if run_time > 0 else np.inf
                    f.write(
                        f", {label}={run_time:9.3f}s "
                        f"(d {delta:7.2f}%, s {speedup:5.2f}x)"
                    )
                else:
                    f.write(f", {label}={run_time:9.3f}s")
            f.write("\n")

    print(f"✓ Saved: {summary_file}")


def main() -> None:
    """Parse CLI arguments and compare timing CSV profiles."""
    parser = argparse.ArgumentParser(
        description="Compare one or more timing CSV profiles by name"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Timing CSV files. First file is the reference run; remaining "
            "files are compared against it."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=None,
        help="Labels for runs (default: parent directory names)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=18,
        help="Maximum operations to show on the plot (default: 18)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=1.0,
        help=(
            "Minimum percent contribution in any run to keep an operation "
            "(default: 1.0)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Output directory for plot and summary",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.inputs) < 2:
        raise ValueError("Provide at least two input timing CSV files.")

    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same length as --inputs.")

    labels = (
        args.labels
        if args.labels is not None
        else [path.parent.name for path in args.inputs]
    )

    timing_data = {
        label: load_timing(path) for label, path in zip(labels, args.inputs)
    }

    compare_many(
        timing_data=timing_data,
        labels=labels,
        output_dir=args.output_dir,
        top_n=args.top_n,
        min_fraction=args.min_fraction,
    )


if __name__ == "__main__":
    main()
