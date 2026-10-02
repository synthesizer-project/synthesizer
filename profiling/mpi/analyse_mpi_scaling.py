"""Plot MPI weak or strong scaling results from pipeline_mpi_scaling.py.

Produces two panels: the slowest rank's run time against the number of
ranks (with the ideal scaling), and the parallel efficiency.

    - Weak scaling efficiency is T(1) / T(n), ideally 1.
    - Strong scaling efficiency is T(1) / (n T(n)), ideally 1.

Usage:
    python profiling/mpi/analyse_mpi_scaling.py \
        --input profiling/outputs/mpi_weak/mpi_weak.csv \
        --output profiling/outputs/mpi_weak/mpi_weak_scaling.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def main():
    """Plot the scaling results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Use the latest measurement at each rank count
    df = pd.read_csv(args.input)
    df = df.drop_duplicates("nranks", keep="last").sort_values("nranks")
    mode = df["mode"].iloc[0]
    nranks = df["nranks"].to_numpy()
    times = df["run_time_max"].to_numpy()

    # Ideal scaling and efficiency relative to the smallest rank count
    base_ranks, base_time = nranks[0], times[0]
    if mode == "weak":
        ideal = np.full_like(times, base_time)
        efficiency = base_time / times
    else:
        ideal = base_time * base_ranks / nranks
        efficiency = base_time * base_ranks / (nranks * times)

    fig, (ax_time, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_time.plot(nranks, times, "o-", label="Measured")
    ax_time.plot(nranks, ideal, "k--", label="Ideal")
    ax_time.set_xscale("log", base=2)
    if mode == "weak":
        # Ideal weak scaling is flat, so show it from zero on a linear axis
        ax_time.set_ylim(0, 1.2 * max(times.max(), ideal.max()))
    else:
        ax_time.set_yscale("log")
    ax_time.set_xlabel("MPI ranks")
    ax_time.set_ylabel("Pipeline.run time (s)")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3, which="both")

    ax_eff.plot(nranks, efficiency, "o-")
    ax_eff.axhline(1.0, color="k", ls="--")
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_ylim(0, 1.1)
    ax_eff.set_xlabel("MPI ranks")
    ax_eff.set_ylabel("Parallel efficiency")
    ax_eff.grid(True, alpha=0.3, which="both")

    # Label the measured rank counts directly
    for ax in (ax_time, ax_eff):
        ax.set_xticks(nranks)
        ax.set_xticklabels([str(n) for n in nranks])
        ax.minorticks_off()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
