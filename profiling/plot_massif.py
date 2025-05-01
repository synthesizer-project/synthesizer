"""A script to plot the memory usage of a script profiled with valgrind.

This script is used to visualize memory usage from valgrind massif tool output
files. You can provide one or more massif output files to plot their memory
usage over time.

Usage:
    python plot_massif.py massif.out.1 massif.out.2

Example of profiling a script with valgrind:
    valgrind --tool=massif --time-unit=ms --detailed-freq=1
       --massif-out-file=massif.out python your_script.py

This will generate a massif output file named massif.out, which can then be
visualized using this script.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_massif_file(filename):
    """Parse a massif output file to extract time points and memory usage.

    Args:
        filename (str):
            The name of the massif output file.

    Returns:
        tuple:
            Two lists containing time points (in milliseconds) and memory usage
            (in KB).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    time_points = []
    memory_usage = []

    for line in lines:
        # Skip lines that don't contain memory usage data
        if not line.startswith("time=") and not line.startswith("mem_heap_B"):
            continue

        # Extract time and memory usage data
        if line.startswith("time="):
            time_points.append(int(line.split("=")[1]))
        else:
            memory_usage.append(int(line.split("=")[1]) / 1024)

    # Convert to arrays
    time_points = np.array(time_points)
    memory_usage = np.array(memory_usage)

    # Sort by time points
    sort_idx = np.argsort(time_points)
    time_points = time_points[sort_idx]
    memory_usage = memory_usage[sort_idx]

    return time_points, memory_usage


def plot_massif_files(filenames):
    """Plot memory usage over time for multiple massif output files.

    Args:
        filenames (list):
            A list of massif output file names.
    """
    plt.figure(figsize=(10, 6))

    for filename in filenames:
        time_points, memory_usage = parse_massif_file(filename)
        plt.semilogy(
            time_points, memory_usage, label="_".join(filename.split(".")[:-1])
        )

    plt.xlabel("Time (ms)")
    plt.ylabel("Memory Usage (KB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("massif_memory_usage.png")
    plt.show()


def main():
    """Plot the contents of valgrind massif outputs."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Plot memory usage over time from massif output files.\n\n"
            "This script is used to visualize memory usage from valgrind "
            "massif tool output files.\n"
            "You can provide one or more massif output files to plot their "
            "memory usage over time.\n\n"
            "Example usage:\n"
            "  python plot_massif.py massif.out.1 massif.out.2\n\n"
            "Example of profiling a script with valgrind:\n"
            "  valgrind --tool=massif --massif-out-file=massif.out "
            "python your_script.py\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "files",
        metavar="F",
        type=str,
        nargs="+",
        help="massif output files to plot",
    )

    args = parser.parse_args()

    # Plot the memory usage for the provided massif output files
    plot_massif_files(args.files)


if __name__ == "__main__":
    main()
