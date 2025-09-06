"""A submodule containing some utility functions for profiling Synthesizer.

This module defines a set of helper functions for running different types
of "onboard" performance tests on the Synthesizer package.

For further details see the documentation on the functions below.
"""

import os
import sys
import tempfile
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7

# Set the seed
np.random.seed(42)


def _run_averaged_scaling_test(
    max_threads,
    average_over,
    log_outpath,
    operation_function,
    kwargs,
    total_msg,
):
    """Run a scaling test and average the result at each thread count.

    Args:
        max_threads (int): The maximum number of threads to test.
        average_over (int): The number of times to average the test over.
        log_outpath (str): The path to save the log file.
        operation_function (function): The function to test.
        kwargs (dict): The keyword arguments to pass to the function.
        total_msg (str): The message to print for the total time.

    Returns:
        output (str): The captured output from the test.
        threads (list): The list of thread counts used in the test.
    """
    # Save original stdout file descriptor and redirect stdout to a
    # temporary file
    original_stdout_fd = sys.stdout.fileno()
    temp_stdout = os.dup(original_stdout_fd)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        os.dup2(temp_file.fileno(), original_stdout_fd)

        # Setup lists for times
        threads = []

        # Loop over the number of threads
        nthreads = 1
        while nthreads <= max_threads:
            print(f"=== Testing with {nthreads} threads ===")
            for i in range(average_over):
                spec_start = time.time()
                operation_function(**kwargs, nthreads=nthreads)
                execution_time = time.time() - spec_start

                print(f"[Total] {total_msg}:", execution_time)

                if i == 0:
                    threads.append(nthreads)

            nthreads *= 2
            print()
        else:
            if max_threads not in threads:
                print(f"=== Testing with {max_threads} threads ===")
                for i in range(average_over):
                    spec_start = time.time()
                    operation_function(**kwargs, nthreads=max_threads)
                    execution_time = time.time() - spec_start

                    print(f"[Total] {total_msg}:", execution_time)

                    if i == 0:
                        threads.append(max_threads)

    # Reset stdout to original
    os.dup2(temp_stdout, original_stdout_fd)
    os.close(temp_stdout)

    # Read the captured output from the temporary file
    with open(temp_file.name, "r") as temp_file:
        output = temp_file.read()
    os.unlink(temp_file.name)

    return output, threads


def parse_and_collect_runtimes(
    output,
    threads,
    average_over,
    log_outpath,
    low_thresh,
):
    """Parse the output from the scaling test and collect runtimes.

    Args:
        output (str): The captured output from the test.
        threads (list): The list of thread counts used in the test.
        average_over (int): The number of times to average the test over.
        log_outpath (str): The path to save the log file.
        low_thresh (float): The threshold for low runtimes.

    Returns:
        atomic_runtimes (dict):
            A dictionary containing the runtimes for each key.
        linestyles (dict):
            A dictionary mapping keys to their respective linestyles.
    """
    # Split the output into lines
    output_lines = output.splitlines()

    # Set up our output dictionaries
    atomic_runtimes = {}
    linestyles = {}

    # Loop over the logs and collect the runtimes
    for line in output_lines:
        if ":" in line:
            # Get the key and value from the line
            key, value = line.split(":")

            # Get the stripped key
            stripped_key = (
                key.replace("[Python]", "")
                .replace("[C]", "")
                .replace("took", "")
                .replace("took (in serial)", "")
                .replace("[Total]", "")
                .strip()
            )

            # Replace the total key
            if "[Total]" in key:
                stripped_key = "Total"

            # Set the linestyle
            if key not in linestyles:
                if "[C]" in key or stripped_key == "Total":
                    linestyles[stripped_key] = "-"
                elif "[Python]" in key:
                    linestyles[stripped_key] = "--"

            # Convert the value to a float
            value = float(value.replace("seconds", "").strip())

            atomic_runtimes.setdefault(stripped_key, []).append(value)
        print(line)

    # Average every average_over runs
    for key in atomic_runtimes.keys():
        atomic_runtimes[key] = [
            np.mean(atomic_runtimes[key][i : i + average_over])
            for i in range(0, len(atomic_runtimes[key]), average_over)
        ]

    # Some operations get repeated multiple times these will have
    # more entries in atomic_runtimes lets split them
    # into their own list
    for key in atomic_runtimes.keys():
        if len(atomic_runtimes[key]) > len(threads):
            # How many times is it repeated
            n_repeats = len(atomic_runtimes[key]) // len(threads)

            # Average every n_repeats runs
            atomic_runtimes[key] = [
                np.sum(atomic_runtimes[key][i : i + n_repeats])
                for i in range(0, len(atomic_runtimes[key]), n_repeats)
            ]

    # Compute the overhead
    overhead = [
        atomic_runtimes["Total"][i]
        for i in range(len(atomic_runtimes["Total"]))
    ]
    for key in atomic_runtimes.keys():
        if key != "Total":
            for i in range(len(atomic_runtimes[key])):
                overhead[i] -= atomic_runtimes[key][i]
    atomic_runtimes["Untimed Overhead"] = overhead
    linestyles["Untimed Overhead"] = ":"

    # Temporarily add the threads to the dictionary for saving
    atomic_runtimes["Threads"] = threads

    # Convert dictionary to a structured array
    dtype = [(key, "f8") for key in atomic_runtimes.keys()]
    values = np.array(list(zip(*atomic_runtimes.values())), dtype=dtype)

    # Define the header
    header = ", ".join(atomic_runtimes.keys())

    # Save to a text file
    np.savetxt(
        log_outpath,
        values,
        fmt=[
            "%.10f" if key != "Threads" else "%d"
            for key in atomic_runtimes.keys()
        ],
        header=header,
        delimiter=",",
    )

    # Remove the threads from the dictionary
    atomic_runtimes.pop("Threads")

    # Remove any entries which are taking a tiny fraction of the time
    # and are not the total
    minimum_time = atomic_runtimes["Total"][-1] * low_thresh
    old_keys = list(atomic_runtimes.keys())
    for key in old_keys:
        if key == "Total":
            continue
        if np.mean(atomic_runtimes[key]) < minimum_time:
            atomic_runtimes.pop(key)
            linestyles.pop(key)

    # Return the runtimes and linestyles
    return atomic_runtimes, linestyles


def plot_speed_up_plot(
    atomic_runtimes,
    threads,
    linestyles,
    outpath,
    paper_style,
):
    """Plot a strong scaling test, optionally in paper style.

    Args:
        atomic_runtimes (dict):
            A dictionary containing the runtimes for each key.
        threads (list):
            A list of thread counts.
        linestyles (dict):
            A dictionary mapping keys to their respective linestyles.
        outpath (str):
            The path to save the plot.
        paper_style (bool):
            If True, produces a figure sized 3.5" x 8" with the main legend
            placed below the speedup plot's x-axis.
    """
    if paper_style:
        _plot_speed_up_paper(atomic_runtimes, threads, linestyles, outpath)
    else:
        _plot_speed_up_default(atomic_runtimes, threads, linestyles, outpath)


def _wrap_label(label, max_length=20):
    """Add new lines to a label if it exceeds a maximum length.

    The added new line will follow the word containing max_length characters.

    Args:
        label (str): The label to wrap.
        max_length (int): The maximum length of a line before wrapping.
    """
    if len(label) <= max_length:
        return label

    # Find the last space before the max length
    split_index = label.rfind(" ", 0, max_length)
    if split_index == -1:  # No space found, just split at max_length
        split_index = max_length

    # Wrap the label
    wrapped_label = label[:split_index] + "\n" + label[split_index:].strip()
    return wrapped_label


def _plot_speed_up_default(atomic_runtimes, threads, linestyles, outpath):
    # Default full-size layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(
        3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 0.05], hspace=0.0
    )

    ax_main = fig.add_subplot(gs[0, 0])
    for key in atomic_runtimes:
        ax_main.semilogy(
            threads,
            atomic_runtimes[key],
            "s" if key == "Total" else "o",
            label=key,
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )
    ax_main.set_ylabel("Time (s)")
    ax_main.grid(True)

    ax_speedup = fig.add_subplot(gs[1, 0], sharex=ax_main)
    for key in atomic_runtimes:
        t0 = atomic_runtimes[key][0]
        speedup = [t0 / t for t in atomic_runtimes[key]]
        ax_speedup.plot(
            threads,
            speedup,
            "s" if key == "Total" else "o",
            label=key,
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )
    ax_speedup.plot(
        [threads[0], threads[-1]],
        [threads[0], threads[-1]],
        "-.",
        color="black",
        label="Ideal",
        alpha=0.7,
    )
    ax_speedup.set_xlabel("Number of Threads")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.grid(True)

    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_legend = fig.add_subplot(gs[0:2, 1])
    ax_legend.axis("off")
    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.legend(
        handles, labels, loc="center left", bbox_to_anchor=(-0.3, 0.5)
    )

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="C Extension"),
        Line2D([0], [0], color="black", linestyle="--", label="Python"),
        Line2D(
            [0], [0], color="black", linestyle="-.", label="Perfect Scaling"
        ),
    ]
    ax_speedup.legend(handles=style_handles, loc="upper left")

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def _plot_speed_up_paper(atomic_runtimes, threads, linestyles, outpath):
    # Compact paper-style layout
    fig = plt.figure(figsize=(3.5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.1, 1], hspace=0.0)
    gs1 = gridspec.GridSpec(3, 1, height_ratios=[2, 1.1, 1], hspace=0.75)

    ax_main = fig.add_subplot(gs[0])
    for key in atomic_runtimes:
        ax_main.semilogy(
            threads,
            atomic_runtimes[key],
            "s" if key == "Total" else "o",
            label=_wrap_label(key, max_length=30),
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )
    ax_main.set_ylabel("Time (s)")
    ax_main.grid(True)

    ax_speedup = fig.add_subplot(gs[1], sharex=ax_main)
    for key in atomic_runtimes:
        t0 = atomic_runtimes[key][0]
        speedup = [t0 / t for t in atomic_runtimes[key]]
        ax_speedup.plot(
            threads,
            speedup,
            "s" if key == "Total" else "o",
            label=key,
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )
    ax_speedup.plot(
        [threads[0], threads[-1]],
        [threads[0], threads[-1]],
        "-.",
        color="black",
        label="Ideal",
        alpha=0.7,
    )
    ax_speedup.set_xlabel("Number of Threads")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.grid(True)

    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Sacrificial axis for atomic key legend below speedup
    ax_legend = fig.add_subplot(gs1[2])
    ax_legend.axis("off")
    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
    )

    # Inline style legend on speedup axis
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="C Extension"),
        Line2D([0], [0], color="black", linestyle="--", label="Python"),
        Line2D(
            [0], [0], color="black", linestyle="-.", label="Perfect Scaling"
        ),
    ]
    ax_speedup.add_artist(
        ax_speedup.legend(handles=style_handles, loc="upper left")
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def run_scaling_test(
    max_threads,
    average_over,
    log_outpath,
    plot_outpath,
    operation_function,
    kwargs,
    total_msg,
    low_thresh,
    paper_style=True,
):
    """Run a scaling test for the Synthesizer package.

    For this to deliver the full profiling potential Synthesizer should be
    installed with the ATOMIC_TIMING configuration option.

    Args:
        max_threads (int): The maximum number of threads to test.
        average_over (int): The number of times to average the test over.
        log_outpath (str): The path to save the log file.
        plot_outpath (str): The path to save the plot.
        operation_function (function): The function to test.
        kwargs (dict): The keyword arguments to pass to the function.
        total_msg (str): The message to print for the total time.
        low_thresh (float): The threshold for low runtimes.
        paper_style (bool): If True, produces a figure sized 3.5" x 8" with
            the main legend placed below the speedup plot's x-axis.
    """
    # Run the scaling test itself
    output, threads = _run_averaged_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        operation_function,
        kwargs,
        total_msg,
    )

    # Parse the output
    runtimes, linestyles = parse_and_collect_runtimes(
        output,
        threads,
        average_over,
        log_outpath,
        low_thresh,
    )

    # Plot the results
    plot_speed_up_plot(
        runtimes,
        threads,
        linestyles,
        plot_outpath,
        paper_style,
    )
