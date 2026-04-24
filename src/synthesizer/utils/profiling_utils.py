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

from synthesizer.instruments import Instrument, InstrumentCollection

plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7

# Set the seed
np.random.seed(42)


def get_instrument_profile(label, filepath, filters=None, resolution=None):
    """Load an instrument from a file or create and save it if not found.

    This ensures that instruments (and their filters) are cached locally,
    allowing scripts to run on systems without internet access.

    Args:
        label (str): The label of the instrument.
        filepath (str): The path to the HDF5 file.
        filters (FilterCollection, optional): The filters to use if creating.
        resolution (unyt_quantity, optional): The resolution to use if
            creating.

    Returns:
        Instrument: The loaded or created instrument.
    """
    if os.path.exists(filepath):
        print(f"Loading instrument '{label}' from {filepath}")
        collection = InstrumentCollection(filepath)
        return collection[label]

    print(f"Instrument file {filepath} not found. Creating instrument...")

    # Create the instrument
    instrument = Instrument(
        label=label,
        filters=filters,
        resolution=resolution,
    )

    # Save it to a collection file
    collection = InstrumentCollection()
    collection.add_instruments(instrument)
    collection.write_instruments(filepath)
    print(f"Saved instrument '{label}' to {filepath}")

    return instrument


def _run_single(
    timers,
    operation_function,
    kwargs,
    nthreads,
    run_idx,
    total_msg,
):
    """Run a single profiling iteration and capture timing data."""
    # Reset timers for this individual run
    timers.reset()

    # Time the operation
    spec_start = time.time()
    operation_function(**kwargs, nthreads=nthreads)
    total_time = time.time() - spec_start

    print(f"[Total] {total_msg} took: {total_time} s", flush=True)

    # Extract timing data from this run
    run_data = {
        "nthreads": nthreads,
        "run": run_idx,
        "total_time": total_time,
        "operations": {},
    }

    # Copy all accumulated operation timings from this run
    for op in timers.keys():
        cumulative_time, call_count, source = timers[op]
        run_data["operations"][op] = {
            "cumulative_time": cumulative_time,
            "call_count": call_count,
            "source": source,
        }

    return run_data


def _run_averaged_scaling_test(
    max_threads,
    average_over,
    log_outpath,
    operation_function,
    kwargs,
    total_msg,
):
    """Run a scaling test and collect per-run timing data.

    This function runs the operation_function multiple times with different
    thread counts. For each individual run, it resets the atomic timers,
    executes the operation, and extracts the accumulated timing data.
    This allows Python to average the timings across multiple runs.

    Args:
        max_threads (int): The maximum number of threads to test.
        average_over (int): The number of times to average the test over.
        log_outpath (str): The path to save the log file.
        operation_function (function): The function to test.
        kwargs (dict): The keyword arguments to pass to the function.
        total_msg (str): The message to print for the total time.

    Returns:
        output (str): The captured output from the test (for logging).
        threads (list): The list of thread counts used in the test.
        run_data_list (list): List of dicts containing per-run timing data.
    """
    from synthesizer.utils.operation_timers import OperationTimers

    # Create timers instance
    timers = OperationTimers()

    # Store all run data
    run_data_list = []
    # Save original stdout file descriptor and redirect stdout to a
    # temporary file (for logging)
    original_stdout_fd = sys.stdout.fileno()
    temp_stdout = os.dup(original_stdout_fd)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        os.dup2(temp_file.fileno(), original_stdout_fd)

        # Setup lists
        threads = []

        # Loop over the number of threads
        nthreads = 1
        while nthreads <= max_threads:
            print(f"=== Testing with {nthreads} threads ===", flush=True)

            for run_idx in range(average_over):
                run_data = _run_single(
                    timers,
                    operation_function,
                    kwargs,
                    nthreads,
                    run_idx,
                    total_msg,
                )
                run_data_list.append(run_data)

                if run_idx == 0:
                    threads.append(nthreads)

            nthreads *= 2
            print(flush=True)
        else:
            if max_threads not in threads:
                print(
                    f"=== Testing with {max_threads} threads ===", flush=True
                )
                for run_idx in range(average_over):
                    run_data = _run_single(
                        timers,
                        operation_function,
                        kwargs,
                        max_threads,
                        run_idx,
                        total_msg,
                    )
                    run_data_list.append(run_data)

                    if run_idx == 0:
                        threads.append(max_threads)

    # Reset stdout to original
    os.dup2(temp_stdout, original_stdout_fd)
    os.close(temp_stdout)

    # Read the captured output from the temporary file
    with open(temp_file.name, "r") as temp_file:
        output = temp_file.read()
    os.unlink(temp_file.name)

    return output, threads, run_data_list


def parse_and_collect_runtimes(
    threads,
    run_data_list,
    average_over,
    log_outpath,
    low_thresh,
):
    """Process per-run timing data and prepare for plotting.

    This function takes the list of per-run timing dictionaries and processes
    them into the format expected by the plotting functions. It groups runs
    by thread count, averages them, and filters out operations that don't
    meet the low_thresh criterion.

    Args:
        threads (list): The list of thread counts used in the test.
        run_data_list (list): List of dicts with per-run timing data.
        average_over (int): The number of times to average the test over.
        log_outpath (str): The path to save the log file.
        low_thresh (float): Fraction of total time threshold for inclusion.

    Returns:
        atomic_runtimes (dict):
            A dictionary containing the runtimes for each key.
        linestyles (dict):
            A dictionary mapping keys to their respective linestyles.
    """
    # Collect all operation names across all runs
    all_operations = set()
    for run_data in run_data_list:
        all_operations.update(run_data["operations"].keys())

    # Initialize dictionaries
    atomic_runtimes = {}
    linestyles = {}
    call_counts = {}

    # For each operation, collect times grouped by thread count
    for operation in all_operations:
        # Collect cumulative times for each run, grouped by thread count
        times_by_thread = {nt: [] for nt in threads}
        counts_by_thread = {nt: [] for nt in threads}
        source = None

        for run_data in run_data_list:
            if operation in run_data["operations"]:
                op_data = run_data["operations"][operation]
                times_by_thread[run_data["nthreads"]].append(
                    op_data["cumulative_time"]
                )
                counts_by_thread[run_data["nthreads"]].append(
                    op_data["call_count"]
                )
                # Capture source (same for all runs of this operation)
                if source is None:
                    source = op_data["source"]

        # Average across runs for each thread count
        averaged_times = []
        averaged_counts = []
        for nt in threads:
            if times_by_thread[nt]:
                averaged_times.append(np.mean(times_by_thread[nt]))
                averaged_counts.append(np.mean(counts_by_thread[nt]))
            else:
                # Operation didn't appear in any runs for this thread count
                averaged_times.append(np.nan)
                averaged_counts.append(np.nan)

        atomic_runtimes[operation] = averaged_times
        call_counts[operation] = averaged_counts

        # Set linestyle based on source
        if source == "C":
            linestyles[operation] = "-"
        elif source == "Python":
            linestyles[operation] = "--"

    # Process Total times
    total_times_by_thread = {nt: [] for nt in threads}
    for run_data in run_data_list:
        total_times_by_thread[run_data["nthreads"]].append(
            run_data["total_time"]
        )

    # Average Total across runs for each thread count
    averaged_totals = []
    for nt in threads:
        averaged_totals.append(np.mean(total_times_by_thread[nt]))

    atomic_runtimes["Total"] = averaged_totals
    linestyles["Total"] = "-"

    # Compute the overhead (only if Total exists)
    if "Total" in atomic_runtimes:
        overhead = [
            atomic_runtimes["Total"][i]
            for i in range(len(atomic_runtimes["Total"]))
        ]
        for key in atomic_runtimes.keys():
            if key != "Total":
                for i in range(len(atomic_runtimes[key])):
                    safe_value = np.nan_to_num(
                        atomic_runtimes[key][i], nan=0.0
                    )
                    overhead[i] -= safe_value
        atomic_runtimes["Untimed Overhead"] = overhead
        linestyles["Untimed Overhead"] = ":"
    else:
        print("WARNING: No 'Total' timing found in output")

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
    if "Total" in atomic_runtimes:
        minimum_time = atomic_runtimes["Total"][-1] * low_thresh
        old_keys = list(atomic_runtimes.keys())
        for key in old_keys:
            if key == "Total" or key == "Untimed Overhead":
                continue
            if np.mean(atomic_runtimes[key]) < minimum_time:
                atomic_runtimes.pop(key)
                linestyles.pop(key)
                call_counts.pop(key, None)  # Remove from counts too

    # Return the runtimes, linestyles, and call counts
    return atomic_runtimes, linestyles, call_counts


def plot_speed_up_plot(
    atomic_runtimes,
    threads,
    linestyles,
    call_counts,
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
        call_counts (dict):
            A dictionary containing call counts for each operation.
        outpath (str):
            The path to save the plot.
        paper_style (bool):
            If True, produces a figure sized 3.5" x 8" with the main legend
            placed below the speedup plot's x-axis.
    """
    if paper_style:
        _plot_speed_up_paper(
            atomic_runtimes, threads, linestyles, call_counts, outpath
        )
    else:
        _plot_speed_up_default(
            atomic_runtimes, threads, linestyles, call_counts, outpath
        )


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


def _plot_speed_up_default(
    atomic_runtimes, threads, linestyles, _call_counts, outpath
):
    # _call_counts is unused; kept for API symmetry.
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
        speedup = [
            t0 / t if t > 0 and not np.isnan(t) else np.nan
            for t in atomic_runtimes[key]
        ]
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


def _plot_speed_up_paper(
    atomic_runtimes, threads, linestyles, _call_counts, outpath
):
    # _call_counts is unused; kept for API symmetry.
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
        speedup = [
            t0 / t if t > 0 and not np.isnan(t) else np.nan
            for t in atomic_runtimes[key]
        ]
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
    output, threads, run_data_list = _run_averaged_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        operation_function,
        kwargs,
        total_msg,
    )

    # Parse the output
    runtimes, linestyles, call_counts = parse_and_collect_runtimes(
        threads,
        run_data_list,
        average_over,
        log_outpath,
        low_thresh,
    )

    # Plot the results
    plot_speed_up_plot(
        runtimes,
        threads,
        linestyles,
        call_counts,
        plot_outpath,
        paper_style,
    )
