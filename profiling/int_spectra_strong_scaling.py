"""A script to test the strong scaling of the integrated spectra calculation.

Usage:
    python int_spectra_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5
"""

import argparse
import os
import sys
import tempfile
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfhz
from unyt import Myr

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def int_spectra_strong_scaling(
    basename,
    max_threads=8,
    nstars=10**5,
    average_over=10,
    gam="cic",
):
    """Profile the cpu time usage of the integrated spectra calculation."""
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the grid (normally this would be defined by an SPS grid)
    log10ages = np.arange(6.0, 10.5, 0.1)
    metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
    metal_dist = ZDist.Normal(0.005, 0.01)
    sfh = SFH.Constant(100 * Myr)  # constant star formation

    # Generate the star formation metallicity history
    mass = 10**10
    param_stars = ParametricStars(
        log10ages,
        metallicities,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=mass,
    )

    # Sample the SFZH, producing a Stars object
    stars = sample_sfhz(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        current_masses=np.full(nstars, 10**8.7 / nstars),
        redshift=1,
    )

    # Get spectra in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial serial spectra calculation")
    stars.get_spectra_incident(grid, nthreads=1, grid_assignment_method=gam)
    print()

    # Step 1: Save original stdout file descriptor and redirect
    # stdout to a temporary file
    original_stdout_fd = sys.stdout.fileno()
    temp_stdout = os.dup(original_stdout_fd)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        os.dup2(temp_file.fileno(), original_stdout_fd)

        # Setup lists for times
        times = []
        threads = []

        # Loop over the number of threads
        nthreads = 1
        while nthreads <= max_threads:
            print(f"=== Testing with {nthreads} threads ===")
            for i in range(average_over):
                spec_start = time.time()
                stars.get_spectra_incident(
                    grid, nthreads=nthreads, grid_assignment_method=gam
                )
                execution_time = time.time() - spec_start

                print(
                    "[Total] Getting spectra execution time:",
                    execution_time,
                )

                times.append(execution_time)
                if i == 0:
                    threads.append(nthreads)

            nthreads *= 2
            print()
        else:
            if max_threads not in threads:
                print(f"=== Testing with {max_threads} threads ===")
                for i in range(average_over):
                    spec_start = time.time()
                    stars.get_spectra_incident(
                        grid,
                        nthreads=max_threads,
                        grid_assignment_method=gam,
                    )
                    execution_time = time.time() - spec_start

                    print(
                        "[Total] Getting spectra execution time:",
                        execution_time,
                    )

                    times.append(execution_time)
                    if i == 0:
                        threads.append(max_threads)

    # Step 3: Reset stdout to original
    os.dup2(temp_stdout, original_stdout_fd)
    os.close(temp_stdout)

    # Step 4: Read the captured output from the temporary file
    with open(temp_file.name, "r") as temp_file:
        output = temp_file.read()
    os.unlink(temp_file.name)

    # Step 5: Parse the output lines and store in a dictionary
    output_lines = output.splitlines()
    atomic_runtimes = {}

    linestyles = {}
    for line in output_lines:
        if "===" in line:
            nthreads = int(line.split()[3])
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
        f"{basename}_integrated_strong_scaling_{gam}_{nstars}.txt",
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

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(
        3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 0.05], hspace=0.0
    )

    # Main plot
    ax_main = fig.add_subplot(gs[0, 0])
    for key in atomic_runtimes.keys():
        ax_main.semilogy(
            threads,
            atomic_runtimes[key],
            "s" if key == "Total" else "o",
            label=key,
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )

    ax_main.set_ylabel("Time (s)")
    ax_main.set_title(f"Integrated Spectra Strong Scaling ({nstars} stars)")
    ax_main.grid(True)

    # Speedup plot
    ax_speedup = fig.add_subplot(gs[1, 0], sharex=ax_main)
    for key in atomic_runtimes.keys():
        initial_time = atomic_runtimes[key][0]
        speedup = [initial_time / t for t in atomic_runtimes[key]]
        ax_speedup.plot(
            threads,
            speedup,
            "s" if key == "Total" else "o",
            label=key,
            linestyle=linestyles[key],
            linewidth=3 if key == "Total" else 1,
        )

    # PLot a 1-1 line
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

    # Hide x-tick labels for the main plot
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Sacrificial axis for the legend
    ax_legend = fig.add_subplot(gs[0:2, 1])
    ax_legend.axis("off")  # Hide the sacrificial axis

    # Create the legend
    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.legend(
        handles, labels, loc="center left", bbox_to_anchor=(-0.3, 0.5)
    )

    # Add a second key for linestyle
    handles = [
        plt.Line2D(
            [0], [0], color="black", linestyle="-", label="C Extension"
        ),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Python"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-.", label="Perfect Scaling"
        ),
    ]
    ax_speedup.legend(handles=handles, loc="upper left")

    fig.savefig(
        f"{basename}_integrated_strong_scaling_{gam}_NStars"
        f"{nstars}_TotThreahs{max_threads}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    # Get the command line args
    args = argparse.ArgumentParser()

    args.add_argument(
        "--basename",
        type=str,
        default="test",
        help="The basename of the output files.",
    )

    args.add_argument(
        "--max_threads",
        type=int,
        default=8,
        help="The maximum number of threads to use.",
    )

    args.add_argument(
        "--nstars",
        type=int,
        default=10**5,
        help="The number of stars to use in the simulation.",
    )

    args.add_argument(
        "--average_over",
        type=int,
        default=10,
        help="The number of times to average over.",
    )

    args.add_argument(
        "--grid_assign",
        type=str,
        default="cic",
        help="The grid assignment method (cic or ngp).",
    )

    args = args.parse_args()

    int_spectra_strong_scaling(
        args.basename,
        args.max_threads,
        args.nstars,
        args.average_over,
        args.grid_assign,
    )
