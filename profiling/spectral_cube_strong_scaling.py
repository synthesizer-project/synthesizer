"""A script to test the strong scaling of the spectral data cube calculation.

Usage:
    python spectral_cube_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5 --average_over 10
"""

import argparse
import os
import sys
import tempfile
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from unyt import Myr, kpc

from synthesizer.emission_models import IncidentEmission
from synthesizer.filters import FilterCollection as Filters
from synthesizer.grid import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def cube_strong_scaling(
    basename,
    max_threads=8,
    nstars=10**4,
    average_over=10,
    type="smoothed",
):
    """Profile the cpu time usage of the data cube calculation."""
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir, lam_lims=(1000, 100000))

    # Define the width of the data cube
    width = 30 * kpc

    # Define data cube resolution (here we arbitrarily set it to 100
    # pixels along an axis)
    resolution = width / 100

    # Define the grid (normally this would be defined by an SPS grid)
    log10ages = np.arange(6.0, 10.5, 0.1)
    metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
    Z_p = {"metallicity": 0.01}
    metal_dist = ZDist.DeltaConstant(**Z_p)
    sfh_p = {"duration": 100 * Myr}
    sfh = SFH.Constant(**sfh_p)  # constant star formation

    # Generate the star formation metallicity history
    mass = 10**10
    param_stars = ParametricStars(
        log10ages,
        metallicities,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=mass,
    )

    # Generate some random coordinates
    coords = CoordinateGenerator.generate_3D_gaussian(
        nstars,
        mean=np.array([50, 50, 50]),
        cov=np.array([[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]]),
    )

    # Calculate the smoothing lengths
    smls = calculate_smoothing_lengths(coords, num_neighbors=56)

    # Sample the SFZH, producing a Stars object
    # we will also pass some keyword arguments for attributes
    # we will need for imaging
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        coordinates=coords,
        current_masses=np.full(nstars, 10**8.7 / nstars),
        smoothing_lengths=smls,
        redshift=1,
        centre=np.array([50, 50, 50]),
    )

    # Create galaxy object
    gal = Galaxy("Galaxy", stars=stars, redshift=1)
    gal.stars.get_radii()
    print(gal.stars.radii.max())

    # Calculate the stellar rest frame SEDs for all particles in erg / s / Hz
    model = IncidentEmission(grid, per_particle=True)
    sed = gal.stars.get_spectra(model)

    # Calculate the observed SED in nJy
    sed.get_fnu(cosmo, gal.redshift, igm=None)

    filter_codes = [
        "JWST/NIRCam.F200W",
    ]

    filters = Filters(filter_codes, new_lam=grid.lam)

    gal.stars.particle_spectra["incident"].get_photo_luminosities(filters)

    # Get the kernel
    kernel = Kernel().get_kernel()

    # Get the tau_vs in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial serial calculation")
    gal.get_data_cube(
        resolution,
        fov=width,
        lam=grid.lam,
        cube_type=type,
        stellar_spectra="incident",
        blackhole_spectra=None,
        kernel=kernel,
        kernel_threshold=1,
    )
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
                gal.get_data_cube(
                    resolution,
                    fov=width,
                    lam=grid.lam,
                    cube_type=type,
                    stellar_spectra="incident",
                    blackhole_spectra=None,
                    kernel=kernel,
                    kernel_threshold=1,
                    nthreads=nthreads,
                )
                execution_time = time.time() - spec_start

                print(
                    "[Total] Getting data cube execution time:",
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
                    gal.get_data_cube(
                        resolution,
                        fov=width,
                        lam=grid.lam,
                        cube_type=type,
                        stellar_spectra="incident",
                        blackhole_spectra=None,
                        kernel=kernel,
                        kernel_threshold=1,
                        nthreads=max_threads,
                    )
                    execution_time = time.time() - spec_start

                    print(
                        "[Total] Getting data cube execution time:",
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
        f"{basename}_data_cube_strong_scaling_{nstars}.txt",
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
    ax_main.set_title(f"Data Cube Strong Scaling ({nstars} stars)")
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
        f"{basename}_data_cube_scaling_NStars"
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
        default=10**4,
        help="The number of stars to use in the simulation.",
    )

    args.add_argument(
        "--average_over",
        type=int,
        default=10,
        help="The number of times to average over.",
    )

    args.add_argument(
        "--type",
        type=str,
        default="smoothed",
        help="What type of spectral data cube to make, smoothed or hist.",
    )

    args = args.parse_args()

    cube_strong_scaling(
        args.basename,
        args.max_threads,
        args.nstars,
        args.average_over,
        args.type,
    )
