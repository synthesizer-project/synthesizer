"""A test for the speed and accuracy of the tree los calculation.

This script will compare the speed and accuracy of the tree los calculation
against the loop los calculation for a range of star and gas counts.

Example usage:

    $ python compare_los_loop_tree.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from unyt import Myr

from synthesizer.grid import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)

start = time.time()

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
# script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

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

xs = {}
loop_ys = {}
tree_ys = {}
precision = {}

for n in [100, 1000, 1000]:
    xs.setdefault(n, [])
    loop_ys.setdefault(n, [])
    tree_ys.setdefault(n, [])
    precision.setdefault(n, [])

    # First make the stars

    # Generate some random coordinates
    coords = CoordinateGenerator.generate_3D_gaussian(
        n,
        mean=np.array([50, 50, 50]),
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
        n,
        coordinates=coords,
        current_masses=np.full(n, 10**8.7 / n),
        smoothing_lengths=smls,
        redshift=1,
    )

    for ngas in np.logspace(2, 4, 3, dtype=int):
        # Now make the gas

        # Generate some random coordinates
        coords = CoordinateGenerator.generate_3D_gaussian(
            ngas,
            mean=np.array([50, 50, 50]),
        )

        # Calculate the smoothing lengths
        smls = calculate_smoothing_lengths(coords, num_neighbors=56)

        gas = Gas(
            masses=np.random.uniform(10**6, 10**6.5, ngas),
            metallicities=np.random.uniform(0.01, 0.05, ngas),
            coordinates=coords,
            smoothing_lengths=smls,
            dust_to_metal_ratio=0.2,
        )

        # Create galaxy object
        galaxy = Galaxy("Galaxy", stars=stars, gas=gas, redshift=1)

        # Get the kernel
        kernel = Kernel().get_kernel()

        # Calculate the tau_vs
        start = time.time()
        tau_v = galaxy.calculate_los_tau_v(
            kappa=0.07,
            kernel=kernel,
            force_loop=1,
        )
        loop_time = time.time() - start
        loop_sum = np.sum(tau_v)

        # Calculate the tau_vs
        start = time.time()
        tau_v = galaxy.calculate_los_tau_v(
            kappa=0.07,
            kernel=kernel,
            min_count=100,
        )
        tree_time = time.time() - start
        tree_sum = np.sum(tau_v)

        xs[n].append(ngas)
        loop_ys[n].append(loop_time)
        tree_ys[n].append(tree_time)
        precision[n].append(np.abs(tree_sum - loop_sum) / loop_sum * 100)

        print(
            f"LOS calculation with tree took {tree_time:.4f} "
            f"seconds for nstar={n} and ngas={ngas}"
        )
        print(
            f"LOS calculation with loop took {loop_time:.4f} "
            f"seconds for nstar={n} and ngas={ngas}"
        )
        print(
            "Ratio in wallclock: "
            f"Time_loop/Time_tree={loop_time / tree_time:.4f}"
        )
        print(
            f"Tree gave={tree_sum:.2e} Loop gave={loop_sum:.2e} "
            "Normalised residual="
            f"{np.abs(tree_sum - loop_sum) / loop_sum * 100:.4f}"
        )
        print()

# Convert to numpy arrays
for n in xs.keys():
    xs[n] = np.array(xs[n])
    sinds = np.argsort(xs[n])
    xs[n] = xs[n][sinds]
    loop_ys[n] = np.array(loop_ys[n])[sinds]
    tree_ys[n] = np.array(tree_ys[n])[sinds]
    precision[n] = np.array(precision[n])[sinds]

# Create lists of colours to use
colours = ["b", "g", "r", "c", "m", "y", "k"]

# Create the legend handles
legend_handles = [
    Line2D([0], [0], color="k", lw=2, label="Loop", linestyle="-"),
    Line2D([0], [0], color="k", lw=2, label="Tree", linestyle="--"),
]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog()
ax.grid()

# Plot for each star count
for i, n in enumerate(xs.keys()):
    ax.plot(xs[n], loop_ys[n], color=colours[i])
    ax.plot(xs[n], tree_ys[n], color=colours[i], linestyle="--")

    # Add a handle for this star count
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colours[i],
            lw=2,
            label=f"$N_\\star={n}$",
            linestyle="-",
        )
    )

ax.set_ylabel("Wallclock (s)")
ax.set_xlabel(r"$N_\mathrm{gas}$")

ax.legend(handles=legend_handles)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogx()
ax.grid()


# Plot for each star count
for i, n in enumerate(xs.keys()):
    ax.plot(xs[n], precision[n], color=colours[i], label=f"$N_\\star={n}$")


ax.set_ylabel(r"$|\tau_{V, tree} - \tau_{V, loop}|" r" / \tau_{V, loop}$ (%)")
ax.set_xlabel("$N_\\mathrm{gas}$")

ax.legend()

plt.show()
