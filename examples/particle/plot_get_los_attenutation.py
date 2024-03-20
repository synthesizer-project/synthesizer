"""
Plot line of sight diagnostics
==============================

This example shows how to compute line of sight dust surface densities,
and plots some diagnostics.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfhz
from unyt import Myr

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

for n in [10, 100]:  # , 1000, 10000]:
    xs = []
    loop_ys = []
    tree_ys = []
    precision = []
    for ngas in np.logspace(np.log10(n), 6, 20, dtype=int):
        # Make a fake galaxy

        # First make the stars

        # Generate some random coordinates
        coords = CoordinateGenerator.generate_3D_gaussian(n)

        # Calculate the smoothing lengths from radii
        cent = np.mean(coords, axis=0)
        rs = np.sqrt(
            (coords[:, 0] - cent[0]) ** 2
            + (coords[:, 1] - cent[1]) ** 2
            + (coords[:, 2] - cent[2]) ** 2
        )
        rs[rs < 0.2] = 0.6  # Set a lower bound on the "smoothing length"

        # Sample the SFZH, producing a Stars object
        # we will also pass some keyword arguments for attributes
        # we will need for imaging
        stars = sample_sfhz(
            param_stars.sfzh,
            param_stars.log10ages,
            param_stars.log10metallicities,
            n,
            coordinates=coords,
            current_masses=np.full(n, 10**8.7 / n),
            smoothing_lengths=rs / 2,
            redshift=1,
        )

        # Now make the gas

        # Generate some random coordinates
        coords = CoordinateGenerator.generate_3D_gaussian(ngas)

        # Calculate the smoothing lengths from radii
        cent = np.mean(coords, axis=0)
        rs = np.sqrt(
            (coords[:, 0] - cent[0]) ** 2
            + (coords[:, 1] - cent[1]) ** 2
            + (coords[:, 2] - cent[2]) ** 2
        )
        rs[rs < 0.2] = 0.6  # Set a lower bound on the "smoothing length"

        gas = Gas(
            masses=np.random.uniform(10**6, 10**6.5, ngas),
            metallicities=np.random.uniform(0.01, 0.05, ngas),
            coordinates=coords,
            smoothing_lengths=rs / 4,
            dust_to_metal_ratio=0.2,
        )

        # Create galaxy object
        galaxy = Galaxy("Galaxy", stars=stars, gas=gas, redshift=1)

        # Create a fake kernel
        kernel = np.random.normal(0.5, 0.25, 100)
        kernel = np.sort(kernel)[::-1]
        kernel /= np.sum(kernel)

        # Calculate the tau_vs
        start = time.time()
        tau_v = galaxy.calculate_los_tau_v(
            kappa=0.07, kernel=kernel, force_loop=1
        )
        loop_time = time.time() - start
        loop_sum = np.sum(tau_v)

        # Calculate the tau_vs
        start = time.time()
        tau_v = galaxy.calculate_los_tau_v(kappa=0.07, kernel=kernel)
        tree_time = time.time() - start
        tree_sum = np.sum(tau_v)

        xs.append(n * ngas)
        loop_ys.append(loop_time)
        tree_ys.append(tree_time)
        precision.append(np.abs(tree_sum - loop_sum) / loop_sum * 100)

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
            f"Tree gave={tree_sum:.2f} Loop gave={loop_sum:.2f} "
            "Normalised residual="
            f"{np.abs(tree_sum - loop_sum) / loop_sum * 100:.4f}"
        )

    xs = np.array(xs)
    sinds = np.argsort(xs)
    xs = xs[sinds]
    loop_ys = np.array(loop_ys)[sinds]
    tree_ys = np.array(tree_ys)[sinds]
    precision = np.array(precision)[sinds]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    ax.grid()

    ax.plot(xs, loop_ys, label="Loop")
    ax.plot(xs, tree_ys, label="Tree")

    ax.set_ylabel("Wallclock (s)")
    ax.set_xlabel(r"$N_\star N_\mathrm{gas}$")

    ax.legend()

    plt.show()
    # fig.savefig("../los_timing_nstar%d.png" % n,
    #     dpi=100, bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx()
    ax.grid()

    ax.plot(xs, precision, label="Loop / Tree")

    ax.set_ylabel(
        r"$|\tau_{V, tree} - \tau_{V, loop}|" r" / \tau_{V, loop}$ (%)"
    )
    ax.set_xlabel("$N_\\star N_\\mathrm{gas}$")

    ax.legend()

    # fig.savefig("../los_precision_nstar%d.png" % n,
    #     dpi=100, bbox_inches="tight")
    plt.show()
