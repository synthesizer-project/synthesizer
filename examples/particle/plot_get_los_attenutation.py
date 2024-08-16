"""
Plot line of sight optical depth calculations
=============================================

This example shows how to compute line of sight optical depths and compares
the simple loop method with the complex tree method.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from unyt import Myr

from synthesizer.grid import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfhz


def calculate_smoothing_lengths(positions, num_neighbors=56):
    """Calculate the SPH smoothing lengths for a set of coordinates."""
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=num_neighbors + 1)

    # The k-th nearest neighbor distance (k = num_neighbors)
    kth_distances = distances[:, num_neighbors]

    # Set the smoothing length to the k-th nearest neighbor
    # distance divided by 2.0
    smoothing_lengths = kth_distances / 2.0

    return smoothing_lengths


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

n = 100

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
stars = sample_sfhz(
    param_stars.sfzh,
    param_stars.log10ages,
    param_stars.log10metallicities,
    n,
    coordinates=coords,
    current_masses=np.full(n, 10**8.7 / n),
    smoothing_lengths=smls,
    redshift=1,
)

ngas = 1000

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
loop_tau_v = galaxy.calculate_los_tau_v(
    kappa=0.07,
    kernel=kernel,
    force_loop=1,
)
loop_time = time.time() - start
loop_sum = np.sum(loop_tau_v)

# Calculate the tau_vs
start = time.time()
tree_tau_v = galaxy.calculate_los_tau_v(
    kappa=0.07,
    kernel=kernel,
    min_count=100,
)
tree_time = time.time() - start
tree_sum = np.sum(tree_tau_v)

print(
    f"LOS calculation with tree took {tree_time:.4f} "
    f"seconds for nstar={n} and ngas={ngas}"
)
print(
    f"LOS calculation with loop took {loop_time:.4f} "
    f"seconds for nstar={n} and ngas={ngas}"
)
print(
    "Ratio in wallclock: " f"Time_loop/Time_tree={loop_time / tree_time:.4f}"
)
print(
    f"Tree gave={tree_sum:.2e} Loop gave={loop_sum:.2e} "
    "Normalised residual="
    f"{np.abs(tree_sum - loop_sum) / loop_sum * 100:.4f}"
)
print()

# Plot the optical depths against each other.
fig, ax = plt.subplots()
ax.scatter(tree_tau_v, loop_tau_v, s=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Tree method")
ax.set_ylabel("Loop method")

plt.show()
