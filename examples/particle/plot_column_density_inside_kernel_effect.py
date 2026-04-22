"""
Point LOS Inside A Source Kernel
================================

This example compares point-particle LOS column densities for two matched
stellar samples:

1. stars placed inside the foreground gas kernel,
2. stars placed fully behind the same gas kernel.

The inside-kernel sample only sees part of the source along the line of sight,
while the fully-behind sample sees the whole projected source column.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Mpc, Msun, Myr

from synthesizer.kernel_functions import Kernel
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def make_stars(positions):
    """Construct a Stars object at the supplied coordinates."""
    nstars = positions.shape[0]
    stars = Stars(
        initial_masses=np.ones(nstars) * Msun,
        ages=np.ones(nstars) * 10.0 * Myr,
        metallicities=np.full(nstars, 0.02),
        redshift=0.0,
        tau_v=np.zeros(nstars),
        coordinates=positions * Mpc,
    )
    stars.smoothing_lengths = np.ones(nstars) * Mpc
    return stars


def make_gas():
    """Construct a single foreground gas particle."""
    return Gas(
        masses=np.array([1e6]) * Msun,
        metallicities=np.array([0.01]),
        redshift=0.0,
        coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )


np.random.seed(42)

nstar = 1000
kernel = Kernel(name="uniform", binsize=128)
gas = make_gas()

# Draw a shared set of projected x/y offsets uniformly across the kernel
# diameter so the geometry plot can show the full source support directly in an
# x-z slice.
rng = np.random.default_rng(42)
xy = rng.uniform(-0.85, 0.85, size=(nstar, 2))

# Place one sample inside the gas kernel and another well behind it.
z_inside = rng.uniform(-0.8, 0.8, size=nstar)
z_behind = rng.uniform(1.4, 2.2, size=nstar)

inside_positions = np.column_stack((xy, z_inside))
behind_positions = np.column_stack((xy, z_behind))

stars_inside = make_stars(inside_positions)
stars_behind = make_stars(behind_positions)

gal_inside = Galaxy("inside", stars=stars_inside, gas=gas, redshift=0.0)
gal_behind = Galaxy("behind", stars=stars_behind, gas=gas, redshift=0.0)

col_den_inside = stars_inside.get_los_column_density(
    gas,
    "dust_masses",
    kernel,
    force_loop=1,
    min_count=10,
)
col_den_behind = stars_behind.get_los_column_density(
    gas,
    "dust_masses",
    kernel,
    force_loop=1,
    min_count=10,
)

tau_inside = gal_inside.get_stellar_los_tau_v(
    kappa=0.07,
    kernel=kernel,
    force_loop=1,
    min_count=10,
)
tau_behind = gal_behind.get_stellar_los_tau_v(
    kappa=0.07,
    kernel=kernel,
    force_loop=1,
    min_count=10,
)

print(f"Median inside-kernel column density: {np.median(col_den_inside):.4e}")
print(f"Median fully-behind column density: {np.median(col_den_behind):.4e}")
print(f"Median inside-kernel tau_v: {np.median(tau_inside):.4e}")
print(f"Median fully-behind tau_v: {np.median(tau_behind):.4e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the LOS geometry in the x-z plane so the full spherical support is
# visible directly.
boundary_x = np.linspace(-1.0, 1.0, 256)
boundary_z = np.sqrt(1.0 - boundary_x**2)

axes[0].fill_between(
    boundary_x,
    -boundary_z,
    boundary_z,
    color="tab:blue",
    alpha=0.2,
    label="Gas kernel",
)
axes[0].scatter(
    inside_positions[:, 0],
    z_inside,
    s=6,
    alpha=0.5,
    label="Stars inside",
)
axes[0].scatter(
    behind_positions[:, 0],
    z_behind,
    s=6,
    alpha=0.5,
    label="Stars behind",
)
axes[0].annotate(
    "Line of sight",
    xy=(0.65, 1.1),
    xytext=(0.65, 2.0),
    arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "k"},
    ha="center",
)
axes[0].set_xlim(-1.1, 1.1)
axes[0].set_ylim(-1.1, 2.3)
axes[0].set_xlabel("x [Mpc]")
axes[0].set_ylabel("z [Mpc]")
axes[0].set_title("LOS geometry")
axes[0].legend(loc="upper left")

# Compare the measured column densities for the two matched samples.
axes[1].scatter(col_den_behind, col_den_inside, s=8, alpha=0.6)
max_col = max(np.max(col_den_inside), np.max(col_den_behind))
axes[1].plot([0.0, max_col], [0.0, max_col], "k--", lw=1)
axes[1].set_xlabel("Fully-behind column density")
axes[1].set_ylabel("Inside-kernel column density")
axes[1].set_title("Partial LOS path lowers the column density")

plt.tight_layout()
plt.show()
