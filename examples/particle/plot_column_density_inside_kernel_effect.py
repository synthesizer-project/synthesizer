"""
Point LOS Column Densities With A Truncated Source Kernel
=========================================================

This example demonstrates the effect of ignoring the effects of an input
particle (e.g. star) lying within a source particle's (e.g. gas) kernel when
calculating line-of-sight column densities and treating the input as a point
source.

This is done by placing particles throughout the kernel of a foreground gas
particle and comparing the true LOS columns to a simple point-particle
approximation that snaps stars in front of the gas centre to fully in front of
the kernel and stars behind the gas centre to fully behind it.

This example demonstrates the importance of a change implemented to the
LOS column density calculation in v1.1.0.
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
        coordinates=np.array([[0.0, 0.0, 1.2]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )


nstar = 1000
kernel = Kernel(binsize=128)
gas = make_gas()
gas_z = gas.coordinates[0, 2].value
approx_offset = 1.1

# Draw stars uniformly throughout the spherical kernel support so the example
# compares stars in front of and behind the gas-centre plane at matched random
# positions within the same kernel volume.
rng = np.random.default_rng(42)
offsets = []
while len(offsets) < nstar:
    trial = rng.uniform(-1.0, 1.0, size=(nstar, 3))
    trial = trial[np.sum(trial**2, axis=1) <= 1.0]
    offsets.extend(trial.tolist())
offsets = np.asarray(offsets[:nstar])

# Shift the sampled offsets onto the gas-particle centre. These are the true
# star positions inside the gas kernel.
inside_positions = offsets.copy()
inside_positions[:, 2] += gas_z

# Build a simple point-particle approximation by snapping stars on the observer
# side of the gas centre to fully in front of the kernel and stars on the far
# side to fully behind it.
front_mask = inside_positions[:, 2] < gas_z
behind_mask = ~front_mask

approx_positions = inside_positions.copy()
approx_positions[front_mask, 2] = gas_z - approx_offset
approx_positions[behind_mask, 2] = gas_z + approx_offset

# Also build a matched full-column reference by placing every star fully behind
# the gas kernel at the same projected separation.
full_column_positions = inside_positions.copy()
full_column_positions[:, 2] = gas_z + approx_offset

stars_inside = make_stars(inside_positions)
stars_full = make_stars(full_column_positions)

gal_inside = Galaxy("inside", stars=stars_inside, gas=gas, redshift=0.0)

col_den_inside = stars_inside.get_los_column_density(
    gas,
    "dust_masses",
    kernel,
    force_loop=1,
    min_count=10,
)
col_den_full = stars_full.get_los_column_density(
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

inside_fraction = (col_den_inside / col_den_full).value

print(f"Median inside-kernel column density: {np.median(col_den_inside):.4e}")
print(f"Median inside-kernel tau_v: {np.median(tau_inside):.4e}")
print(f"Median true column fraction: {np.median(inside_fraction):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the LOS geometry in the x-z plane so the full spherical support is
# visible directly.
boundary_x = np.linspace(-1.0, 1.0, 256)
boundary_z = np.sqrt(1.0 - boundary_x**2)

axes[0].fill_between(
    boundary_x,
    gas_z - boundary_z,
    gas_z + boundary_z,
    color="#8ecae6",
    alpha=0.35,
    label="Gas kernel support",
)
axes[0].axhline(
    gas_z,
    color="#023047",
    lw=1.2,
    ls=":",
    label="Gas-centre plane",
)
axes[0].scatter(
    inside_positions[front_mask, 0],
    inside_positions[front_mask, 2],
    s=24,
    marker="*",
    alpha=0.75,
    color="#d62828",
    label="Real stars inside: observer side",
)
axes[0].scatter(
    inside_positions[behind_mask, 0],
    inside_positions[behind_mask, 2],
    s=24,
    marker="*",
    alpha=0.75,
    color="#2a9d8f",
    label="Real stars inside: far side",
)
axes[0].scatter(
    approx_positions[front_mask, 0],
    approx_positions[front_mask, 2],
    s=26,
    marker="*",
    facecolors="none",
    edgecolors="#9d0208",
    linewidths=0.9,
    alpha=0.9,
    label="Point approximation: fully in front",
)
axes[0].scatter(
    approx_positions[behind_mask, 0],
    approx_positions[behind_mask, 2],
    s=26,
    marker="*",
    facecolors="none",
    edgecolors="#1b7f6b",
    linewidths=0.9,
    alpha=0.9,
    label="Point approximation: fully behind",
)
axes[0].scatter(
    [0.0],
    [gas_z],
    color="#023047",
    s=90,
    marker="o",
    label="Gas particle",
)
axes[0].set_xlim(-1.55, 1.55)
axes[0].set_ylim(gas_z - 1.55, gas_z + 1.55)
axes[0].set_aspect("equal")
axes[0].set_xlabel("x [Mpc]")
axes[0].set_ylabel("LOS z [Mpc]")
axes[0].set_title("Real stars and the point-particle approximation")

# Compare the true fractional column to the simplified centre-based point
# approximation.
axes[1].scatter(
    inside_positions[front_mask, 2] - gas_z,
    inside_fraction[front_mask],
    s=18,
    marker="*",
    alpha=0.75,
    color="#d62828",
    label="Truncated LOS",
)
axes[1].scatter(
    inside_positions[behind_mask, 2] - gas_z,
    inside_fraction[behind_mask],
    s=18,
    marker="*",
    alpha=0.75,
    color="#2a9d8f",
)
axes[1].step(
    [-1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    where="post",
    color="#264653",
    lw=2.0,
    ls="--",
    label="Full column",
)
axes[1].set_xlim(-1.05, 1.05)
axes[1].set_ylim(-0.02, 1.05)
axes[1].set_xlabel(
    r"Relative LOS position $(z_\star - z_{\rm gas}) / h_{\rm gas}$"
)
axes[1].set_ylabel("LOS column / full foreground column")
axes[1].set_title("Continuous truth versus step-function approximation")
axes[1].legend(loc="upper left")

plt.tight_layout()
plt.show()
