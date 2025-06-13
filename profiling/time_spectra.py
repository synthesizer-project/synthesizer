"""Time creating different types of spectra."""

import time

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, km, s

from synthesizer.emission_models import IntrinsicEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.stars import sample_sfzh

# Define the grid
grid = Grid("test_grid")

# Define the emission model
model = IntrinsicEmission(grid, fesc=0.0)

# define the grid (normally this would be defined by an SPS grid)
log10ages = grid.log10age
metallicities = grid.metallicity
print(
    f"Grid has {len(log10ages)} ages and {len(metallicities)} metallicities."
)

# define the parameters of the star formation and metal enrichment histories
Z_p = {"metallicity": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)
print(f"Metallicity distribution: {metal_dist}")
sfh_p = {"duration": 100 * Myr}
print(f"Star formation history: {sfh_p}")
sfh = SFH.Constant(**sfh_p)  # constant star formation
print(f"SFH: {sfh}")

sfzh = ParametricStars(
    log10ages,
    metallicities,
    sf_hist=sfh,
    metal_dist=metal_dist,
    initial_mass=10**9 * Msun,
)
print(sfzh)
print(
    f"SFZH has {len(sfzh.sfzh)} ages and "
    f"{len(sfzh.log10metallicities)} metallicities."
)


# Time bog standard parametric spectra
start = time.time()
sfzh.get_spectra(model)
param_time = time.time() - start
print(f"Time for parametric spectra: {param_time:.2f} s")


# --- create stars object
# Dict to hold times
runtimes = {}
ns = [10**2, 10**3, 10**4, 5 * 10**4, 10**5]
for N in ns:
    print(f"Running for N = {N}")
    # Generate some fake velocities
    vels = np.random.normal(100, 500, N) * km / s

    stars = sample_sfzh(
        sfzh.sfzh,
        sfzh.log10ages,
        sfzh.log10metallicities,
        N,
        velocities=vels,
    )

    # --- create galaxy object

    galaxy = Galaxy(stars=stars)

    # Turn off per particle spectra
    model.set_per_particle(False)

    # Get the integrated spectrum
    start = time.time()
    galaxy.stars.get_spectra(model)
    runtimes.setdefault("integrated", []).append(time.time() - start)

    # Convert the model to a per particle model
    model.set_per_particle(True)

    # Get the per particle spectra
    start = time.time()
    galaxy.stars.get_spectra(model)
    runtimes.setdefault("per_particle", []).append(time.time() - start)

    # And include a velocity shift
    start = time.time()
    galaxy.stars.get_spectra(model, vel_shift=True)
    runtimes.setdefault("velocity_shifted", []).append(time.time() - start)
    model.set_vel_shift(False)

# Plot the timings
fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel(r"$N_\star$")
ax.set_ylabel("Time [s]")

for key, val in runtimes.items():
    ax.loglog(ns, val, label=key, marker="o")

ax.axhline(param_time, color="k", linestyle="--", label="parametric")
ax.legend()

fig.savefig("time_imaging.png", dpi=300, bbox_inches="tight")
