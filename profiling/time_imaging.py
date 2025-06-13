"""Time creating images."""

import time

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, km, kpc, s

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.instruments.filters import UVJ
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths

# Define the grid
grid = Grid("test_grid")

# Define the emission model
model = IncidentEmission(grid)

# Convert the model to a per particle model
model.set_per_particle(True)


# define the grid (normally this would be defined by an SPS grid)
log10ages = grid.log10age
metallicities = grid.metallicity

# define the parameters of the star formation and metal enrichment histories

Z_p = {"metallicity": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)

sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(**sfh_p)  # constant star formation
sfzh = ParametricStars(
    log10ages,
    metallicities,
    sf_hist=sfh,
    metal_dist=metal_dist,
    initial_mass=10**9 * Msun,
)
print(sfzh)

# Get the SPH kernel
sph_kernel = Kernel()
kernel_data = sph_kernel.get_kernel()

fov = 100 * kpc

filters = UVJ()
centre = np.array([0, 0, 0]) * kpc


# --- create stars object
# Dict to hold times
runtimes_part = []
ns = [10**2, 10**3, 10**4, 5 * 10**4]
res = [0.01 * kpc, 0.1 * kpc, 1 * kpc, 10 * kpc]
for N in ns:
    print(f"Running for N = {N}")
    # Generate some fake velocities
    vels = np.random.normal(100, 500, N) * km / s

    # Generate random coordinates
    coords = np.random.uniform(-fov / 2, fov / 2, (N, 3)) * kpc

    stars = sample_sfzh(
        sfzh.sfzh,
        sfzh.log10ages,
        sfzh.log10metallicities,
        N,
        velocities=vels,
        coordinates=coords,
        centre=centre,
        smoothing_lengths=calculate_smoothing_lengths(coords),
    )

    galaxy = Galaxy(stars=stars, centre=centre)

    # Get the per particle spectra
    galaxy.stars.get_spectra(model)
    galaxy.get_photo_lnu(filters)

    # Get the image
    start = time.time()
    smooth_imgs = galaxy.get_images_luminosity(
        0.1 * kpc,
        fov=fov,
        emission_model=model,
        img_type="smoothed",
        kernel=kernel_data,
        kernel_threshold=1,
    )
    runtimes_part.append(time.time() - start)

    fixed_res = smooth_imgs.npix[0]


N = 10**4

# Generate some fake velocities
vels = np.random.normal(100, 500, N) * km / s

# Generate random coordinates
coords = np.random.uniform(-fov / 2, fov / 2, (N, 3)) * kpc

stars = sample_sfzh(
    sfzh.sfzh,
    sfzh.log10ages,
    sfzh.log10metallicities,
    N,
    velocities=vels,
    coordinates=coords,
    smoothing_lengths=calculate_smoothing_lengths(coords),
    centre=centre,
)

galaxy = Galaxy(stars=stars)

# Get the per particle spectra
galaxy.stars.get_spectra(model)
galaxy.get_photo_lnu(filters)

# Loop over resolutions
runtimes_res = []
npixs = []
for r in res:
    print(f"Running for res = {r}")
    start = time.time()
    imgs = galaxy.get_images_luminosity(
        r,
        fov=fov,
        emission_model=model,
        img_type="smoothed",
        kernel=kernel_data,
        kernel_threshold=1,
    )
    runtimes_res.append(time.time() - start)
    npixs.append(imgs.npix[0])

# Plot the timings
fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel(r"$N$")
ax.set_ylabel("Time [s]")

ax.loglog(
    ns,
    runtimes_part,
    label=r"$N_\star$ ($N_{\rm pix}=$" f"{fixed_res})",
    marker="o",
)
ax.loglog(
    npixs, runtimes_res, label=r"$N_{\rm pix}$ ($N_\star=$" f"{N})", marker="o"
)

ax.legend()

fig.savefig("time_imaging.png", dpi=300, bbox_inches="tight")
