"""
Spatial resampling of gas and star particles
==============================================

Demonstrate the :meth:`~synthesizer.particle.gas.Gas.spatially_resample`
and :meth:`~synthesizer.particle.stars.Stars.spatially_resample`
methods on a CAMELS-IllustrisTNG galaxy.

The 4 rows show gas (hist), stars (hist), gas (smoothed), stars (smoothed)
surface-density maps. Columns correspond to the original resolution and
resampling factors of 5, 10, and 100.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Mpc

from synthesizer import TEST_DATA_DIR
from synthesizer.kernel_functions import Kernel
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.particle.galaxy import Galaxy

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

np.random.seed(42)

# ---------------------------------------------------------------------------
# Load the CAMELS test galaxy and centre the coordinates
# ---------------------------------------------------------------------------
galaxies = load_CAMELS_IllustrisTNG(
    TEST_DATA_DIR,
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    group_dir=TEST_DATA_DIR,
)
original = galaxies[0]

star_centre = original.stars.coordinates.mean(axis=0)
original.stars.centre = star_centre
original.gas.centre = star_centre

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
resample_factors = [5, 10, 100]

fov_val = 0.02
resolution_val = 0.0002

kernel = Kernel("cubic")

# ---------------------------------------------------------------------------
# Assemble galaxies: original + resampled
# ---------------------------------------------------------------------------
galaxy_list = [original]

for factor in resample_factors:
    resampled_gas = original.gas.spatially_resample(
        factor,
        seed=42,
        kernel=kernel,
    )
    resampled_gas.calculate_smoothing_lengths()
    resampled_stars = original.stars.spatially_resample(
        factor,
        seed=42,
        kernel=kernel,
    )
    resampled_stars.calculate_smoothing_lengths()
    gal = Galaxy(
        name=f"Resample factor={factor}",
        stars=resampled_stars,
        gas=resampled_gas,
        redshift=original.redshift,
        centre=star_centre,
    )
    galaxy_list.append(gal)

# ---------------------------------------------------------------------------
# Generate mass maps: 2 img_types x 2 components x 4 galaxies
# ---------------------------------------------------------------------------
labels = ["Original", "Factor=5", "Factor=10", "Factor=100"]

gas_hist = []
star_hist = []
gas_smooth = []
star_smooth = []

for gal in galaxy_list:
    gas_hist.append(
        gal.get_map_gas_mass(
            resolution=resolution_val * Mpc,
            fov=fov_val * Mpc,
            img_type="hist",
        )
    )
    star_hist.append(
        gal.get_map_stellar_mass(
            resolution=resolution_val * Mpc,
            fov=fov_val * Mpc,
            img_type="hist",
        )
    )
    gas_smooth.append(
        gal.get_map_gas_mass(
            resolution=resolution_val * Mpc,
            fov=fov_val * Mpc,
            img_type="smoothed",
            kernel=kernel,
        )
    )
    star_smooth.append(
        gal.get_map_stellar_mass(
            resolution=resolution_val * Mpc,
            fov=fov_val * Mpc,
            img_type="smoothed",
            kernel=kernel,
        )
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
row_specs = [
    ("Gas (hist)", gas_hist),
    ("Stars (hist)", star_hist),
    ("Gas (smoothed)", gas_smooth),
    ("Stars (smoothed)", star_smooth),
]

fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 7),
    gridspec_kw=dict(wspace=0, hspace=0),
)

for col in range(4):
    for row, (title_prefix, maps) in enumerate(row_specs):
        ax = axes[row, col]
        im = maps[col]
        if hasattr(im, "arr"):
            arr = im.arr.copy()
            fv = float(im._fov[0])
        else:
            arr = np.asarray(im).copy()
            fv = fov_val
        arr = np.maximum(arr, 1e-10)
        ax.imshow(
            arr,
            origin="lower",
            extent=[-fv / 2, fv / 2, -fv / 2, fv / 2],
            aspect="equal",
            norm="log",
            cmap="viridis",
        )
        if row == 0:
            ax.set_title(labels[col], fontsize=10)
        if col == 0:
            ax.set_ylabel(title_prefix, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
