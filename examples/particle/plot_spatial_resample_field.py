"""
Deterministic field-mode spatial resampling
===========================================

Demonstrates :meth:`~synthesizer.particle.gas.Gas.spatially_resample` and
:meth:`~synthesizer.particle.stars.Stars.spatially_resample` with
``method="field"``.

The example compares the existing stochastic resampling path
(``method="random"``) to the new deterministic field-aware path
(``method="field"``) on a CAMELS-IllustrisTNG galaxy. The resulting maps show
how field mode preserves the user-facing ``resample_factor`` contract while
removing random spatial jitter and interpolating intensive quantities from the
local SPH field.

Rows show gas-mass maps, stellar-mass maps, and gas metallicity maps. Columns
show the original galaxy, random resampling, and field resampling.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Mpc

from synthesizer import TEST_DATA_DIR, check_atomic_timing
from synthesizer.kernel_functions import Kernel
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.particle.galaxy import Galaxy
from synthesizer.utils.operation_timers import OperationTimers

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

timers = OperationTimers()
if check_atomic_timing():
    timers.reset()


def _prepare_image(image, fallback_fov, floor=1e-12):
    """Extract a plain numpy array and field of view from an image object."""
    if hasattr(image, "arr"):
        arr = image.arr.copy()
        fov = float(image._fov[0])
    else:
        arr = np.asarray(image).copy()
        fov = fallback_fov

    return np.maximum(arr, floor), fov


# ---------------------------------------------------------------------------
# Load the CAMELS test galaxy and centre it on the stellar distribution.
# ---------------------------------------------------------------------------
galaxies = load_CAMELS_IllustrisTNG(
    TEST_DATA_DIR,
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    group_dir=TEST_DATA_DIR,
)
original = galaxies[0]

centre = original.stars.coordinates.mean(axis=0)
original.stars.centre = centre
original.gas.centre = centre

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FACTOR = 10
FOV = 0.025  # Mpc
RESOLUTION = 0.0002  # Mpc
KERNEL = Kernel("cubic")

# ---------------------------------------------------------------------------
# Resample the galaxy under both modes.
# ---------------------------------------------------------------------------
random_gas = original.gas.spatially_resample(
    FACTOR,
    kernel=KERNEL,
    seed=42,
    method="random",
)
random_stars = original.stars.spatially_resample(
    FACTOR,
    kernel=KERNEL,
    seed=42,
    method="random",
)
random_gas.calculate_smoothing_lengths()
random_stars.calculate_smoothing_lengths()

field_gas = original.gas.spatially_resample(
    FACTOR,
    kernel=KERNEL,
    method="field",
)
field_stars = original.stars.spatially_resample(
    FACTOR,
    kernel=KERNEL,
    method="field",
)
field_gas.calculate_smoothing_lengths()
field_stars.calculate_smoothing_lengths()

gal_random = Galaxy(
    name="random",
    gas=random_gas,
    stars=random_stars,
    redshift=original.redshift,
    centre=centre,
)
gal_field = Galaxy(
    name="field",
    gas=field_gas,
    stars=field_stars,
    redshift=original.redshift,
    centre=centre,
)

# ---------------------------------------------------------------------------
# Generate images for the original and both resampling modes.
# ---------------------------------------------------------------------------
img_kw = dict(
    resolution=RESOLUTION * Mpc,
    fov=FOV * Mpc,
    img_type="smoothed",
    kernel=KERNEL,
)

datasets = [
    ("Original", original),
    (f"Random (x{FACTOR})", gal_random),
    (f"Field (x{FACTOR})", gal_field),
]

gas_mass_maps = []
stellar_mass_maps = []
gas_metallicity_maps = []

for _, galaxy in datasets:
    gas_mass_maps.append(
        _prepare_image(galaxy.get_map_gas_mass(**img_kw), fallback_fov=FOV)
    )
    stellar_mass_maps.append(
        _prepare_image(galaxy.get_map_stellar_mass(**img_kw), fallback_fov=FOV)
    )
    gas_metallicity_maps.append(
        _prepare_image(
            galaxy.get_map_gas_metallicity(**img_kw),
            fallback_fov=FOV,
            floor=1e-6,
        )
    )

# ---------------------------------------------------------------------------
# Plot the comparison.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    3,
    3,
    figsize=(9.0, 8.6),
    gridspec_kw=dict(wspace=0.04, hspace=0.08),
)

rows = [
    ("Gas mass", gas_mass_maps, "magma", "log"),
    ("Stellar mass", stellar_mass_maps, "viridis", "log"),
    ("Gas metallicity", gas_metallicity_maps, "cividis", None),
]

for col, (title, _) in enumerate(datasets):
    for row, (ylabel, maps, cmap, norm) in enumerate(rows):
        ax = axes[row, col]
        arr, fov = maps[col]
        ax.imshow(
            arr,
            origin="lower",
            extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
            aspect="equal",
            cmap=cmap,
            norm=norm,
        )
        if row == 0:
            ax.set_title(title, fontsize=11, pad=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle(
    "Random versus deterministic field-mode spatial resampling",
    fontsize=13,
    y=0.98,
)
plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])

if check_atomic_timing():
    print("\nOperationTimers summary:\n")
    timers.print_table()

plt.show()
