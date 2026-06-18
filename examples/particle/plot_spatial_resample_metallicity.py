"""
Spatial resampling with metallicity scatter
===========================================

Demonstrates :meth:`~synthesizer.particle.gas.Gas.spatially_resample`
with ``attr_modes={"metallicities": ("normal", sigma)}`` to scatter gas
and stellar metallicities around their parent-particle values.

A CAMELS-IllustrisTNG galaxy is resampled by a factor of 10 under two
modes — duplicated (default) and normal with a large fixed σ — and the
resulting gas-mass maps and metallicity histograms are compared.

The ``"normal"`` mode with a large σ introduces visible scatter in the
metallicity distribution while still preserving the global shape and
the underlying mass–metallicity relation.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from unyt import Mpc

from synthesizer import TEST_DATA_DIR
from synthesizer.kernel_functions import Kernel
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.particle.galaxy import Galaxy

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# ---------------------------------------------------------------------------
# Load CAMELS test galaxy
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

FACTOR = 10
FOV = 0.025  # Mpc
RESOLUTION = 0.0002  # Mpc
KERNEL = Kernel("cubic")
SIGMA_Z = 1.0  # large fixed sigma to make scatter visible

# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------
g_dup = original.gas.spatially_resample(FACTOR, seed=42, kernel=KERNEL)
s_dup = original.stars.spatially_resample(FACTOR, seed=42, kernel=KERNEL)
g_dup.calculate_smoothing_lengths()
s_dup.calculate_smoothing_lengths()

g_norm = original.gas.spatially_resample(
    FACTOR,
    seed=42,
    kernel=KERNEL,
    attr_modes={"metallicities": ("normal", SIGMA_Z)},
)
s_norm = original.stars.spatially_resample(
    FACTOR,
    seed=42,
    kernel=KERNEL,
    attr_modes={"metallicities": ("normal", SIGMA_Z)},
)
g_norm.calculate_smoothing_lengths()
s_norm.calculate_smoothing_lengths()

gal_orig = original
gal_dup = Galaxy(
    name="dup",
    stars=s_dup,
    gas=g_dup,
    redshift=original.redshift,
    centre=star_centre,
)
gal_norm = Galaxy(
    name="norm",
    stars=s_norm,
    gas=g_norm,
    redshift=original.redshift,
    centre=star_centre,
)

# ---------------------------------------------------------------------------
# Generate gas-mass smoothed images
# ---------------------------------------------------------------------------
img_kw = dict(
    resolution=RESOLUTION * Mpc,
    fov=FOV * Mpc,
    img_type="smoothed",
    kernel=KERNEL,
)


def _prep(im, floor=1e-10):
    arr = im.arr.copy() if hasattr(im, "arr") else np.asarray(im).copy()
    fv = float(im._fov[0]) if hasattr(im, "_fov") else FOV
    return np.maximum(arr, floor), fv


gas_mass_orig, fv = _prep(gal_orig.get_map_gas_mass(**img_kw))
gas_mass_dup, _ = _prep(gal_dup.get_map_gas_mass(**img_kw))
gas_mass_norm, _ = _prep(gal_norm.get_map_gas_mass(**img_kw))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 3, figure=fig, wspace=0.02)
ext = [-fv / 2, fv / 2, -fv / 2, fv / 2]

for col, (arr, title) in enumerate(
    [
        (gas_mass_orig, "Original"),
        (gas_mass_dup, f"Duplicated (x{FACTOR})"),
        (gas_mass_norm, f"Normal σ={SIGMA_Z} (x{FACTOR})"),
    ]
):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(
        arr,
        origin="lower",
        extent=ext,
        aspect="equal",
        norm="log",
        cmap="magma",
    )
    ax.set_title(title, fontsize=11)
    if col == 0:
        ax.set_ylabel("Gas mass density", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# ---------------------------------------------------------------------------
# Second figure: metallicity histograms + mass–Z hexbin
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(
    1, 3, figsize=(14, 4.5), gridspec_kw=dict(wspace=0.35)
)

z_bins = np.linspace(0, 0.04, 50)

# Gas metallicity
ax = axes[0]
ax.hist(
    original.gas.metallicities,
    bins=z_bins,
    color="grey",
    alpha=0.4,
    label="Original",
    density=True,
)
ax.hist(
    g_dup.metallicities,
    bins=z_bins,
    color="steelblue",
    histtype="step",
    lw=2.5,
    label=f"Duplicated (x{FACTOR})",
    density=True,
)
ax.hist(
    g_norm.metallicities,
    bins=z_bins,
    color="darkorange",
    histtype="step",
    lw=2.5,
    label=f"Normal σ={SIGMA_Z} (x{FACTOR})",
    density=True,
)
ax.set_xlabel("Gas metallicity")
ax.set_ylabel("PDF")
ax.set_title("Gas metallicity")
ax.legend(fontsize=7.5)

# Stellar metallicity
ax = axes[1]
ax.hist(
    original.stars.metallicities,
    bins=z_bins,
    color="grey",
    alpha=0.4,
    label="Original",
    density=True,
)
ax.hist(
    s_dup.metallicities,
    bins=z_bins,
    color="steelblue",
    histtype="step",
    lw=2.5,
    label=f"Duplicated (x{FACTOR})",
    density=True,
)
ax.hist(
    s_norm.metallicities,
    bins=z_bins,
    color="darkorange",
    histtype="step",
    lw=2.5,
    label=f"Normal σ={SIGMA_Z} (x{FACTOR})",
    density=True,
)
ax.set_xlabel("Stellar metallicity")
ax.set_ylabel("PDF")
ax.set_title("Stellar metallicity")
ax.legend(fontsize=7.5)

# Mass–metallicity hexbin for normal-mode gas
ax = axes[2]
h = ax.hexbin(
    np.log10(g_norm.masses.value.clip(1e-10)),
    g_norm.metallicities.clip(1e-6),
    gridsize=35,
    cmap="magma_r",
    mincnt=1,
    bins="log",
)
plt.colorbar(h, ax=ax, label="N particles", shrink=0.8)
ax.set_xlabel(r"$\log_{10}(M_\mathrm{gas} / M_\odot)$")
ax.set_ylabel("Gas Z  (normal mode)")
ax.set_title("Normal-mode mass–metallicity")

plt.show()
