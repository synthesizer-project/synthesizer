"""
Spatial resampling with parametric SFH and MetalDist
=====================================================

Demonstrates resampling stellar ages and metallicities from
parametrised star-formation histories and metallicity distributions
using :meth:`~synthesizer.particle.stars.Stars.spatially_resample`.

A toy stellar population of 100 particles is resampled by a factor of
50 (→ 5 000 particles) under three scenarios:

1. **sfh only** — ages are sampled from a declining exponential SFH;
   metallicities are kept from the original particles.
2. **metal_dist only** — metallicities are sampled from a Gaussian
   MetalDist; ages are kept from the original particles.
3. **sfh + metal_dist** — ages and metallicities are jointly sampled
   from the 2-D SFZH formed by the outer product of the two
   distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Gyr, Mpc, Msun, Myr, km, s

from synthesizer.kernel_functions import Kernel
from synthesizer.parametric.metal_dist import Normal as ZNormal
from synthesizer.parametric.sf_hist import TruncatedExponential
from synthesizer.particle.stars import Stars

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# ---------------------------------------------------------------------------
# Toy data: 100 stellar particles with a basic age–metallicity distribution
# ---------------------------------------------------------------------------
NP = 100
FACTOR = 50
rng = np.random.default_rng(42)

stars = Stars(
    initial_masses=np.ones(NP) * 1e6 * Msun,
    ages=rng.uniform(10, 10000, NP) * Myr,
    metallicities=rng.uniform(0.001, 0.03, NP),
    coordinates=rng.uniform(-1, 1, (NP, 3)) * Mpc,
    velocities=rng.uniform(-50, 50, (NP, 3)) * (km / s),
    smoothing_lengths=np.ones(NP) * 0.3 * Mpc,
    softening_lengths=np.ones(NP) * 0.1 * Mpc,
    redshift=0.1,
)

# ---------------------------------------------------------------------------
# Parametric SFH and MetalDist
# ---------------------------------------------------------------------------
sfh = TruncatedExponential(tau=3 * Gyr, max_age=15 * Gyr, min_age=0 * Gyr)
metal_dist = ZNormal(mean=0.02, sigma=0.005)

# ---------------------------------------------------------------------------
# Resample under each scenario
# ---------------------------------------------------------------------------
kernel = Kernel("cubic")

s_sfh = stars.spatially_resample(
    FACTOR,
    kernel=kernel,
    seed=1,
    sfh=sfh,
)

s_md = stars.spatially_resample(
    FACTOR,
    kernel=kernel,
    seed=1,
    metal_dist=metal_dist,
)

s_both = stars.spatially_resample(
    FACTOR,
    kernel=kernel,
    seed=1,
    sfh=sfh,
    metal_dist=metal_dist,
)

# ---------------------------------------------------------------------------
# Plot age and metallicity histograms
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    3, 2, figsize=(11, 9), gridspec_kw=dict(wspace=0.3, hspace=0.3)
)

age_bins = np.logspace(np.log10(1), np.log10(15000), 60)
z_bins = np.linspace(0.0001, 0.04, 40)

for row_idx, (label, samp) in enumerate(
    [
        ("sfh only", s_sfh),
        ("metal_dist only", s_md),
        ("sfh + metal_dist", s_both),
    ]
):
    ax_age = axes[row_idx, 0]
    ax_age.hist(
        stars.ages.value,
        bins=age_bins,
        color="grey",
        alpha=0.3,
        label=f"Original ({len(stars.ages)} particles)",
        density=False,
    )
    ax_age.hist(
        samp.ages.value,
        bins=age_bins,
        color="tab:blue",
        alpha=0.6,
        label=f"Resampled ({len(samp.ages)} particles)",
        density=False,
    )
    ax_age.set_xscale("log")
    ax_age.set_xlabel("Age / yr")
    ax_age.set_ylabel("N")
    ax_age.set_title(f"Ages — {label}")
    ax_age.legend(fontsize=7.5)

    ax_z = axes[row_idx, 1]
    ax_z.hist(
        stars.metallicities,
        bins=z_bins,
        color="grey",
        alpha=0.3,
        label="Original",
        density=False,
    )
    ax_z.hist(
        samp.metallicities,
        bins=z_bins,
        color="tab:orange",
        alpha=0.6,
        label="Resampled",
        density=False,
    )
    ax_z.set_xlabel("Metallicity")
    ax_z.set_ylabel("N")
    ax_z.set_title(f"Metallicities — {label}")
    ax_z.legend(fontsize=7.5)

plt.suptitle(
    f"Parametric SFH + MetalDist resampling  ({NP} → {NP * FACTOR} particles)",
    fontsize=13,
    y=1.01,
)
plt.show()
