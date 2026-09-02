"""
Dust curves example
===================

Plot dust curves
"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from unyt import Angstrom, um, unyt_quantity

from synthesizer.emission_models import attenuation

models = [
    "PowerLaw",
    "Calzetti2000",
    "MWN18",
    "GrainModels",
    "GrainModels",
    "GrainModels",
    "ParametricLi08",
    "ParametricLi08",
    "ParametricLi08",
    "ParametricLi08",
    "SommovigoBartlett2026",
    "SommovigoBartlett2026",
    "SommovigoBartlett2026",
]

params = [
    {"slope": -1.0},
    {
        "slope": 0.0,
        "cent_lam": 0.2175 * um,
        "ampl": 1.26,
        "gamma": 0.0356 * um,
    },
    {},
    {"model": "WD01", "submodel": "MW"},
    {"model": "WD01", "submodel": "SMC"},
    {"model": "WD01", "submodel": "LMC"},
    {"model": "MW"},
    {"model": "LMC"},
    {"model": "SMC"},
    {"model": "Calzetti"},
    {"B_0": 1.199, "B_1s": -0.076, "B_2s": 0.467, "B_3": 3.036},
    {"B_0": 0.000, "B_1s": -0.005, "B_2s": -1.330, "B_3": 4.121},
    {"B_0": 0.001, "B_1s": 0.053, "B_2s": -0.030, "B_3": 1.506},
]

# Custom labels for SommovigoBartlett2026 entries showing the dust
# mixture and grain-model reference instead of raw parameter values.
sb26_labels = {
    10: "SB26 (TNG-SKIRT, MW WD01 mix)",
    11: "SB26 (TNG-SKIRT, SMC WD01 mix)",
    12: "SB26 (TNG-SKIRT, stellar HA19 mix)",
}

colors = cmr.take_cmap_colors("cmr.guppy", len(models))

lam = np.arange(1000, 10000, 10) * Angstrom

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for ii, (model, param) in enumerate(zip(models, params)):
    emodel = getattr(attenuation, model)(**param)
    ls = "-"
    lw = 1.5
    # Use custom label for SB26 entries, otherwise build from params
    if ii in sb26_labels:
        label = sb26_labels[ii]
        ls = ":"
        lw = 2.0
    else:
        param_strs = []
        for p, v in param.items():
            if isinstance(v, unyt_quantity):
                param_strs.append(f"{p}={v.value} {v.units}")
            else:
                param_strs.append(f"{p}={v}")
        label = f"{model} ({', '.join(param_strs)})"

    ax.plot(
        lam, emodel.get_tau(lam), color=colors[ii], label=label, ls=ls, lw=lw
    )

ax.set_xlabel(r"$\lambda/(\AA)$", fontsize=12)
ax.set_ylabel(r"A$_{\lambda}/$A$_{V}$", fontsize=12)
ax.set_yticks(np.arange(0, 14))
ax.set_xlim(np.min(lam), np.max(lam))

ax.legend(fontsize=9, loc="upper right")
ax.grid(True)

plt.show()

# --- Plot 2: SB26 predicted from median TNG galaxy properties ---
from synthesizer.emission_models.transformers.dust_attenuation import (
    SommovigoBartlett2026,
)

# Median galaxy properties from the combined TNG50+TNG100
# sample in Sommovigo & Bartlett (2026)
tng_medians = {
    "sigma_sfr": 0.0213,
    "inclination": 65.63,
    "z_gas": 0.0202,
    "log10_mstar": 10.17,
    "ssfr": 0.0956,
}

dust_models = {
    "(TNG-SKIRT, MW WD01 mix)": "MW",
    "(TNG-SKIRT, SMC WD01 mix)": "SMC",
    "(TNG-SKIRT, stellar HA19 mix)": "stellar",
}

dm_colors = {
    "(TNG-SKIRT, MW WD01 mix)": "#3b0f6f",
    "(TNG-SKIRT, SMC WD01 mix)": "#b63679",
    "(TNG-SKIRT, stellar HA19 mix)": "#fd9f6c",
}

fig2, ax2 = plt.subplots(figsize=(8, 6))

for label, dm in dust_models.items():
    params = SommovigoBartlett2026.predict(
        **tng_medians,
        dust_model=dm,
        add_noise=False,
    )
    curve = SommovigoBartlett2026(
        B_0=params["B_0"],
        B_1s=params["B_1s"],
        B_2s=params["B_2s"],
        B_3=params["B_3"],
    )
    ax2.plot(
        lam,
        curve.get_tau(lam),
        color=dm_colors[label],
        lw=2.5,
        label=(
            f"SB26 {label}\n"
            f"  $A_V$={params['A_V']:.3f}, $B_0$={params['B_0']:.4f}, "
            f"$B_{{1s}}$={params['B_1s']:.4f}, "
            f"$B_{{2s}}$={params['B_2s']:.4f}, "
            f"$B_3$={params['B_3']:.3f}"
        ),
    )

ax2.set_xlabel(r"$\lambda/(\AA)$", fontsize=12)
ax2.set_ylabel(r"A$_{\lambda}/$A$_{V}$", fontsize=12)
ax2.set_title(
    r"SB26 predicted from median TNG50-100 z=0.07 galaxy properties"
    "\n"
    r"($\Sigma_{\rm SFR}$=0.021 $M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2}$, "
    r"$i$=65.6°, "
    r"$Z_{\rm gas}$=1.4 $Z_\odot$, "
    r"$\log_{10} M_\star/M_\odot$=10.17, "
    r"sSFR=0.096 Gyr$^{-1}$)",
    fontsize=10,
)
ax2.set_yticks(np.arange(0, 14))
ax2.set_xlim(np.min(lam), np.max(lam))
ax2.set_ylim(0, 14)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True)

plt.tight_layout()
plt.show()
