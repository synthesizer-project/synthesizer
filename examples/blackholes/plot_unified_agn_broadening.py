"""
UnifiedAGN line-region broadening
=================================

Compare the BLR and NLR spectra from a ``UnifiedAGN`` model with and without
the optional Doppler broadening applied by ``velocity_dispersion_blr`` and
``velocity_dispersion_nlr``.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, deg, kelvin, km, s, yr

from synthesizer.emission_models import Greybody, UnifiedAGN
from synthesizer.emissions import plot_spectra
from synthesizer.grid import Grid
from synthesizer.parametric import BlackHole

# Define a single black hole and the grids used by the AGN model.
black_hole = BlackHole(
    mass=10**8 * Msun,
    inclination=45 * deg,
    accretion_rate=1 * Msun / yr,
    metallicity=0.01,
)
nlr_grid = Grid("test_grid_agn-nlr")
blr_grid = Grid("test_grid_agn-blr")

# Common model settings. The first model leaves line-region spectra at the grid
# resolution; the second makes the final BLR/NLR components Doppler broadened.
common_model_kwargs = {
    "nlr_grid": nlr_grid,
    "blr_grid": blr_grid,
    "torus_emission_model": Greybody(1000 * kelvin, 1.5),
    "covering_fraction_nlr": 0.1,
    "covering_fraction_blr": 0.1,
    "ionisation_parameter_nlr": 1e-2,
    "ionisation_parameter_blr": 1e-2,
    "hydrogen_density_nlr": 1e3,
    "hydrogen_density_blr": 1e3,
}
unbroadened_model = UnifiedAGN(**common_model_kwargs)
broadened_model = UnifiedAGN(
    **common_model_kwargs,
    velocity_dispersion_nlr=500 * km / s,
    velocity_dispersion_blr=2000 * km / s,
)

# Generate the two spectra independently so the second call does not overwrite
# the first model outputs on the black hole object.
unbroadened_bh = black_hole
broadened_bh = BlackHole(
    mass=10**8 * Msun,
    inclination=45 * deg,
    accretion_rate=1 * Msun / yr,
    metallicity=0.01,
)
unbroadened_bh.get_spectra(unbroadened_model)
broadened_bh.get_spectra(broadened_model)

# Plot the final BLR and NLR model outputs. In the broadened model these are
# transformations of the internal unbroadened_blr/unbroadened_nlr components.
for component in ("blr", "nlr"):
    fig, (spectra_ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
        sharex=True,
    )

    unbroadened_sed = unbroadened_bh.spectra[component]
    broadened_sed = broadened_bh.spectra[component]
    plot_spectra(
        spectra={
            f"Unbroadened {component.upper()}": unbroadened_sed,
            f"Broadened {component.upper()}": broadened_sed,
        },
        fig=fig,
        ax=spectra_ax,
        quantity_to_plot="luminosity",
        show=False,
    )
    spectra_ax.loglog()
    spectra_ax.set_xlabel("")

    # Plot fractional residuals where the unbroadened model is non-zero. The
    # broadened model conserves integrated luminosity but redistributes it in
    # wavelength, so the residual panel highlights line-profile changes.
    ok = unbroadened_sed.luminosity > 0
    residual = np.zeros_like(unbroadened_sed.luminosity.value)
    residual[ok] = (
        broadened_sed.luminosity[ok] / unbroadened_sed.luminosity[ok]
    ).value - 1.0

    residual_ax.axhline(0.0, color="0.5", lw=1, ls="--")
    residual_ax.plot(unbroadened_sed.lam, residual, color="k", lw=1)
    residual_ax.set_xscale("log")
    residual_ax.set_ylabel("Frac. residual")
    residual_ax.set_xlabel("Rest-frame wavelength")

    plt.show()
