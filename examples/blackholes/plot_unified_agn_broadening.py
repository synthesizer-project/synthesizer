"""
UnifiedAGN line-region broadening
=================================

Compare the BLR and NLR spectra from a ``UnifiedAGN`` model with and without
the optional Doppler broadening applied by ``velocity_dispersion_blr`` and
``velocity_dispersion_nlr``.
"""

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
    plot_spectra(
        spectra={
            f"Unbroadened {component.upper()}": unbroadened_bh.spectra[
                component
            ],
            f"Broadened {component.upper()}": broadened_bh.spectra[component],
        },
        quantity_to_plot="luminosity",
        show=True,
    )
