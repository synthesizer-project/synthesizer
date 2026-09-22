"""
Compare stellar velocity dispersion configurations
==================================================

This example compares final-system Doppler broadening in the Pacman stellar
emission model.
"""

import matplotlib.pyplot as plt
from unyt import km, s

from synthesizer import TEST_DATA_DIR
from synthesizer.emission_models import PacmanEmission
from synthesizer.emissions import plot_spectra
from synthesizer.grid import Grid
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG

grid = Grid("test_grid")
galaxy = load_CAMELS_IllustrisTNG(
    TEST_DATA_DIR,
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    physical=True,
)[0]

configurations = {
    "No broadening": {},
    "100 km/s": {"velocity_dispersion": 100 * km / s},
    "200 km/s": {"velocity_dispersion": 200 * km / s},
}

spectra = {}
for name, dispersions in configurations.items():
    model = PacmanEmission(
        grid,
        tau_v=0.3,
        fesc=0.0,
        fesc_ly_alpha=1.0,
        **dispersions,
    )
    spectra[name] = galaxy.stars.get_spectra(model)
    galaxy.clear_all_emissions()

plot_spectra(
    spectra,
    xlimits=(1000, 10000),
    figsize=(8, 5),
    show=False,
)
plt.show()
