"""
Compare single black hole particle, parametric and grid spectra
===============================================================

A sanity check example for a single blackhole, comparing the spectra generated
from the parametric, particle and grid method. These should give
indistinguishable results.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, deg, kelvin, yr

from synthesizer.emission_models import UnifiedAGN
from synthesizer.emission_models.dust.emission import Greybody
from synthesizer.emissions import plot_spectra
from synthesizer.grid import Grid
from synthesizer.parametric import BlackHole
from synthesizer.particle import BlackHoles

# Set a random number seed to ensure consistent results
np.random.seed(42)

# Define black hole properties
mass = 10**8 * Msun
inclination = 60 * deg
accretion_rate = 1 * Msun / yr
metallicity = 0.01

# Define the particle and parametric black holes
para_bh = BlackHole(
    mass=mass,
    inclination=inclination,
    accretion_rate=accretion_rate,
    metallicity=metallicity,
)
part_bh = BlackHoles(
    masses=mass,
    inclinations=inclination,
    accretion_rates=accretion_rate,
    metallicities=metallicity,
)

# Define the emission model
nlr_grid = Grid("test_grid_agn-nlr", grid_dir="../../tests/test_grid")
blr_grid = Grid("test_grid_agn-blr", grid_dir="../../tests/test_grid")
model = UnifiedAGN(
    nlr_grid,
    blr_grid,
    covering_fraction_nlr=0.1,
    covering_fraction_blr=0.1,
    torus_emission_model=Greybody(1000 * kelvin, 1.5),
    ionisation_parameter=1e-2,
    hydrogen_density=1e3,
    velocity_dispersion=100 * 1e3,
)

# Get the spectra assuming this emission model
ngp_para_spectra = para_bh.get_spectra(
    model,
    grid_assignment_method="ngp",
)
ngp_part_spectra = part_bh.get_spectra(
    model,
    grid_assignment_method="ngp",
)

# Now plot spectra each comparison
for key in para_bh.spectra:
    # Create spectra dict for plotting
    spectra = {
        "Parametric (NGP)" + key: ngp_para_spectra,
        "Particle (NGP)" + key: ngp_part_spectra,
    }

    plot_spectra(spectra, show=True, quantity_to_plot="luminosity")
    plt.close()
