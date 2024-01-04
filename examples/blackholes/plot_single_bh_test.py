"""
Compare single black hole particle, parametric and grid spectra
===========================

A sanity check example for a single blackhole, comparing the spectra generated
from the parametric, particle and grid method. These should give
indistinguishable results.
"""
import numpy as np
import matplotlib.pyplot as plt
from unyt import Msun, yr, Angstrom, deg, K, unyt_quantity, unyt_array

from synthesizer.blackhole_emission_models import UnifiedAGN
from synthesizer.parametric import BlackHole
from synthesizer.particle import BlackHoles
from synthesizer.sed import plot_spectra

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
grid_dir = "../../tests/test_grid/"
emission_model = UnifiedAGN(
    disc_model="test_grid_agn", photoionisation_model="", grid_dir=grid_dir
)

grid_spectra = emission_model.get_spectra(
    mass=mass,
    cosine_inclination=np.cos(inclination.to("rad").value),
    accretion_rate_eddington=0.4505151287294214,
    metallicity=metallicity,
    bolometric_luminosity=part_bh.bolometric_luminosity,
)

# Get the spectra assuming this emission model
ngp_para_spectra = para_bh.get_spectra_intrinsic(
    emission_model,
    grid_assignment_method="ngp",
)
ngp_part_spectra = part_bh.get_particle_spectra_intrinsic(
    emission_model,
    grid_assignment_method="ngp",
)
# cic_para_spectra = para_bh.get_spectra_intrinsic(
#     emission_model,
#     grid_assignment_method="cic",
# )
# cic_part_spectra = part_bh.get_spectra_intrinsic(
#     emission_model,
#     grid_assignment_method="cic",
# )

for key in part_bh.particle_spectra:
    ngp_part_spectra[key]._lnu = ngp_part_spectra[key]._lnu[0, :]

# Now plot spectra each comparison
for key in para_bh.spectra:
    # Create spectra dict for plotting
    spectra = {
        "Parametric (NGP)" + key: ngp_para_spectra[key],
        "Particle (NGP)" + key: ngp_part_spectra[key],
        # "Parametric (CIC)" + key: cic_para_spectra[key],
        # "Particle (CIC)" + key: cic_part_spectra[key],
        "Grid " + key: grid_spectra[key],
    }

    plot_spectra(spectra, show=True, quantity_to_plot="luminosity")
    plt.close()
