"""
Adding Particles objects
========================

This example demonstrates how to add Particles and its child objects together.

It creates Stars, Gas, and BlackHoles objects, adds them and demonstrates
what happens when an improper addition is attempted.
"""

import numpy as np
from unyt import Msun, Myr

from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars

# Create fake stars, gas, and black holes objects to add
stars1 = Stars(
    initial_masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
    ages=np.array([1.0, 2.0, 3.0]) * Myr,
    metallicities=np.array([0.01, 0.02, 0.03]),
    redshift=1.0,
    tau_v=np.array([0.1, 0.2, 0.3]),
    coordinates=np.random.rand(3, 3),
    dummy_attr=1.0,
)
stars2 = Stars(
    initial_masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
    ages=np.array([4.0, 5.0, 6.0, 7.0]) * Myr,
    metallicities=np.array([0.04, 0.05, 0.06, 0.07]),
    redshift=1.0,
    tau_v=np.array([0.4, 0.5, 0.6, 0.7]),
    coordinates=np.random.rand(4, 3),
    dummy_attr=1.2,
)
gas1 = Gas(
    masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
    metallicities=np.array([0.01, 0.02, 0.03]),
    redshift=1.0,
    coordinates=np.random.rand(3, 3),
)
gas2 = Gas(
    masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
    metallicities=np.array([0.04, 0.05, 0.06, 0.07]),
    redshift=1.0,
    coordinates=np.random.rand(4, 3),
)
# blackholes1 = BlackHoles(
#     masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
#     accretion_rates=np.array([0.1, 0.2, 0.3]),
#     redshift=1.0,
#     coordinates=np.random.rand(3, 3),
# )
# blackholes2 = BlackHoles(
#     masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
#     accretion_rates=np.array([0.4, 0.5, 0.6, 0.7]),
#     redshift=1.0,
#     coordinates=np.random.rand(4, 3),
# )

print("Stars 1:")
print(stars1)
print("Stars 2:")
print(stars2)
print("Stars 1 + Stars 2:")
print(stars1 + stars2)
print("Gas 1:")
print(gas1)
print("Gas 2:")
print(gas2)
print("Gas 1 + Gas 2:")
print(gas1 + gas2)
# print("Black Holes 1:")
# print(blackholes1)
# print("Black Holes 2:")
# print(blackholes2)
# print("Black Holes 1 + Black Holes 2:")
# print(blackholes1 + blackholes2)

# We can't add different types of particles together
try:
    print("Stars 1 + Gas 1:")
    print(stars1 + gas1)
except TypeError as e:
    print(f"Error: {e}")

# We also can't add particles with different redshifts
stars2.redshift = 2.0
try:
    print("Stars 1 + Stars 2:")
    print(stars1 + stars2)
except ValueError as e:
    print(f"Error: {e}")

# If an attribute is absent from one of the particles, it will be ignored
stars2.dummy_attr = None
print("Stars 1 + Stars 2 (no dummy attr):")
print(stars1 + stars2)
