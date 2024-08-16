"""
Age lookup test
===============

Test using a lookup grid for particle ages (calculated
from the scsale factor representation native to most
cosmological snapshot outputs), and compare to the direct
astropy calculation. Show dependence of relative error
on lookup grid resolution.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15

from synthesizer.load_data.utils import age_lookup_table, lookup_age

# get scale factors from test file
with h5py.File("../../tests/data/camels_snap.hdf5", "r") as hf:
    form_time = hf["PartType4/GFM_StellarFormationTime"][:]

# Calculate ages of these explicitly using astropy
part_ages_proper = Planck15.age(1.0 / form_time - 1)

fig, ax = plt.subplots(1, 1)

# Loop over different look up grid resolutions
for delta_a in [1e-2, 1e-3, 1e-4, 1e-5, 5e-6]:
    # create the lookup grid
    scale_factors, ages = age_lookup_table(Planck15, delta_a=delta_a)

    # Look up the ages for the particles
    part_ages = lookup_age(form_time, scale_factors, ages)

    ax.scatter(
        part_ages_proper,
        np.log10(np.abs((part_ages - part_ages_proper).value)),
        s=1,
        alpha=1,
        label=delta_a,
    )

ax.legend(title=r"$\Delta a$:")
ax.set_ylabel(r"$\Delta \mathrm{log_{10} \, age \;\; (Gyr)}$")
plt.show()
