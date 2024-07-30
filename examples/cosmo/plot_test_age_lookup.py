import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from synthesizer.load_data.utils import age_lookup_table, lookup_age

# get scale factors from test file
with h5py.File("../../tests/data/camels_snap.hdf5", "r") as hf:
    form_time = hf["PartType4/GFM_StellarFormationTime"][:]

# Calculate ages of these explicitly using astropy
part_ages_proper = Planck15.age(1.0 / form_time - 1)

# Loop over different look up grid resolutions
for resolution in [100, 500, 1000, 2000, 4000]:
    # create the lookup grid
    scale_factors, ages = age_lookup_table(Planck15, resolution=resolution)

    # Look up the ages for the particles
    part_ages = lookup_age(form_time, scale_factors, ages)

    plt.scatter(
        part_ages_proper,
        np.log10(np.abs((part_ages - part_ages_proper).value)),
        s=1,
        alpha=1,
        label=resolution
    )

plt.legend(title='Resolution:')
plt.ylabel('$\Delta \mathrm{log_{10} \, age \;\; (Gyr)}$')
plt.show()
