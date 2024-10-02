"""
Filters example
===============

A demonstration of filter and filter collections creation and usage.
"""

import numpy as np
from unyt import angstrom

from synthesizer.filters import UVJ, Filter, FilterCollection

# Define some wavelengths (in A) over which we want to define our filters
lams = np.linspace(2000, 55000, 1000) * angstrom

# Define an individual filter from SVO.
filt = Filter("JWST/NIRCam.F200W", new_lam=lams)

# And define a fake transmission curve.
trans = np.zeros(lams.size)
trans[int(lams.size / 4) : int(lams.size / 2)] = 1

# You can either define a FilterCollection from a single filter type
fs = [f"JWST/NIRCam.{f}" for f in ["F070W", "F444W"]]
fc1 = FilterCollection(filter_codes=fs, new_lam=lams)

# ... or a mixture assuming the wavelengths are in the same unit system.
fs = [f"JWST/NIRCam.{f}" for f in ["F090W", "F250M"]]
tophats = {
    "U": {"lam_eff": 3650 * angstrom, "lam_fwhm": 660 * angstrom},
    "V": {"lam_eff": 5510 * angstrom, "lam_fwhm": 880 * angstrom},
    "J": {"lam_eff": 12200 * angstrom, "lam_fwhm": 2130 * angstrom},
}
generics = {"filter1": trans}
fc2 = FilterCollection(
    filter_codes=fs, tophat_dict=tophats, generic_dict=generics, new_lam=lams
)

# You can get the length of a FilterCollection.
print("We have %d filters" % len(fc2))

# Loop over it as if it were a list.
print("My Filters:")
for f in fc2:
    print(f.filter_code)

# Compare FilterCollections.
if fc2 == fc2:
    print("This is the same filter collection!")
if fc2 != fc1:
    print("These are not the same filter collection!")

# Add filters to simply combine filter collections and individual filters
fc = fc1 + filt
fc += fc2

# Print out the new filter codes.
print("My combined Filters:", fc.filter_codes)

# You can even easily plot the transmission curves with a helper method.
fc.plot_transmission_curves()

# There's also a helper function to create the above UVJ filter set.
fc = UVJ(new_lam=lams)
fc.plot_transmission_curves()
