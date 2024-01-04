"""
Line of sight example
=====================

Test the calculation of tau_v along the line of sight
to each star particle
"""

import numpy as np
import matplotlib.pyplot as plt

from synthesizer.load_data import load_CAMELS_IllustrisTNG
from synthesizer.kernel_functions import kernel

gals = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name=f"camels_snap.hdf5",
    fof_name=f"camels_subhalo.hdf5",
    fof_dir="../../tests/data/",
)

kern = kernel()
kern.get_kernel()

gals[0].calculate_los_tau_v(kappa=0.3, kernel=kern.get_kernel())
