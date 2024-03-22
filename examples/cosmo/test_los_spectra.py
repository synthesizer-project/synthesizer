"""
Line of sight example
=====================

Test the calculation of tau_v along the line of sight
to each star particle
"""

from synthesizer.kernel_functions import Kernel
from synthesizer.load_data import load_CAMELS_IllustrisTNG

gals = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="camels_snap.hdf5",
    fof_name="camels_subhalo.hdf5",
    fof_dir="../../tests/data/",
)

kernel = Kernel()
kernel.get_kernel()

gals[0].calculate_los_tau_v(kappa=0.3, kernel=kernel.get_kernel())
