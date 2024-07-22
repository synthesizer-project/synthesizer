"""
Line of sight example
=====================

Test the calculation of tau_v along the line of sight
to each star particle
"""

from synthesizer.kernel_functions import Kernel
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG

gals = load_CAMELS_IllustrisTNG(
    "../../tests/data/",
    snap_name="camels_snap.hdf5",
    group_name="camels_subhalo.hdf5",
    group_dir="../../tests/data/",
)

kernel = Kernel()
kernel.get_kernel()

gals[1].calculate_los_tau_v(kappa=0.3, kernel=kernel.get_kernel())
