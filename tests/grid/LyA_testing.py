import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from synthesizer.grid import Grid

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

iZ: int
ia: int
log10Z: float = -2.0
log10age: float = 6.0

sps_name: str = "bpass-v2.2.1_chab100-bin_cloudy-v17.0_logUref-2"

fesc_LyA: float = 0.5

grid: Grid = Grid(sps_name)

iZ, log10Z = grid.get_nearest_log10Z(log10Z)
print(f"closest metallicity: {log10Z:.2f}")
ia, log10age = grid.get_nearest_log10age(log10age)
print(f"closest age: {log10age:.2f}")


fig: Figure = plt.figure(figsize=(3.5, 5.0))

left: float = 0.15
height: float = 0.8
bottom: float = 0.1
width: float = 0.8

ax: Axes = fig.add_axes((left, bottom, width, height))

Lnu: NDArray[np.float64] = grid.spectra["linecont"][iZ, ia, :]
ax.plot(np.log10(grid.lam), np.log10(Lnu), lw=2, alpha=0.3, c="k")


idx: int = grid.get_nearest_index(1216.0, grid.lam)
Lnu[idx] *= fesc_LyA

ax.plot(np.log10(grid.lam), np.log10(Lnu), lw=1, alpha=1, c="k")


ax.set_xlim([2.8, 3.6])
ax.set_ylim([18.0, 23])
ax.legend(fontsize=8, labelspacing=0.0)
ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
ax.set_ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")

plt.show()
