import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from synthesizer.grid import Grid

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


if __name__ == "__main__":
    # Define choice of SPS model and initial mass function (IMF)
    sps_names: List[str] = [
        "bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2",
        "bpass-v2.2.1-bin_chab-100_cloudy-v17.00_log10Uref-2",
    ]

    iZ: int
    ia: int
    log10Z: float = -2.0  # log10(metallicity)
    log10age: float = 6.0  # log10(age/yr)

    spec_names: List[str] = ["total"]

    grid1: Grid = Grid(sps_names[0])
    grid2: Grid = Grid(sps_names[1])

    iZ, log10Z = grid1.get_nearest_log10Z(log10Z)
    print(f"closest metallicity: {log10Z:.2f}")
    ia, log10age = grid1.get_nearest_log10age(log10age)
    print(f"closest age: {log10age:.2f}")

    fig: Figure = plt.figure(figsize=(3.5, 5.0))
    ax: Axes = fig.add_axes((0.2, 0.1, 0.75, 0.8))

    ax.axhline(c="k", lw=3, alpha=0.05)

    for spec_name in spec_names:
        Lnu1: np.ndarray = grid1.spectra[spec_name][ia, iZ, :]
        Lnu2: np.ndarray = grid2.spectra[spec_name][ia, iZ, :]

        ax.plot(
            np.log10(grid1.lam),
            np.log10(Lnu2 / Lnu1),
            lw=1,
            alpha=0.8,
            label=spec_name,
        )

    ax.set_xlim([3.0, 4.0])
    ax.set_ylim([-0.75, 0.75])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    ax.set_ylabel(r"$\rm log_{10}(L_{\nu}^2/L_{\nu}^1)$")

    fig.savefig("figs/comparison.pdf")
