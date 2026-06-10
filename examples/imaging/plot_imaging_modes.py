"""
Imaging modes example
=====================

Demonstrates the three available particle imaging backends — histogram,
octree-smoothed, and quadtree-smoothed — applied to a CAMELS-IllustrisTNG
galaxy.  The quadtree backend uses the same area-integration algorithm
as the octree backend, driven from the pixel side for better parallelism.

If the C extensions were built with ``ATOMIC_TIMING=1`` the accumulated
operation timers are printed at the end.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from unyt import kpc

from synthesizer import TEST_DATA_DIR
from synthesizer.kernel_functions import Kernel
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG


def _print_timing_table():
    """Print the OperationTimers summary table if any operations were tracked.

    The C++ timer extension only collects data when the package is built
    with ``ATOMIC_TIMING=1 pip install -e .``.  Otherwise this is a no-op.
    """
    from synthesizer.utils.operation_timers import OperationTimers

    timers = OperationTimers()
    if len(timers) == 0:
        print(
            "\nNo operation timings collected.\n"
            "Re-install with  ATOMIC_TIMING=1 pip install -e .  to enable."
        )
        return
    print()
    OperationTimers.print_table()


if __name__ == "__main__":
    # -- load a CAMELS galaxy ----------------------------------------------
    gals = load_CAMELS_IllustrisTNG(
        TEST_DATA_DIR,
        snap_name="camels_snap.hdf5",
        group_name="camels_subhalo.hdf5",
        physical=True,
    )
    galaxy = gals[1]
    print(f"Galaxy has {galaxy.stars.nstars} star particles")

    # -- common imaging parameters -----------------------------------------
    resolution = 0.5 * kpc
    fov = 50 * kpc
    kernel = Kernel()
    kernel_threshold = 1.0

    # -- 1. Histogram image ------------------------------------------------
    print("Generating histogram image ...")
    img_hist = galaxy.get_map_stellar_mass(
        resolution=resolution,
        fov=fov,
        img_type="hist",
    )

    # -- 2. Octree-smoothed image ------------------------------------------
    print("Generating octree-smoothed image ...")
    img_octree = galaxy.get_map_stellar_mass(
        resolution=resolution,
        fov=fov,
        img_type="smoothed",
        kernel=kernel,
        kernel_threshold=kernel_threshold,
        nthreads=4,
        backend="octree",
    )

    # -- 3. Quadtree-smoothed image ----------------------------------------
    print("Generating quadtree-smoothed image ...")
    img_quadtree = galaxy.get_map_stellar_mass(
        resolution=resolution,
        fov=fov,
        img_type="smoothed",
        kernel=kernel,
        kernel_threshold=kernel_threshold,
        nthreads=4,
        backend="quadtree",
    )

    # -- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    vmin = max(img_octree.arr[img_octree.arr > 0].min(), 1e-8)
    vmax = img_octree.arr.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    titles = ["Histogram", "Smoothed (octree)", "Smoothed (quadtree)"]
    images = [img_hist, img_octree, img_quadtree]

    fov_arr = np.atleast_1d(fov.value)
    half_width = fov_arr[0] / 2
    half_height = fov_arr[-1] / 2

    for ax, title, img in zip(axes, titles, images):
        im = ax.imshow(
            img.arr.T,
            origin="lower",
            norm=norm,
            cmap="inferno",
            extent=(-half_width, half_width, -half_height, half_height),
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label(f"Stellar mass [{img_hist.units}]", fontsize=11)

    plt.show()

    # -- Optional: print timing table --------------------------------------
    _print_timing_table()
