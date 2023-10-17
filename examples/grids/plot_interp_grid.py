"""
Example of grid spectra interpolation
====================

Demonstrates the interpolation of grid spectra at the point of instantiation.
- Instantiates grids using different interpolation schemes.
- Plots and compares them with the file version.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import Grid


if __name__ == "__main__":

    # Define the unmodified grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    start = time.time()
    grid = Grid(grid_name, grid_dir=grid_dir)
    print(f"Instatiating unmodified grid took {time.time() - start:.2f} seconds")

    # Define new wavelength arrays
    vhi_new_lams = np.logspace(2, 5, 10000)
    hi_new_lams = np.logspace(2, 5, 1000)
    low_new_lams = np.logspace(2, 5, 100)

    # Instantiate new grids
    start = time.time()
    vhi_lin_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=vhi_new_lams,
                        interp_kind="linear")
    print(f"Instatiating very high resolution (linear) grid took {time.time() - start:.2f} seconds")
    start = time.time()
    hi_lin_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=hi_new_lams,
                       interp_kind="linear")
    print(f"Instatiating high resolution (linear) grid took {time.time() - start:.2f} seconds")
    start = time.time()
    low_lin_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=low_new_lams,
                        interp_kind="linear")
    print(f"Instatiating low resolution (linear) grid took {time.time() - start:.2f} seconds")

    start = time.time()
    vhi_cub_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=vhi_new_lams,
                        interp_kind="cubic")
    print(f"Instatiating very high resolution (cubic) grid took {time.time() - start:.2f} seconds")
    start = time.time()
    hi_cub_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=hi_new_lams,
                       interp_kind="cubic")
    print(f"Instatiating high resolution (cubic) grid took {time.time() - start:.2f} seconds")
    start = time.time()
    low_cub_grid = Grid(grid_name, grid_dir=grid_dir, new_lam=low_new_lams,
                        interp_kind="cubic")
    print(f"Instatiating low resolution (cubic) grid took {time.time() - start:.2f} seconds")

    # Set up plot
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.loglog()

    ax.plot(grid.lam, grid.spectra["incident"][25, 5, :], label="Unmodified")

    ax.plot(vhi_lin_grid.lam, vhi_lin_grid.spectra["incident"][25, 5, :], label="Very High (Linear)")
    ax.plot(hi_lin_grid.lam, hi_lin_grid.spectra["incident"][25, 5, :], label="High (Linear)")
    ax.plot(low_lin_grid.lam, low_lin_grid.spectra["incident"][25, 5, :], label="Low (Linear)")
    ax.plot(vhi_cub_grid.lam, vhi_cub_grid.spectra["incident"][25, 5, :], label="Very High (Cubic)")
    ax.plot(hi_cub_grid.lam, hi_cub_grid.spectra["incident"][25, 5, :], label="High (Cubic)")
    ax.plot(low_cub_grid.lam, low_cub_grid.spectra["incident"][25, 5, :], label="Low (Cubic)")

    ax.set_xlabel(r"$\lambda/\AA$")
    ax.set_ylabel(r"$L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1}$")

    ax.legend()

    plt.show()
