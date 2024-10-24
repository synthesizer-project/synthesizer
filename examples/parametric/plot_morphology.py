"""
Create morphology example
=========================

DRAFT

"""

# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from unyt import kpc, unyt_array

from synthesizer.parametric.morphology import Gaussian2D, PointSource

# Define gaussian values
gaussian = Gaussian2D(x_mean=0, y_mean=0, stddev_x=1, stddev_y=1, rho=0.5)

x_dat = np.linspace(-5, 5, 100)
y_dat = np.linspace(-5, 5, 100)

xx, yy = np.meshgrid(x_dat, y_dat)

# Define density grid
density_grid = gaussian.compute_density_grid(xx, yy)

# Plot figure from
plt.contourf(xx, yy, density_grid, levels=50)
plt.colorbar(label="Density")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title(("Example 2D Gaussian Distribution"))
plt.show()


# PointSource example usage

resolution = 0.1 * kpc
npix = (100, 100)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
redshift = 0.5

offset = unyt_array([2.0, 2.0], units=kpc)
point_source = PointSource(offset=offset, cosmo=cosmo, redshift=redshift)

pt_density_grid = point_source.get_density_grid(resolution, npix)


plt.imshow(pt_density_grid)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Example Point Source")
plt.show()
