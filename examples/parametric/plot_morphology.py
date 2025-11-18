"""
Generate parametric morphology profiles
=======================================

This script demonstrates:

- Defining a resolution and grid for storing binned luminosity values.
- Generating and visualizing:
    * A 2D Gaussian profile.
    * A single annulus of a 2D Gaussian.
    * A point source.
    * A 2D Sersic profile.
    * A single annulus of a 2D Sersic profile.

Each resulting density distribution is visualised with matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from unyt import kpc, unyt_array

from synthesizer.parametric import (
    Gaussian2D,
    Gaussian2DAnnuli,
    PointSource,
    Sersic2D,
    Sersic2DAnnuli,
)

# Common settings
resolution = 0.05 * kpc
npix = (500, 500)
x = unyt_array(np.linspace(-10, 10, npix[0]), "kpc")
y = unyt_array(np.linspace(-10, 10, npix[1]), "kpc")
xx, yy = np.meshgrid(x.value, y.value) * kpc

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z = 0.5

# 2D Gaussian
gauss = Gaussian2D(
    x_mean=0 * kpc,
    y_mean=0 * kpc,
    stddev_x=1 * kpc,
    stddev_y=2 * kpc,
    rho=0.3,
)
print(gauss)
gauss_grid = gauss.get_density_grid(resolution, npix)

# Gaussian annulus
radii = unyt_array([1.0, 3.0, 5.0, 7.0, np.inf], "kpc")
gauss_ann = Gaussian2DAnnuli(
    x_mean=0 * kpc,
    y_mean=0 * kpc,
    stddev_x=1 * kpc,
    stddev_y=2 * kpc,
    radii=radii,
    rho=0.3,
)
print(gauss_ann)
gauss_ann_grid = gauss_ann.get_density_grid(resolution, npix, annulus=1)

# Point source at (2,2) kpc
ps = PointSource(
    offset=unyt_array([2.0, 2.0], "kpc"),
    cosmo=cosmo,
    redshift=z,
)
print(ps)
ps_grid = ps.get_density_grid(resolution, npix)

# 2D Sersic (n=2, e=0.4, θ=30°)
sersic = Sersic2D(
    r_eff=4 * kpc,
    amplitude=1.0,
    sersic_index=2.0,
    x_0=0 * kpc,
    y_0=0 * kpc,
    theta=np.deg2rad(30),
    ellipticity=0.4,
    cosmo=cosmo,
    redshift=z,
)
print(sersic)
sersic_grid = sersic.get_density_grid(resolution, npix)

# Sersic annulus (note that infinity is always added as the last radius)
s_radii = unyt_array([2.0, 4.0, 6.0, 8.0], "kpc")
sersic_ann = Sersic2DAnnuli(
    r_eff=4 * kpc,
    radii=s_radii,
    amplitude=1.0,
    sersic_index=2.0,
    x_0=0 * kpc,
    y_0=0 * kpc,
    theta=np.deg2rad(30),
    ellipticity=0.4,
    cosmo=cosmo,
    redshift=z,
)
print(sersic_ann)
sersic_ann_grid = sersic_ann.get_density_grid(resolution, npix, annulus=1)

# Plot everything in a 2×3 grid
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
plots = [
    (gauss_grid, "Gaussian2D"),
    (gauss_ann_grid, "Gaussian2DAnnuli (outer shell)"),
    (ps_grid, "PointSource"),
    (sersic_grid, "Sersic2D (n=2)"),
    (sersic_ann_grid, "Sersic2DAnnuli (outer shell)"),
]
for ax, (grid, title) in zip(axes.flat, plots):
    im = ax.contourf(xx, yy, grid, levels=50)
    ax.set_title(title)
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    fig.colorbar(im, ax=ax, label="Density")

# hide the empty subplot
axes.flat[-1].axis("off")

plt.tight_layout()
plt.show()
