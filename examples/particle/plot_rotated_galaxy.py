"""
Rotating particle distributions
===============================

This example demonstrates how to rotate a particle distribution. This is
useful for rotating galaxies to a specific angle.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import CoordinateGenerator, Galaxy, Stars
from unyt import Myr, degree, kpc


def calculate_smoothing_lengths(positions, num_neighbors=56):
    """Calculate the SPH smoothing lengths for a set of coordinates."""
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=num_neighbors + 1)

    # The k-th nearest neighbor distance (k = num_neighbors)
    kth_distances = distances[:, num_neighbors]

    # Set the smoothing length to the k-th nearest neighbor
    # distance divided by 2.0
    smoothing_lengths = kth_distances / 2.0

    return smoothing_lengths


# Set the seed
np.random.seed(42)

# First define the covariance matrices for a disk and bulge component
# of a galaxy. We'll use this as a fake example.
disk_cov = np.array(
    [
        [30.0, 0, 0],  # Larger spread in x direction
        [0, 30.0, 0],  # Larger spread in y direction
        [0, 0, 0.5],  # Smaller spread in z direction (flattened)
    ]
)
bulge_cov = np.array(
    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]  # Equal spread in all directions
)

# Now we'll generate some coordinates for the disk and bulge
n_disk = 1000
n_bulge = 500
disk_coords = CoordinateGenerator.generate_3D_gaussian(n_disk, cov=disk_cov)
bulge_coords = CoordinateGenerator.generate_3D_gaussian(n_bulge, cov=bulge_cov)
coords = np.vstack([disk_coords, bulge_coords]) * kpc

# We'll also need to generate some velocities for the disk and bulge. The bulge
# will be in a random direction, while the disk will be in the x-y plane.
vrot = 200  # Circular rotation speed in the disk
sigma_bulge = 50  # Velocity dispersion for bulge particles
disk_velocities = np.zeros((n_disk, 3))
angles = np.arctan2(disk_coords[:, 1], disk_coords[:, 0])
disk_velocities[:, 0] = -vrot * np.sin(angles)  # Tangential velocity in x
disk_velocities[:, 1] = vrot * np.cos(angles)  # Tangential velocity in y
bulge_velocities = np.random.normal(0, sigma_bulge, size=(n_bulge, 3))
velocities = np.vstack([disk_velocities, bulge_velocities])


# Define the other properties we'll need
masses = np.ones(n_disk + n_bulge) * 1e6
ages = np.random.rand(n_disk + n_bulge) * 100 * Myr
metallicities = np.random.rand(n_disk + n_bulge) * 0.02
initial_masses = masses.copy()
redshift = 0.0
centre = np.array([0.0, 0.0, 0.0])
smoothing_lengths = calculate_smoothing_lengths(coords) * kpc

# We'll start by simply using some stars
stars = Stars(
    initial_masses,
    ages,
    metallicities,
    coordinates=coords,
    current_masses=masses,
    velocities=velocities,
    redshift=redshift,
    centre=centre,
    smoothing_lengths=smoothing_lengths,
)

# We can rotate any particle based object (or a galaxy) by any phi and theta
# (these must be passed with units)
phi = np.random.rand() * 360 * degree
theta = np.random.rand() * 180 * degree
print(f"Rotating by phi={phi}, theta={theta}")
stars.rotate_particles(phi=phi, theta=theta, inplace=True)

# So we can simply make images we'll attach these stars to a galaxy
galaxy = Galaxy(stars=stars)

# Lets take a look at the stars after this initial rotation
img = galaxy.get_map_stellar_mass(
    resolution=0.1 * kpc,
    fov=50 * kpc,
    img_type="smoothed",
    kernel=Kernel().get_kernel(),
)
img.arr = np.arcsinh(img.arr)
img.plot_map(
    show=True,
    extent=(-25, 25, -25, 25),
    cmap="magma",
)

# You can also rotate to face-on and edge-on, here we will also leave the
# original stars unchanged and get a new stars object with the rotations
# applied
face_on_stars = stars.rotate_face_on(inplace=False)
edge_on_stars = stars.rotate_edge_on(inplace=False)

# Make a galaxy to generate the images
face_on_galaxy = Galaxy(stars=face_on_stars)
edge_on_galaxy = Galaxy(stars=edge_on_stars)

# Now we can make images of the face-on and edge-on galaxies
face_on_img = face_on_galaxy.get_map_stellar_mass(
    resolution=0.1 * kpc,
    fov=50 * kpc,
    img_type="smoothed",
    kernel=Kernel().get_kernel(),
)
edge_on_img = edge_on_galaxy.get_map_stellar_mass(
    resolution=0.1 * kpc,
    fov=50 * kpc,
    img_type="smoothed",
    kernel=Kernel().get_kernel(),
)
face_on_img.arr = np.arcsinh(face_on_img.arr)
edge_on_img.arr = np.arcsinh(edge_on_img.arr)

# Plot the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
face_on_img.plot_map(
    fig=fig,
    ax=axes[0],
    show=False,
    extent=(-25, 25, -25, 25),
    cmap="magma",
)
axes[0].set_title("Face-on")
edge_on_img.plot_map(
    fig=fig,
    ax=axes[1],
    show=False,
    extent=(-25, 25, -25, 25),
    cmap="magma",
)
axes[1].set_title("Edge-on")
plt.tight_layout()
plt.show()
