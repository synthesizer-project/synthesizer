"""
Get sum of image inside an aperture
===================================

This example shows how to get the sum of the image inside an aperture.
It will create a fake image containing all ones and then calculate the sum
inside various apertures, comparing to the expected value.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from unyt import Hz, Mpc, erg, s

from synthesizer.imaging.image import Image

# Define the resolution of the image
res = 100

# Set up the fake image
img = Image(
    resolution=1 * Mpc,
    fov=res * Mpc,
    img=np.ones((res, res)) * erg / s / Hz,
)

# Define the apertures
app_radii = np.logspace(-1, np.log10(res / 2), 100) * Mpc

# Compute the theoretical sum for each aperture
theoretical_sum = np.pi * app_radii**2

# Define the aperture centre (we'll use the centre of the image which
# is the same as not passing a centre)
# This should be in pixel coordinates
centre = np.array([res / 2, res / 2])

# Calculate the sum inside the apertures
results = []
times = []
for ind, r in enumerate(app_radii):
    start = time.time()
    sum_ = img.get_signal_in_aperture(r, centre, nthreads=1)
    results.append(sum_)
    times.append(time.time() - start)

# Convert to an array
results = np.array(results)

# Plot the residual between truth and calculation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.scatter(theoretical_sum, results)
ax.set_xlabel("Theoretical sum")
ax.set_ylabel("Calculated sum")
ax.set_title(f"Pixel count: ({res}x{res})")
plt.show()
plt.close(fig)
