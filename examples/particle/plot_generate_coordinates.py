"""
Generate mock coordinates
=========================

"""

import os
import matplotlib.pyplot as plt

from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.image import ImageGenerator


# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# --- create a set of particles assuming a 3D gaussian
coords = CoordinateGenerator.generate_3D_gaussian(10000)

print(coords)
print(coords.shape)

# --- make a basic (histogram) image from these coordinates
image = ImageGenerator.generate_histogram(coords, size_pixels=50,
                                          pixel_scale=0.2)

fig, ax = image.make_image_plot()#show=True)
# plt.savefig(script_path + '/plots/coodinates_example.png',
#             bbox_inches='tight'); plt.close()
plt.show()
