"""
Any example of how to do image addition and testing error functionality.
"""
import numpy as np
from synthesizer.exceptions import InconsistentAddition
from synthesizer.filters import SVOFilterCollection as Filters
from synthesizer.imaging.images import Image


# Create a filter collection
filter_codes1 = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W", "JWST/NIRCam.F200W"]
filter_codes2 = filter_codes1[:-1]
filters1 = Filters(filter_codes1)
filters2 = Filters(filter_codes2)

fake_img = np.zeros((100, 100))

# Create fake image objects
img1 = Image(resolution=0.5, npix=100, fov=None, filters=())
img2 = Image(resolution=0.4, npix=100, fov=None, filters=())
img_with_filters1 = Image(resolution=0.5, npix=100, fov=None, filters=filters1)
img_with_filters2 = Image(resolution=0.5, npix=100, fov=None, filters=filters2)
img1.img = fake_img
img2.img = fake_img
for f in filter_codes1:
    img_with_filters1.imgs[f] = fake_img
for f in filter_codes2:
    img_with_filters2.imgs[f] = fake_img

# Add images
composite = img1 + img1 + img1
composite_with_filters = img_with_filters1 + img_with_filters1
composite_mixed1 = img1 + img_with_filters1
composite_mixed2 = img_with_filters1 + img1

# Added images preserve a history of what objects were added
print("Example of image nesting from img1 + img2 + img3:",
      composite.combined_imgs)

print("Error Demonstration:")

# Demonstrate errors
try:
    broken = img1 + img2
except InconsistentAddition as e:
    print(e)
try:
    broken = img_with_filters1 + img_with_filters2
except InconsistentAddition as e:
    print(e)





