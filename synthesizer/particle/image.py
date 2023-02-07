from ..observations import Image
import numpy as np
from numpy import histogram2d


class ImageGenerator:
    def generate_histogram(
        coordinates,
        axes=(0, 1),
        weights=None,
        size=None,
        size_pixels=None,
        pixel_scale=None,
    ):
        """Generate a simple 2D histogram image"""

        x = coordinates[:, axes[0]]
        y = coordinates[:, axes[1]]

        if size_pixels:
            if pixel_scale:

                xedges = pixel_scale * np.arange(
                    -size_pixels / 2, size_pixels / 2
                )
                yedges = pixel_scale * np.arange(
                    -size_pixels / 2, size_pixels / 2
                )

        elif size:
            if pixel_scale:

                # --- this won't work properly, the size will have to be adjusted to give an integer number of pixels
                xedges = np.arange(-size / 2, size / 2, pixel_scale)
                yedges = np.arange(-size / 2, size / 2, pixel_scale)

        else:

            print("WARNING: need some size/pixel scale information")

        return Image(*histogram2d(x, y, bins=[xedges, yedges], weights=weights))
