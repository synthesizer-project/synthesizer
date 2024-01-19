"""A module containing the definition of an image.
"""
import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import Quantity


class Image:
    """ """

    # Define quantities
    resolution = Quantity()
    fov = Quantity()

    def __init__(self, resolution, fov):
        # Set the quantities
        self.resolution = resolution
        self.fov = fov

        # Calculate the shape of the image
        self.npix = (
            int(self.fov[0] / self.resolution),
            int(self.fov[1] / self.resolution),
        )

        # Attribute to hold the image array itself
        self.arr = None

    def _get_img_hist(
        self,
        signal,
        coordinates=None,
    ):
        """
        Calculate an image with no smoothing.

        This is only applicable to particle based images and is just a
        wrapper for numpy.histogram2d.

        Args:
            signal (array_like, float):
                The signal to be sorted into the image.
            coordinates (unyt_array, float):
                The coordinates of the particles.

        Returns:
            img (array_like, float)
                A 2D array containing the pixel values sorted into the image.
                (npix, npix)
        """
        # Strip off any units from the signal if they are present
        if isinstance(signal, (unyt_quantity, unyt_array)):
            signal = signal.value

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value

        self.arr = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=self.npix,
            weights=signal,
        )[0]

        return self.arr

    def _get_img_smoothed(
        self,
        signal,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        density_grid=None,
    ):
        """
        Calculate a smoothed image.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:
            signal (array_like, float):
                The signal to be sorted into the image.
            coordinates (unyt_array, float):
                The coordinates of the particles. (particle case only)
            smoothing_lengths (unyt_array, float):
                The smoothing lengths of the particles. (particle case only)
            kernel (str):
                The kernel to use for smoothing. (particle case only)
            density_grid (array_like, float):
                The density grid to smooth over. (parametric case only)

        Returns:
            img : array_like (float)
                A 2D array containing particles sorted into an image.
                (npix[0], npix[1])

        Raises:
            InconsistentArguments
                If conflicting particle and parametric arguments are passed
                or any arguments are missing an error is raised.
        """
        # Ensure we have the right arguments
        if density_grid is not None and (
            coordinates is not None
            or smoothing_lengths is not None
            or kernel is not None
        ):
            raise exceptions.InconsistentArguments(
                "Parametric smoothed images only require a density grid. You "
                "Shouldn't have particle based quantities in conjunction with "
                "parametric properties, what are you doing?"
            )
        if density_grid is None and (
            coordinates is None or smoothing_lengths is None or kernel is None
        ):
            raise exceptions.InconsistentArguments(
                "Particle based smoothed images require the coordinates, "
                "smoothing_lengths, and kernel arguments to be passed."
            )

        # Handle the parametric case
        if density_grid is not None:
            # Multiply the density grid by the sed to get the IFU
            self.img = density_grid[:, :] * signal

            return self.arr

        from .extensions.image import make_img

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value
        smoothing_lengths = smoothing_lengths.to(self.resolution.units).value

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        smls = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(coordinates[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(coordinates[:, 1], dtype=np.float64)

        self.arr = make_img(
            signal,
            smls,
            xs,
            ys,
            self.kernel,
            self._resolution,
            self.npix[0],
            self.coordinates.shape[0],
            self.kernel_threshold,
            self.kernel_dim,
        )

        return self.arr
