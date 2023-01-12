""" Definitions for image objects
"""
import math
import numpy as np
import synthesizer.exceptions as exceptions
from synthesizer.imaging.observation import (Observation, ParticleObservation,
                                             ParametricObservation)
from synthesizer.imaging.spectral_cubes import (ParticleSpectralCube,
                                                ParametricSpectralCube)


class Image(Observation):
    """
    The Image object, containing attributes and methods for calculating images.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None):

        # Sanitize inputs
        if filters is None:
            filters = ()

        # Initilise the parent class
        Observation.__init__(self, resolution, npix, fov, sed, stars, survey)

        # Set up filter objects
        self.filters = filters

        # Set up img arrays. When multiple filters are provided we need a dict.
        self.img = np.zeros((self.npix, self.npix), dtype=np.float64)
        self.imgs = {f.filter_code:
                     np.zeros((self.npix, self.npix), dtype=np.float64)
                     for f in filters}

    def apply_filter(self, f):
        pass

    def get_psfed_img_single_filter(self):
        pass

    def get_noisy_img_single_filter(self):
        pass

    def get_psfed_img(self):
        pass

    def get_noisy_img(self):
        pass


class ParticleImage(ParticleObservation, Image):
    """
    The Image object, containing attributes and methods for calculating images.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None, positions=None, pixel_values=None):

        # Initilise the parent classes
        ParticleObservation.__init__(self, resolution, npix, fov, sed, stars,
                                     survey, positions)
        Image.__init__(self, resolution, npix, fov, sed, stars, survey)

        # If we have a list of filters make an IFU
        self._ifu_obj = None
        self.ifu = None
        if len(filters) > 0:
            self._ifu_obj = ParticleSpectralCube(sed, resolution, npix, fov,
                                                 stars, survey, positions)

        # Set up pixel values
        self.pixel_values = pixel_values

    def _get_hist_img_single_filter(self):
        """
        A generic function to calculate an image with no smoothing. This
        function returns a single image.
        Just a wrapper for numpy's histogramming function.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 2D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        self.img = np.histogram2d(self.pix_pos[:, 0], self.pix_pos[:, 1],
                                  bins=self.npix,
                                  range=((0, self.npix), (0, self.npix)),
                                  weights=self.pixel_values)[0]

        return self.img

    def get_smoothed_img_single_filter(self):
        pass

    def get_hist_img(self):
        """
        A generic function to calculate an image with no smoothing. This
        function returns a single image.
        Just a wrapper for numpy's histogramming function.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 2D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Handle the possible cases (multiple filters or single image)
        if self.pixel_values is not None:

            return self._get_hist_img_single_filter()

        # Calculate IFU "image"
        self.ifu = self._ifu_obj.get_hist_ifu()

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            self.imgs[f.filter_code] = self.apply_filter(f)

        return self.imgs

    def get_smoothed_img(self):
        pass


class ParametricImage(ParametricObservation, Image):
    """
    The Image object, containing attributes and methods for calculating images.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None, positions=None):

        # Initilise the parent classes
        ParametricObservation.__init__(self, resolution, npix, fov, sed, stars,
                                       survey)
        Image.__init__(self, resolution, npix, fov, sed, stars, survey)

        # If we have a list of filters make an IFU
        self._ifu_obj = None
        self.ifu = None
        if len(filters) > 0:
            self._ifu_obj = ParametricSpectralCube(sed, resolution, npix, fov,
                                                   stars, survey)
