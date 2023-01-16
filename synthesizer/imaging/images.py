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
    
    def __init__(self, resolution, npix=None, fov=None, filters=(), sed=None,
                 survey=None):

        # Sanitize inputs
        if filters is None:
            filters = ()

        # Initilise the parent class
        Observation.__init__(self, resolution=resolution, npix=npix, fov=fov,
                             sed=sed, survey=survey)

        # Intialise IFU attributes
        self._ifu_obj = None
        self.ifu = None

        # Set up filter objects
        self.filters = filters

        # Set up img arrays. When multiple filters are provided we need a dict.
        self.img = np.zeros((self.npix, self.npix), dtype=np.float64)
        self.imgs = {}

    def apply_filter(self, f):
        """
        Applies a filters transmission curve defined by a Filter object to an
        IFU to get a single band image.

        Constructing an IFU first and applying the filter to the image will
        always be faster than calculating particle photometry and making
        multiple images when there are multiple bands. This way the image is
        made once as an IFU and the filter application which is much faster
        is done for each band.

        Parameters
        ----------
        f : obj (Filter)
            The Filter object containing all filter information.

        Returns
        -------
        img : array-like (float)
             A single band image in this filter with shape (npix, npix).
        """

        # Get the mask that removes wavelengths we don't currently care about
        in_band = f.t > 0

        # Multiply the IFU by the filter transmission curve
        transmitted = self.ifu[:, :, in_band] * f.t[in_band]

        # Sum over the final axis to "collect" transmission in this filer
        img = np.sum(transmitted, axis=-1)

        return img

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

    TODO: could enable filter application to individul SEDs if there is
          only 1 filter.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None, positions=None, pixel_values=None,
                  rest_frame=True, redshift=None, cosmo=None, igm=None):
        
        # Initilise the parent classes
        ParticleObservation.__init__(self, resolution=resolution, npix=npix,
                                     fov=fov, sed=sed, stars=stars,
                                     survey=survey, positions=positions)
        Image.__init__(self, resolution=resolution, npix=npix, fov=fov,
                       filters=filters, sed=sed, survey=survey)

        # If we have a list of filters make an IFU
        if len(filters) > 0:
            self._ifu_obj = ParticleSpectralCube(sed=self.sed,
                                                 resolution=self.resolution,
                                                 npix=self.npix, fov=self.fov,
                                                 stars=self.stars,
                                                 survey=self.survey,
                                                 rest_frame=rest_frame,
                                                 redshift=redshift, cosmo=cosmo,
                                                 igm=igm)

        # # If we have a list of filters make an IFU
        # if len(filters) > 0:
        #     self._ifu_obj = ParticleSpectralCube(sed=self.sed,
        #                                          resolution=self.resolution,
        #                                          npix=self.npix, fov=self.fov,
        #                                          stars=self.stars,
        #                                          survey=self.survey)

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
        A generic function to calculate an image with no smoothing.


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

    def get_smoothed_img(self, kernel_func):
        """
        A generic function to calculate an image with smoothing based on a
        kernel.


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
            print("NotYetImplemented")
            return self.img
        
        # Calculate IFU "image"
        self.ifu = self._ifu_obj.get_smoothed_ifu(kernel_func)

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            self.imgs[f.filter_code] = self.apply_filter(f)

        return self.imgs


class ParametricImage(ParametricObservation, Image):
    """
    The Image object, containing attributes and methods for calculating images.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, filters=(),
                 survey=None):

        # Initilise the parent classes
        ParametricObservation.__init__(self, resolution=resolution, npix=npix,
                                       fov=fov, sed=sed, survey=survey)
        Image.__init__(self, resolution=resolution, npix=npix, fov=fov,
                       filters=filters, sed=sed, survey=survey)

        # If we have a list of filters make an IFU
        if len(filters) > 0:
            self._ifu_obj = ParametricSpectralCube(sed, resolution, npix, fov,
                                                   stars, survey)
