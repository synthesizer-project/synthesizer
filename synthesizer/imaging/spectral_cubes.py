""" Definitions for image objects
"""
import math
import numpy as np
import synthesizer.exceptions as exceptions
from synthesizer.imaging.observations import Observation, ParticleObservation, ParametricObservation


class SpectralCube(Observation):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, stars=None,
                 survey=None):

        # Initilise the parent class
        Observation.__init__(resolution, npix, fov, sed, stars, survey)

        # Set up the data cube dimensions
        self.nwlengths = sed.lam.size

        # Assign pixel values
        self.pixel_values = sed.fnu

        # Set up the image itself (populated later)
        self.ifu = np.zeros((self.npix, self.npix, self.nwlengths),
                            dtype=np.float64)

    def get_psfed_ifu(self):
        pass

    def get_noisy_ifu(self):
        pass


class ParticleSpectralCube(ParticleObservation, SpectralCube):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, stars=None,
                 survey=None, positions=None):

        # Initilise the parent class
        ParticleObservation.__init__(
            resolution, npix, fov, sed, stars, survey, positions)
        SpectralCube.__init__(resolution, npix, fov, sed, stars, survey)

    def get_hist_ifu(self):
        """
        A generic function to calculate an image with no smoothing.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 3D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Loop over positions including the sed
        for ind in range(self.npart):

            self.ifu[self.pix_pos[ind, 0],
                     self.pix_pos[ind, 1], :] += self.pixel_values[ind, :]

        return self.ifu

    def get_smoothed_ifu(self):
        pass


class ParametricSpectralCube(ParametricObservation, SpectralCube):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, stars=None,
                 survey=None):

        # Initilise the parent class
        ParticleObservation.__init__(
            resolution, npix, fov, sed, stars, survey, positions=None)
        SpectralCube.__init__(resolution, npix, fov, sed, stars, survey)
