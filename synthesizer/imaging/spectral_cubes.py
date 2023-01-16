""" Definitions for image objects
"""
import math
import numpy as np
import synthesizer.exceptions as exceptions
from synthesizer.imaging.observation import Observation, ParticleObservation, ParametricObservation


class SpectralCube(Observation):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, survey=None):

        # Initilise the parent class
        Observation.__init__(self, resolution=resolution, npix=npix, fov=fov,
                             sed=sed, survey=survey)

        # Set up the data cube dimensions
        self.nwlengths = sed.lam.size

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
                 survey=None, positions=None, centre=None, img_instance=None,
                 rest_frame=True, redshift=None, cosmo=None, igm=None):

        # Initilise the parent class
        ParticleObservation.__init__(self, resolution=resolution, npix=npix,
                                     fov=fov, sed=sed, stars=stars,
                                     survey=survey, positions=positions)
        SpectralCube.__init__(self, sed=sed, resolution=resolution, npix=npix,
                              fov=fov, survey=survey)

        # Lets get the right SED form the object
        self.sed_values = None
        if rest_frame and redshift is None:

            # Get the rest frame SED (this is both sed.fnu0 and sed.lnu)
            self.sed_values = self.sed.lnu
            
        elif redshift is not None and cosmo is not None:

            # Check if we need to calculate sed.fnu, if not calculate it
            if self.sed.fnu is None:
                self.sed.get_fnu(cosmo, redshift, igm)

            # Assign the flux 
            self.sed_values = self.sed.fnu

        else:

            # Raise that we have inconsistent arguments
            raise exceptions.InconsistentArguments(
                "If rest_frame=False, i.e. an observed (flux) SED is requested"
                ", both an cosmo object (from astropy) and the redshift of the"
                " observation must be supplied."
            )

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
                     self.pix_pos[ind, 1], :] += self.sed_values[ind, :]

        return self.ifu

    def get_smoothed_ifu(self, kernel_func):
        """
        A generic function to calculate an image with smoothing.


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

        # Get the size of a pixel
        res = self.resolution

        # Loop over positions including the sed
        for ind in range(self.npart):

            # Get this particles smoothing length and position
            smooth_length = self.stars.smoothing_lengths[ind]
            pos = self.shifted_sim_pos[ind]

            # How many pixels are in the smoothing length?
            delta_pix = math.ceil(smooth_length / self.resolution) + 1

            # Loop over a square aperture around this particle
            for i in range(self.pix_pos[ind, 0] - delta_pix,
                           self.pix_pos[ind, 0] + delta_pix + 1):

                # Skip if outside of image
                if i < 0 or i >= self.npix:
                    continue
                
                for j in range(self.pix_pos[ind, 1] - delta_pix,
                               self.pix_pos[ind, 1] + delta_pix + 1):

                    # Skip if outside of image
                    if j < 0 or j >= self.npix:
                        continue

                    # Compute the distance between the centre of this pixel
                    # and the particle.
                    x_dist = (i * res) + (res / 2) - pos[0]
                    y_dist = (j * res) + (res / 2) - pos[1]
                    dist = np.sqrt(x_dist ** 2 + y_dist ** 2)

                    # Get the value of the kernel here
                    kernel_val = kernel_func(dist / smooth_length)

                    # Add this pixel's contribution
                    self.ifu[i, j, :] += self.sed_values[ind, :] * kernel_val

        return self.ifu


class ParametricSpectralCube(ParametricObservation, SpectralCube):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, survey=None):

        # Initilise the parent class
        ParametricObservation.__init__(self, resolution=resolution, npix=npix,
                                       fov=fov, sed=sed, survey=survey)
        SpectralCube.__init__(self, sed=sed, resolution=resolution, npix=npix,
                              fov=fov, survey=survey)
