""" Definitions for image objects
"""
import warnings
import math
import numpy as np
import synthesizer.exceptions as exceptions
from synthesizer.imaging.observation import Observation, ParticleObservation, ParametricObservation


class SpectralCube(Observation):
    """
    The generic parent IFU/Spectral data cube object, containing common
    attributes and methods for both particle and parametric sIFUs.

    Attributes
    ----------
    spectral_resolution : int
        The number of wavelengths in the spectra, "the resolution".
    ifu : array-like (float)
        The spectral data cube itself. [npix, npix, spectral_resolution]

    """

    def __init__(self, sed, resolution, npix=None, fov=None, survey=None):
        """
        Intialise the SpectralCube.

        Parameters
        ----------
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        resolution : float
            The size a pixel.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU. 
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        survey : obj (Survey)
            WorkInProgress

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Initilise the parent class
        Observation.__init__(self, resolution=resolution, npix=npix, fov=fov,
                             sed=sed, survey=survey)

        # Set up the data cube dimensions
        self.spectral_resolution = sed.lam.size

        # Set up the image itself (populated later)
        self.ifu = np.zeros((self.npix, self.npix, self.spectral_resolution),
                            dtype=np.float64)

    def get_psfed_ifu(self):
        pass

    def get_noisy_ifu(self):
        pass


class ParticleSpectralCube(ParticleObservation, SpectralCube):
    """
    The IFU/Spectral data cube object, used when creating observations from
    particle distributions.

    Attributes
    ----------
    sed_values : array-like (float)
        The number of wavelengths in the spectra, "the resolution".

    Methods
    -------
    get_hist_ifu
        Sorts particles into singular pixels. In each pixel the spectrum of a
        particle is added along the wavelength axis.
    get_smoothed_ifu
        Sorts particles into pixels, smoothing by a user provided kernel. Each
        pixel accumalates a contribution from the spectrum of all particles
        whose kernel includes that pixel, adding this contribution along the
        wavelength axis.

    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.

    """

    def __init__(self, sed, resolution, npix=None, fov=None, stars=None,
                 survey=None, positions=None, centre=None, rest_frame=True,
                 redshift=None, cosmo=None, igm=None):
        """
        Intialise the ParticleSpectralCube.

        Parameters
        ----------
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        resolution : float
            The size a pixel.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU. 
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        stars : obj (Stars)
            The object containing the stars to be placed in a image.
        survey : obj (Survey)
            WorkInProgress
        positons : array-like (float)
            The position of particles to be sorted into the image.
        centre : array-like (float)
            The coordinates around which the image will be centered. The if one
            is not provided then the geometric centre is calculated and used.
        rest_frame : bool
            Are we making an observation in the rest frame?
        redshift : float
            The redshift of the observation. Used when converting rest frame
            luminosity to flux.
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        igm : obj (Inoue14/Madau96)
            Object containing the absorbtion due to an intergalactic medium.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Check what we've been given
        self._check_flux_args(rest_frame, cosmo, redshift)

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

    def _check_flux_args(self, rest_frame, cosmo, redshift):
        """
        Ensures we have a valid combination of inputs.


        Parameters
        ----------
        rest_frame : bool
            Are we making an observation in the rest frame?
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        redshift : float
            The redshift of the observation. Used when converting rest frame
            luminosity to flux.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Warn that specifying redshift does nothing for rest frame observations
        if rest_frame and redshift is not None:
            warnings.warn("Warning, redshift not used when computing rest "
                          "frame SEDs!")

        if not rest_frame and (redshift is None or cosmo is None):
            raise exceptions.InconsistentArguments(
                "For observations not in the rest frame both the redshift and "
                "a cosmology object must be specified!"
            )

    def get_hist_ifu(self):
        """
        A method to calculate an IFU with no smoothing.

        Returns
        -------
        img : array_like (float)
            A 3D array containing the pixel values sorted into individual
            pixels. [npix, npix, spectral_resolution]
        """

        # Loop over positions including the sed
        for ind in range(self.npart):
            self.ifu[self.pix_pos[ind, 0],
                     self.pix_pos[ind, 1], :] += self.sed_values[ind, :]

        return self.ifu

    def get_smoothed_ifu(self, kernel_func):
        """
        A method to calculate an IFU with smoothing. Here the particles are
        smoothed over a kernel, i.e. the full wavelength range of each
        particles spectrum is multiplied by the value of the kernel in each
        pixel it occupies.

        Parameters
        ----------
        kernel_func : function
            A function describing the smoothing kernel that returns a single
            number between 0 and 1. This function can be imported from the
            options in kernel_functions.py or can be user defined. If user
            defined the function must return the kernel value corredsponding
            to the position of a particle with smoothing length h at distance
            r from the centre of the kernel (r/h). 

        Returns
        -------
        img : array_like (float)
            A 3D array containing the pixel values sorted into individual
            pixels. [npix, npix, spectral_resolution]
        """

        # Get the size of a pixel
        res = self.resolution

        # Loop over positions including the sed
        for ind in range(self.npart):

            # Get this particles smoothing length and position
            smooth_length = self.stars.smoothing_lengths[ind]
            pos = self.coords[ind]

            # How many pixels are in the smoothing length?
            delta_pix = math.ceil(smooth_length / self.resolution) + 1

            # Loop over a square aperture around this particle
            # NOTE: This includes "pixels" in front of and behind the image
            #       plane since the kernel is by defintion 3D
            # TODO: Would be considerably more accurate to integrate over the
            #       kernel in z axis since this is not quantised into pixels
            #       like the axes in the image plane.
            for i in range(self.pix_pos[ind, 0] - delta_pix,
                           self.pix_pos[ind, 0] + delta_pix + 1):

                # Skip if outside of image
                if i < 0 or i >= self.npix:
                    continue

                # Compute the x separation
                x_dist = (i * res) + (res / 2) - pos[0]
                
                for j in range(self.pix_pos[ind, 1] - delta_pix,
                               self.pix_pos[ind, 1] + delta_pix + 1):

                    # Skip if outside of image
                    if j < 0 or j >= self.npix:
                        continue

                    # Compute the y separation
                    y_dist = (j * res) + (res / 2) - pos[1]

                    for k in range(self.pix_pos[ind, 2] - delta_pix,
                                   self.pix_pos[ind, 2] + delta_pix + 1):

                        # Compute the z separation
                        z_dist = (k * res) + (res / 2) - pos[2]

                        # Compute the distance between the centre of this pixel
                        # and the particle.
                        dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

                        # Get the value of the kernel here
                        kernel_val = kernel_func(dist / smooth_length)

                        # Add this pixel's contribution
                        self.ifu[i, j, :] += self.sed_values[ind, :] * kernel_val

        return self.ifu


class ParametricSpectralCube(ParametricObservation, SpectralCube):
    """
    The IFU/Spectral data cube object, used when creating parametric
    observations.

    WorkInProgress

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
