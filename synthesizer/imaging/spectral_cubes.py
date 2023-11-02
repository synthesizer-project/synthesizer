""" Definitions for image objects
"""
import synthesizer.exceptions as exceptions
import numpy as np
import math
import warnings
from unyt import kpc, mas
from unyt.dimensions import length, angle
from synthesizer.imaging.scene import Scene, ParticleScene


class SpectralCube:
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

    def __init__(
        self,
        sed,
        rest_frame=True,
    ):
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

        # Set up the data cube dimensions
        self.spectral_resolution = sed.lam.size

        # Attribute to hold the IFU array. This is populated later and
        # allocated in the C extensions or when needed.
        self.ifu = None

        # Lets get the right SED from the object
        self.sed_values = None
        if rest_frame:
            # Get the rest frame SED
            self.sed_values = self.sed._lnu

        elif self.sed._fnu is not None:
            # Assign the flux
            self.sed_values = self.sed._fnu

        else:
            # Raise that we have inconsistent arguments
            raise exceptions.InconsistentArguments(
                "If rest_frame=False, i.e. an observed (flux) SED is requested"
                ",The flux must have been calculated with SED.get_fnu()"
            )

    def get_psfed_ifu(self):
        pass

    def get_noisy_ifu(self):
        pass


class ParticleSpectralCube(ParticleScene, SpectralCube):
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

    def __init__(
        self,
        sed,
        resolution,
        npix=None,
        fov=None,
        stars=None,
        positions=None,
        centre=None,
        psfs=None,
        depths=None,
        apertures=None,
        snrs=None,
        rest_frame=True,
        cosmo=None,
        redshift=None,
    ):
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
        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Check what we've been given handling Nones
        if stars is not None:
            self._check_flux_args(rest_frame, cosmo, stars.redshift)
        else:
            self._check_flux_args(rest_frame, cosmo, None)

        # Initilise the parent class
        ParticleScene.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            sed=sed,
            stars=stars,
            positions=positions,
            cosmo=cosmo,
            redshift=redshift,
            rest_frame=rest_frame,
        )
        SpectralCube.__init__(
            self,
            sed=sed,
            rest_frame=rest_frame,
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
            warnings.warn(
                "Warning, redshift not used when computing rest " "frame SEDs!"
            )

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

        # Set up the IFU array
        self.ifu = np.zeros(
            (self.npix, self.npix, self.spectral_resolution), dtype=np.float64
        )

        # Loop over positions including the sed
        for ind in range(self.npart):
            # Skip particles outside the FOV
            if (
                self.pix_pos[ind, 0] < 0
                or self.pix_pos[ind, 1] < 0
                or self.pix_pos[ind, 0] >= self.npix
                or self.pix_pos[ind, 1] >= self.npix
            ):
                continue

            self.ifu[self.pix_pos[ind, 0], self.pix_pos[ind, 1], :] += self.sed_values[
                ind, :
            ]

        return self.ifu

    def get_ifu(self):
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

        from .extensions.spectral_cube import make_ifu

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        # TODO: Would be more memory efficient to pass the position array
        #       and handle C extraction.
        sed_vals = np.ascontiguousarray(self.sed_values, dtype=np.float64)
        smls = np.ascontiguousarray(self.smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(self.coords[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(self.coords[:, 1], dtype=np.float64)
        zs = np.ascontiguousarray(self.coords[:, 2], dtype=np.float64)

        self.ifu = make_ifu(
            sed_vals,
            smls,
            xs,
            ys,
            zs,
            self.resolution,
            self.npix,
            self.coords.shape[0],
            self.spectral_resolution,
        )

        return self.ifu


class ParametricSpectralCube(Scene, SpectralCube):
    """
    The IFU/Spectral data cube object, used when creating parametric
    observations.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        morphology,
        sed,
        resolution,
        # depths=None,
        # apertures=None,
        npix=None,
        fov=None,
        cosmo=None,
        redshift=None,
        # snrs=None,
        rest_frame=True,
    ):
        # Initilise the parent classes
        Scene.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            sed=sed,
            rest_frame=rest_frame,
            cosmo=cosmo,
            redshift=redshift,
        )
        SpectralCube.__init__(
            self,
            sed=sed,
        )

        # Store the morphology object
        self.morphology = morphology

        # Compute the density grid based on the associated morphology
        self.density_grid = None
        if morphology is not None:
            self._get_density_grid()

    def _check_parametric_ifu_args(self, morphology):
        """
        Checks all arguments agree and do not conflict.

        Raises
        ----------
        """

        # check morphology has the correct method
        # this might not be generic enough
        if (self.spatial_unit == kpc) & (not morphology.model_kpc):
            raise exceptions.InconsistentArguments(
                "To create an image in kpc the morphology object must have a "
                "model defined in kpc. This can be achieved from a "
                "milliarcsecond input as long as a cosmology and redshift "
                "is provided to the Morphology object for the conversion."
            )

        if (self.spatial_unit == mas) & (not morphology.model_mas):
            raise exceptions.InconsistentArguments(
                "To create an image in milliarcsecond the morphology object "
                "must have a model defined in milliarcseconds. This can be "
                "achieved from a kpc input as long as a cosmology and redshift"
                " is provided to the Morphology object for the conversion."
            )

    def _get_density_grid(self):
        # Define 1D bin centres of each pixel
        if self.spatial_unit.dimensions == angle:
            res = (self.resolution * self.spatial_unit).to("mas").value
            bin_centres = res * np.linspace(-self.npix / 2, self.npix / 2, self.npix)
        else:
            res = (self.resolution * self.spatial_unit).to("kpc").value
            bin_centres = res * np.linspace(-self.npix / 2, self.npix / 2, self.npix)

        # Convert the 1D grid into 2D grids coordinate grids
        self._xx, self._yy = np.meshgrid(bin_centres, bin_centres)

        # Extract the density grid from the morphology function
        self.density_grid = self.morphology.compute_density_grid(
            self._xx, self._yy, units=self.spatial_unit
        )

        # And normalise it...
        self.density_grid /= np.sum(self.density_grid)

    def get_ifu(self):
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

        # Multiply the density grid by the sed to get the IFU
        self.ifu = self.density_grid[:, :, None] * self.sed_values

        return self.ifu
