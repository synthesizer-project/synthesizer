"""
"""
import numpy as np

from synthesizer.base_galaxy import BaseGalaxy
from synthesizer import exceptions
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.imaging.images import ParametricImage
from synthesizer.art import Art
from synthesizer.particle import Stars as ParticleStars


class Galaxy(BaseGalaxy):

    """A class defining parametric galaxy objects"""

    def __init__(
        self,
        stars,
        name="parametric galaxy",
        redshift=None,
    ):
        """__init__ method for ParametricGalaxy

        Args:
            stars (parametric.Stars)
                An instance of parametric.Stars containing the combined star
                formation and metallicity history of this galaxy.
            name (str)
                A name to identify the galaxy. Only used for external labelling,
                has no internal use.
            redshift (float)
                The redshift of the galaxy.

        Raises:
            InconsistentArguments
        """

        # Check we haven't been given Stars
        if isinstance(stars, ParticleStars):
            raise exceptions.InconsistentArguments(
                "Stars passed instead of SFZH object (Stars)."
                " Did you mean synthesizer.particle.Galaxy instead?"
            )

        self.name = name

        # Attach the parametric stars object
        self.stars = stars

        # add an extra dimension to the sfzh to allow the fast summation
        # **** TODO: Get rid of this expression or
        # use this throughout?
        self.sfzh = self.stars.sfzh

        # Define the dictionary to hold spectra
        self.spectra = {}

        # Define the dictionary to hold images
        self.images = {}

        # The redshift of the galaxy
        self.redshift = redshift

    def __str__(self):
        """Function to print a basic summary of the Galaxy object.

        Returns a string containing the total mass formed and lists of the
        available SEDs, lines, and images.

        Returns
        -------
        str
            Summary string containing the total mass formed and lists of the
            available SEDs, lines, and images.
        """

        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF PARAMETRIC GALAXY" + "\n"
        pstr += Art.galaxy + "\n"
        pstr += str(self.__class__) + "\n"
        pstr += (
            f"log10(stellar mass formed/Msol): \
            {np.log10(np.sum(self.sfzh))}"
            + "\n"
        )
        pstr += "available SEDs: \n"
        pstr += f"    Stellar:  {list(self.stars.spectra.keys())}" + "\n"
        pstr += f"    Combined: {list(self.spectra.keys())}" + "\n"
        pstr += f"available lines: {list(self.stars.lines.keys())}" + "\n"
        pstr += f"available images: {list(self.images.keys())}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr

    def __add__(self, second_galaxy):
        """Allows two Galaxy objects to be added together.

        Parameters
        ----------
        second_galaxy : ParametricGalaxy
            A second ParametricGalaxy to be added to this one.

        NOTE: functionality for adding lines and images not yet implemented.

        Returns
        -------
        ParametricGalaxy
            New ParametricGalaxy object containing summed SFZHs, SEDs, lines,
            and images.
        """

        # Sum the Stellar populations
        new_stars = self.stars + second_galaxy.stars

        # Create the new galaxy
        new_galaxy = Galaxy(new_stars)

        # add together spectra
        for spec_name, spectra in self.stars.spectra.items():
            if spec_name in second_galaxy.stars.spectra.keys():
                new_galaxy.stars.spectra[spec_name] = (
                    spectra + second_galaxy.stars.spectra[spec_name]
                )
            else:
                raise exceptions.InconsistentAddition(
                    "Both galaxies must contain the same spectra to be \
                    added together"
                )

        # add together lines
        for line_type in self.stars.lines.keys():
            new_galaxy.spectra.lines[line_type] = {}

            if line_type not in second_galaxy.stars.lines.keys():
                raise exceptions.InconsistentAddition(
                    "Both galaxies must contain the same sets of line types \
                        (e.g. intrinsic / attenuated)"
                )
            else:
                for line_name, line in self.stars.lines[line_type].items():
                    if line_name in second_galaxy.stars.lines[line_type].keys():
                        new_galaxy.stars.lines[line_type][line_name] = (
                            line + second_galaxy.stars.lines[line_type][line_name]
                        )
                    else:
                        raise exceptions.InconsistentAddition(
                            "Both galaxies must contain the same emission \
                                lines to be added together"
                        )

        # add together images
        for img_name, image in self.images.items():
            if img_name in second_galaxy.images.keys():
                new_galaxy.images[img_name] = image + second_galaxy.images[img_name]
            else:
                raise exceptions.InconsistentAddition(
                    (
                        "Both galaxies must contain the same"
                        " images to be added together"
                    )
                )

        return new_galaxy

    def get_Q(self, grid):
        """
        Return the ionising photon luminosity (log10Q) for a given SFZH.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.

        Returns:
            Log of the ionising photon luminosity over the grid dimensions
        """

        return np.sum(10 ** grid.log10Q["HI"] * self.sfzh, axis=(0, 1))

    def make_images(
        self,
        resolution,
        fov=None,
        sed=None,
        filters=(),
        psfs=None,
        depths=None,
        snrs=None,
        aperture=None,
        noises=None,
        rest_frame=True,
        cosmo=None,
        redshift=None,
        psf_resample_factor=1,
    ):
        """
        Makes images in each filter provided in filters. Additionally an image
        can be made with or without a PSF and noise.
        NOTE: Either npix or fov must be defined.

        Parameters
        ----------
        resolution : float
           The size of a pixel.
           (Ignoring any supersampling defined by psf_resample_factor)
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.
        sed : obj (SED)
            An sed object containing the spectra for this image.
        filters : obj (FilterCollection)
            An imutable collection of Filter objects. If provided images are
            made for each filter.
        psfs : dict
            A dictionary containing the psf in each filter where the key is
            each filter code and the value is the psf in that filter.
        depths : dict
            A dictionary containing the depth of an observation in each filter
            where the key is each filter code and the value is the depth in
            that filter.
        aperture : float/dict
            Either a float describing the size of the aperture in which the
            depth is defined or a dictionary containing the size of the depth
            aperture in each filter.
        rest_frame : bool
            Are we making an observation in the rest frame?
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        redshift : float
            The redshift of the observation. Used when converting between
            physical cartesian coordinates and angular coordinates.
        psf_resample_factor : float
            The factor by which the image should be resampled for robust PSF
            convolution. Note the images after PSF application will be
            downsampled to the native pixel scale.
        Returns
        -------
        Image : array-like
            A 2D array containing the image.
        """

        # Handle a super resolution image
        if psf_resample_factor is not None:
            if psf_resample_factor != 1:
                resolution /= psf_resample_factor

        # Instantiate the Image object.
        img = ParametricImage(
            morphology=self.stars.morphology,
            resolution=resolution,
            fov=fov,
            sed=sed,
            filters=filters,
            rest_frame=rest_frame,
            redshift=redshift,
            cosmo=cosmo,
            psfs=psfs,
            depths=depths,
            apertures=aperture,
            snrs=snrs,
        )

        # Compute image
        img.get_imgs()

        if psfs is not None:
            # Convolve the image/images
            img.get_psfed_imgs()

            # Downsample to the native resolution if we need to.
            if psf_resample_factor is not None:
                if psf_resample_factor != 1:
                    img.downsample(1 / psf_resample_factor)

        if depths is not None or noises is not None:
            img.get_noisy_imgs(noises)

        return img
