"""
"""
import numpy as np

from synthesizer.base_galaxy import BaseGalaxy
from synthesizer import exceptions
from synthesizer.imaging.images import ParametricImage
from synthesizer.art import Art, get_centred_art
from synthesizer.particle import Stars as ParticleStars


class Galaxy(BaseGalaxy):

    """A class defining parametric galaxy objects"""

    def __init__(
        self,
        stars=None,
        name="parametric galaxy",
        black_holes=None,
        redshift=None,
    ):
        """__init__ method for ParametricGalaxy

        Args:
            stars (parametric.Stars)
                An instance of parametric.Stars containing the combined star
                formation and metallicity history of this galaxy.
            name (str)
                A name to identify the galaxy. Only used for external
                labelling, has no internal use.
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

        # Set the type of galaxy
        self.galaxy_type = "Parametric"

        # Instantiate the parent
        BaseGalaxy.__init__(
            self,
            stars=stars,
            gas=None,
            black_holes=black_holes,
            redshift=redshift,
        )

        # The name
        self.name = name

        # Local pointer to SFZH array
        self.sfzh = self.stars.sfzh

        # Define the dictionary to hold spectra
        self.spectra = {}

        # Define the dictionary to hold images
        self.images = {}

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

        # Define the width to print within
        width = 79

        pstr = ""
        pstr += "-" * width + "\n"
        pstr += "SUMMARY OF PARAMETRIC GALAXY".center(width + 4) + "\n"
        pstr += get_centred_art(Art.galaxy, width) + "\n"
        pstr += str(self.__class__) + "\n"
        pstr += f"log10(stellar mass formed/Msol): \
            {np.log10(np.sum(self.sfzh))}\n"
        pstr += "available SEDs: \n"

        # Define the connecting character for list wrapping
        conn_char = "\n" + (15 * " ")

        # Print stellar spectra keys
        if self.stars is not None and len(self.stars.spectra) > 0:
            # Print keys nicely so they don't spill over
            spectra_keys = [""]
            iline = 0
            for key in self.stars.spectra.keys():
                if len(spectra_keys[iline]) + len(key) + 2 < width - 14:
                    spectra_keys[iline] += key + ", "
                else:
                    iline += 1
                    spectra_keys.append("")

            # Slice off the last two entries, we don't need then
            spectra_keys[iline] = spectra_keys[iline][:-2]

            pstr += "    Stellar:  [" + conn_char.join(spectra_keys) + "]\n"

        else:
            pstr += "    Stellar:  []\n"

        # Define the connecting character for list wrapping
        conn_char = "\n" + (19 * " ")

        # Print stellar spectra keys
        if self.black_holes is not None and len(self.black_holes.spectra) > 0:
            # Print keys nicely so they don't spill over
            spectra_keys = [""]
            iline = 0
            for key in self.black_holes.spectra.keys():
                if len(spectra_keys[iline]) + len(key) + 2 < width - 19:
                    spectra_keys[iline] += key + ", "
                else:
                    iline += 1
                    spectra_keys.append("")

            # Slice off the last two entries, we don't need then
            spectra_keys[iline] = spectra_keys[iline][:-2]

            pstr += (
                "    Black Holes:  [" + conn_char.join(spectra_keys) + "]\n"
            )

        else:
            pstr += "    Black Holes:  []" + "\n"

        # Define the connecting character for list wrapping
        conn_char = "\n" + (15 * " ")

        # Print combined spectra keys
        if len(self.spectra) > 0:
            # Print keys nicely so they don't spill over
            spectra_keys = [""]
            iline = 0
            for key in self.spectra.keys():
                if len(spectra_keys[iline]) + len(key) + 2 < width - 14:
                    spectra_keys[iline] += key + ", "
                else:
                    iline += 1
                    spectra_keys.append("")

            # Slice off the last two entries, we don't need then
            spectra_keys[iline] = spectra_keys[iline][:-2]

            pstr += (
                "    Combined: [" + conn_char.join(spectra_keys) + "]" + "\n"
            )

        else:
            pstr += "    Combined: []" + "\n"

        # Print line keys
        if len(self.stars.lines) > 0:
            # Redefine the connecting character for list wrapping
            conn_char = "\n" + (18 * " ")

            # Print keys nicely so they don't spill over
            line_keys = [""]
            iline = 0
            for key in self.stars.lines.keys():
                if len(line_keys[iline]) + len(key) + 2 < width - 14:
                    line_keys[iline] += key + ", "
                else:
                    iline += 1
                    line_keys.append("")

            # Slice off the last two entries, we don't need then
            line_keys[iline] = line_keys[iline][:-2]

            pstr += (
                "available lines: [" + conn_char.join(line_keys) + "]" + "\n"
            )

        else:
            pstr += "available lines: []" + "\n"

        # Print image keys
        if len(self.images) > 0:
            # Redefine the connecting character for list wrapping
            conn_char = "\n" + (19 * " ")

            # Print keys nicely so they don't spill over
            img_keys = [""]
            iline = 0
            for key in self.images.keys():
                if len(img_keys[iline]) + len(key) + 2 < width - 14:
                    img_keys[iline] += key + ", "
                else:
                    iline += 1
                    img_keys.append("")

            # Slice off the last two entries, we don't need then
            img_keys[iline] = img_keys[iline][:-2]

            pstr += (
                "available images: [" + conn_char.join(img_keys) + "]" + "\n"
            )

        else:
            pstr += "available images: []" + "\n"

        pstr += "-" * width + "\n"
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
                    if (
                        line_name
                        in second_galaxy.stars.lines[line_type].keys()
                    ):
                        new_galaxy.stars.lines[line_type][line_name] = (
                            line
                            + second_galaxy.stars.lines[line_type][line_name]
                        )
                    else:
                        raise exceptions.InconsistentAddition(
                            "Both galaxies must contain the same emission \
                                lines to be added together"
                        )

        # add together images
        for img_name, image in self.images.items():
            if img_name in second_galaxy.images.keys():
                new_galaxy.images[img_name] = (
                    image + second_galaxy.images[img_name]
                )
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
        Return the ionising photon luminosity (log10_specific_ionising_lum) for a given SFZH.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.

        Returns:
            Log of the ionising photon luminosity over the grid dimensions
        """

        return np.sum(
            10 ** grid.log10_specific_ionising_lum["HI"] * self.sfzh,
            axis=(0, 1),
        )

    def make_images(
        self,
        resolution,
        fov=None,
        npix=None,
        stellar_spectra_type=None,
        blackhole_spectra_type=None,
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
        Makes images in each filter provided in filters for either specific
        components of the galaxy or any combination.

        Additionally an image can be made with or without a PSF and noise.

        If multiple components are requested they will be combined into a
        single output image.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (float)
               The size of a pixel.
               (Ignoring any supersampling defined by psf_resample_factor)
            npix (int)
                The number of pixels along an axis.
            fov : float
                The width of the image in image coordinates.
            stellar_spectra_type (string)
                The stellar spectra key to use for the image.
            blackhole_spectra_type (string)
                The black hole spectra key to use for the image.
            filters (FilterCollection)
                An imutable collection of Filter objects. If provided images
                are made for each filter.
            psfs (dict)
                A dictionary containing the psf in each filter where the key is
                each filter code and the value is the psf in that filter.
            depths (dict)
                A dictionary containing the depth of an observation in each
                filter where the key is each filter code and the value is
                the depth in that filter.
            aperture (float/dict)
                Either a float describing the size of the aperture in which the
                depth is defined or a dictionary containing the size of the
                depth aperture in each filter.
            rest_frame (bool)
                Are we making an observation in the rest frame?
            cosmo : obj (astropy.cosmology)
                A cosmology object from astropy, used for cosmological
                calculations when converting rest frame luminosity to flux.
            redshift (float)
                The redshift of the observation. Used when converting between
                physical cartesian coordinates and angular coordinates.
            psf_resample_factor (float)
                The factor by which the image should be resampled for robust
                PSF convolution. Note the images after PSF application will
                be downsampled to the native pixel scale.

        Returns:
            Image
                An instance of an Image object contain the created images.
        """

        # Handle a super resolution image
        if psf_resample_factor is not None:
            if psf_resample_factor != 1:
                resolution /= psf_resample_factor

        # Make sure we have an image to make
        if stellar_spectra_type is None and blackhole_spectra_type is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_spectra_type or blackhole_spectra_type)!"
                " What component do you want images of?"
            )

        # Make stellar image if requested
        if stellar_spectra_type is not None:
            # Instantiate the Image object.
            stellar_img = ParametricImage(
                morphology=self.stars.morphology,
                resolution=resolution,
                fov=fov,
                npix=npix,
                sed=self.stars.spectra[stellar_spectra_type],
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
            stellar_img.get_imgs()

            if psfs is not None:
                # Convolve the image/images
                stellar_img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        stellar_img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                stellar_img.get_noisy_imgs(noises)

        # Make black hole image if requested
        if blackhole_spectra_type is not None:
            # Instantiate the Image object.
            blackhole_img = ParametricImage(
                morphology=self.black_holes.morphology,
                resolution=resolution,
                fov=fov,
                npix=npix,
                sed=self.black_holes.spectra[blackhole_spectra_type],
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
            blackhole_img.get_imgs()

            if psfs is not None:
                # Convolve the image/images
                blackhole_img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        blackhole_img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                blackhole_img.get_noisy_imgs(noises)

        # Combine images
        if stellar_spectra_type is not None and blackhole_spectra_type is None:
            img = stellar_img
        elif (
            stellar_spectra_type is not None
            and blackhole_spectra_type is not None
        ):
            img = stellar_img + blackhole_img
        else:
            img = blackhole_img

        return img
