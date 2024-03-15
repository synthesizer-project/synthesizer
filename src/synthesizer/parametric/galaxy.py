"""
"""

import numpy as np

from synthesizer.base_galaxy import BaseGalaxy
from synthesizer import exceptions
from synthesizer.imaging import ImageCollection, SpectralCube
from synthesizer.art import Art, get_centred_art


class Galaxy(BaseGalaxy):
    """A class defining parametric galaxy objects"""

    def __init__(
        self,
        stars=None,
        name="parametric galaxy",
        black_holes=None,
        redshift=None,
        centre=None,
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

        # Set the type of galaxy
        self.galaxy_type = "Parametric"

        # Instantiate the parent
        BaseGalaxy.__init__(
            self,
            stars=stars,
            gas=None,
            black_holes=black_holes,
            redshift=redshift,
            centre=centre,
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
        Return the ionising photon luminosity (log10_specific_ionising_lum) for
        a given SFZH.

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

    def get_images_luminosity(
        self,
        resolution,
        fov,
        stellar_photometry=None,
        blackhole_photometry=None,
    ):
        """
        Make an ImageCollection from luminosities.

        Images are calculated by smoothing photometry over the component
        morphology. The photometry is taken from the Sed stored
        on a component under the key defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output image.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Make sure we have an image to make
        if stellar_photometry is None and blackhole_photometry is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_photometry or blackhole_photometry)!"
                " What component/s do you want images of?"
            )

        # Make stellar image if requested
        if stellar_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the density grid
            stellar_density = self.stars.morphology.get_density_grid(
                resolution, stellar_imgs.npix
            )

            # Make the image
            stellar_imgs.get_imgs_smoothed(
                photometry=self.stars.spectra[
                    stellar_photometry
                ].photo_luminosities,
                density_grid=stellar_density,
            )

        # Make blackhole image if requested
        if blackhole_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the density grid
            blackhole_density = self.black_holes.morphology.get_density_grid(
                resolution, blackhole_imgs.npix
            )

            # Compute the image
            blackhole_imgs.get_imgs_smoothed(
                photometry=self.black_holes.spectra[
                    blackhole_photometry
                ].photo_luminosities,
                density_grid=blackhole_density,
            )

        # Return the images, combining if there are multiple components
        if stellar_photometry is not None and blackhole_photometry is not None:
            return stellar_imgs + blackhole_imgs
        elif stellar_photometry is not None:
            return stellar_imgs
        return blackhole_imgs

    def get_images_flux(
        self,
        resolution,
        fov,
        stellar_photometry=None,
        blackhole_photometry=None,
    ):
        """
        Make an ImageCollection from fluxes.

        Images are calculated by smoothing photometry over the component
        morphology. The photometry is taken from the Sed stored
        on a component under the key defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output image.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Make sure we have an image to make
        if stellar_photometry is None and blackhole_photometry is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_photometry or blackhole_photometry)!"
                " What component/s do you want images of?"
            )

        # Make stellar image if requested
        if stellar_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the density grid
            stellar_density = self.stars.morphology.get_density_grid(
                resolution, stellar_imgs.npix
            )

            # Make the image
            stellar_imgs.get_imgs_smoothed(
                photometry=self.stars.spectra[stellar_photometry].photo_fluxes,
                density_grid=stellar_density,
            )

        # Make blackhole image if requested
        if blackhole_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the density grid
            blackhole_density = self.black_holes.morphology.get_density_grid(
                resolution, blackhole_imgs.npix
            )

            # Compute the image
            blackhole_imgs.get_imgs_smoothed(
                photometry=self.black_holes.spectra[
                    blackhole_photometry
                ].photo_fluxes,
                density_grid=blackhole_density,
            )

        # Return the images, combining if there are multiple components
        if stellar_photometry is not None and blackhole_photometry is not None:
            return stellar_imgs + blackhole_imgs
        elif stellar_photometry is not None:
            return stellar_imgs
        return blackhole_imgs

    def get_data_cube(
        self,
        resolution,
        fov,
        lam,
        stellar_spectra=None,
        blackhole_spectra=None,
        quantity="lnu",
    ):
        """
        Make a SpectralCube from an Sed.

        Data cubes are calculated by smoothing spectra over the component
        morphology. The Sed used is defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output data cube.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            lam (unyt_array, float)
                The wavelength array to use for the data cube.
            stellar_spectra (string)
                The stellar spectra key to make into a data cube.
            blackhole_spectra (string)
                The black hole spectra key to make into a data cube.
            quantity (str):
                The Sed attribute/quantity to sort into the data cube, i.e.
                "lnu", "llam", "luminosity", "fnu", "flam" or "flux".

        Returns:
            SpectralCube
                The spectral data cube object containing the derived
                data cube.
        """
        # Make sure we have an image to make
        if stellar_spectra is None and blackhole_spectra is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_spectra or blackhole_spectra)!"
                " What component/s do you want a data cube of?"
            )

        # Make stellar image if requested
        if stellar_spectra is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_cube = SpectralCube(
                resolution=resolution, fov=fov, lam=lam
            )

            # Compute the density grid
            stellar_density = self.stars.morphology.get_density_grid(
                resolution, stellar_cube.npix
            )

            # Make the image
            stellar_cube.get_data_cube_smoothed(
                sed=self.stars.spectra[stellar_spectra],
                density_grid=stellar_density,
                quantity=quantity,
            )

        # Make blackhole image if requested
        if blackhole_spectra is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_cube = SpectralCube(
                resolution=resolution, fov=fov, lam=lam
            )

            # Compute the density grid
            blackhole_density = self.black_holes.morphology.get_density_grid(
                resolution, blackhole_cube.npix
            )

            # Compute the image
            blackhole_cube.get_data_cube_smoothed(
                sed=self.black_holes.spectra[blackhole_spectra],
                density_grid=blackhole_density,
                quantity=quantity,
            )

        # Return the images, combining if there are multiple components
        if stellar_spectra is not None and blackhole_spectra is not None:
            return stellar_cube + blackhole_cube
        elif stellar_spectra is not None:
            return stellar_cube
        return blackhole_cube
