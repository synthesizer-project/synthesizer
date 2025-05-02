"""A submodule defining a parametric galaxy object.

This module defines a parametric galaxy object, which is a subclass of the
BaseGalaxy class. The parametric galaxy object is used to represent a galaxy
comprised of parametric components.

Example usage:

    from synthesizer.parametric import Galaxy

    # Create a parametric galaxy object
    galaxy = Galaxy(stars=stars, black_holes=black_holes, redshift=0.1)

    # Get the galaxies spectra
    spectra = galaxy.get_spectra(model)

    # Get the ionising photon luminosity for a given SFZH
    ionising_photon_luminosity = galaxy.get_Q(grid)

    # Create a spectral cube from the galaxy's spectra
    spectral_cube = galaxy.get_data_cube(resolution=0.1, fov=10, lam=lam)

    # Add two parametric galaxies together
    new_galaxy = galaxy1 + galaxy2

"""

import numpy as np
from unyt import Mpc

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.imaging import SpectralCube
from synthesizer.synth_warnings import deprecated
from synthesizer.units import accepts


class Galaxy(BaseGalaxy):
    """A class defining parametric galaxy objects.

    This class is a subclass of the BaseGalaxy class and is used to
    represent a galaxy comprised of parametric components. The class
    provides methods for creating and manipulating parametric galaxies,
    including adding galaxies together, getting spectra, and creating
    spectral cubes.

    Attributes:
        stars (Stars):
            An instance of Stars containing the combined star
            formation and metallicity history of this galaxy.
        name (str):
            A name to identify the galaxy. Only used for external
            labelling, has no internal use.
        redshift (float):
            The redshift of the galaxy.
        centre (unyt_array):
            The centre of the galaxy.
        black_holes (BlackHole):
            An instance of BlackHole containing the black hole
            particle data.
    """

    @accepts(centre=Mpc)
    def __init__(
        self,
        stars=None,
        name="parametric galaxy",
        black_holes=None,
        redshift=None,
        centre=None,
        **kwargs,
    ):
        """Initialise a parametric galaxy object.

        Args:
            stars (Stars):
                An instance of Stars containing the combined star
                formation and metallicity history of this galaxy.
            name (str):
                A name to identify the galaxy. Only used for external
                labelling, has no internal use.
            redshift (float):
                The redshift of the galaxy.
            centre (unyt_array):
                The centre of the galaxy.
            black_holes (BlackHole):
                An instance of BlackHole containing the black hole
                particle data.
            **kwargs (dict):
                Additional keyword arguments to be passed to the BaseGalaxy
                __init__ method.

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

        # Attach any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __add__(self, second_galaxy):
        """Add a parametric Galaxy.

        NOTE: functionality for adding lines and images not yet implemented.

        Args:
            second_galaxy (Galaxy):
                The second galaxy to be added to this one.

        Returns:
            ParametricGalaxy:
                A new galaxy object containing the combined
                properties of both galaxies.
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

    @deprecated(
        "get_Q is deprecared due to incorrect naming converntion. Please use "
        "get_ionising_photon_luminosity instead."
    )
    def get_Q(self, grid):
        """Calculate the ionising photon luminosity.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.

        Returns:
            Log of the ionising photon luminosity over the grid dimensions
        """
        return self.get_ionising_photon_luminosity(grid)

    def get_ionising_photon_luminosity(self, grid):
        """Calculate the ionising photon luminosity.

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

    def get_data_cube(
        self,
        resolution,
        fov,
        lam,
        stellar_spectra=None,
        blackhole_spectra=None,
        quantity="lnu",
    ):
        """Make a SpectralCube from an Sed.

        Data cubes are calculated by smoothing spectra over the component
        morphology. The Sed used is defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output data cube.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (unyt_quantity of float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (unyt_quantity of float):
                The width of the image in image coordinates.
            lam (unyt_array, float):
                The wavelength array to use for the data cube.
            stellar_spectra (str):
                The stellar spectra key to make into a data cube.
            blackhole_spectra (str):
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
