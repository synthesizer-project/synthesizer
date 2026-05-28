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

from unyt import Mpc

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
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

    def get_data_cube(
        self,
        fov,
        instrument,
        label=None,
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
            fov (unyt_quantity of float):
                The width of the image in image coordinates.
            instrument (IntegratedFieldUnit):
                The instrument to use for the data cube.
            label (str):
                A saved spectrum label to resolve across attached components.
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
        # Normalise the legacy explicit-component arguments and the newer
        # label-based API into one list of requested cube labels.
        labels = []
        if label is not None:
            if stellar_spectra is not None or blackhole_spectra is not None:
                raise exceptions.InconsistentArguments(
                    "Pass either label or explicit component labels to "
                    "get_data_cube, not both."
                )
            labels = [label]
        else:
            if stellar_spectra is not None:
                labels.append(stellar_spectra)
            if (
                blackhole_spectra is not None
                and blackhole_spectra not in labels
            ):
                labels.append(blackhole_spectra)

        if len(labels) == 0:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(label, stellar_spectra or blackhole_spectra)!"
                " What component/s do you want a data cube of?"
            )

        if label is None:
            # Legacy explicit-component requests stay component-owned: generate
            # each requested cube on the relevant component and add them only
            # when the caller asked for more than one component explicitly.
            cubes = []

            if stellar_spectra is not None and self.stars is not None:
                cubes.append(
                    self.stars.get_data_cube(
                        stellar_spectra,
                        fov=fov,
                        instrument=instrument,
                        quantity=quantity,
                    )
                )

            if blackhole_spectra is not None and self.black_holes is not None:
                cubes.append(
                    self.black_holes.get_data_cube(
                        blackhole_spectra,
                        fov=fov,
                        instrument=instrument,
                        quantity=quantity,
                    )
                )

            if len(cubes) == 1:
                return cubes[0]
            if len(cubes) == 0:
                raise exceptions.MissingAttribute(
                    "No requested component is present on this galaxy."
                )
            return cubes[0] + cubes[1]

        # Label-based requests use the galaxy-level orchestration path, which
        # mirrors imaging: route labels to components, build any galaxy-level
        # combinations, then apply IFU post-processing once at the owning
        # level.
        return BaseGalaxy._generate_data_cubes(
            self,
            *labels,
            fov=fov,
            instrument=instrument,
            quantity=quantity,
        )
