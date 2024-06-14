"""The base module from which all other particle types inherit.

This generic particle class forms the base class containing all attributes and
methods common to all child particle types. It should rarely if ever be
directly instantiated.
"""

import numpy as np
from numpy.random import multivariate_normal
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import Quantity


class Particles:
    """
    The base particle class.

    All attributes of this class are optional keyword arguments in the child
    classes. Here we demand they are passed as positional arguments to protect
    against future changes.

    Attributes:
        coordinates (array-like, float)
            The 3D coordinates of each particle.
        velocities (array-like, float)
            The 3D velocity of each stellar particle.
        masses (array-like, float)
            The mass of each particle in Msun.
        redshift (array-like/float)
            The redshift/s of the stellar particles.
        softening_length (float)
            The physical gravitational softening length.
        nparticle : int
            How many particles are there?
    """

    # Define class level Quantity attributes
    coordinates = Quantity()
    velocities = Quantity()
    masses = Quantity()
    softening_lengths = Quantity()

    def __init__(
        self,
        coordinates,
        velocities,
        masses,
        redshift,
        softening_length,
        nparticles,
        centre,
    ):
        """
        Intialise the Particles.

        Args:
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            masses (array-like, float)
                The mass of each particle.
            redshift (float)
                The redshift/s of the particles.
            softening_length (float)
                The physical gravitational softening length.
            nparticle (int)
                How many particles are there?
            centre (array, float)
                Centre of the particle distribution.
        """
        # Check arguments are valid

        # Set phase space coordinates
        self.coordinates = coordinates
        self.velocities = velocities

        # Initialise the particle photometry dictionaries
        self.particle_photo_luminosities = {}
        self.particle_photo_fluxes = {}

        # Set unit information

        # Set the softening length
        self.softening_lengths = softening_length

        # Set the particle masses
        self.masses = masses

        # Set the redshift of the particles
        self.redshift = redshift

        # How many particles are there?
        self.nparticles = nparticles

        # Set the centre of the particle distribution
        self.centre = centre

    def _check_part_args(
        self, coordinates, velocities, masses, softening_length
    ):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all quantities have units
        if not isinstance(coordinates, unyt_array):
            raise exceptions.InconsistentArguments(
                "coordinates must have unyt units associated to them."
            )
        if not isinstance(velocities, unyt_array):
            raise exceptions.InconsistentArguments(
                "velocities must have unyt units associated to them."
            )
        if not isinstance(masses, unyt_array):
            raise exceptions.InconsistentArguments(
                "masses must have unyt units associated to them."
            )
        if not isinstance(softening_length, unyt_quantity):
            raise exceptions.InconsistentArguments(
                "softening_length must have unyt units associated to them."
            )

    def rotate_particles(self):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def convert_to_physical_properties(
        self,
    ):
        """
        Converts comoving coordinates and velocities to physical coordinates
        and velocties.

        Note that redshift must be provided to perform this conversion.

        Since smoothing lengths are not universal quantities their existence is
        checked before trying to convert them.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def convert_to_comoving_properties(
        self,
    ):
        """
        Converts comoving coordinates and velocities to physical coordinates
        and velocties.

        Note that redshift must be provided to perform this conversion.

        Since smoothing lengths are not universal quantities their existence is
        checked before trying to convert them.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    @property
    def centered_coordinates(self):
        """Returns the coordinates centred on the geometric mean."""
        return self.coordinates - np.average(
            self.coordinates, axis=0, weights=self.masses
        )

    def get_particle_photo_luminosities(self, filters, verbose=True):
        """
        Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            photo_luminosities (dict)
                A dictionary of rest frame broadband luminosities.
        """
        # Loop over spectra in the component
        for spectra in self.particle_spectra:
            # Create the photometry collection and store it in the object
            self.particle_photo_luminosities[spectra] = self.particle_spectra[
                spectra
            ].get_photo_luminosities(filters, verbose)

        return self.particle_photo_luminosities

    def get_particle_photo_fluxes(self, filters, verbose=True):
        """
        Calculate flux photometry using a FilterCollection object.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            (dict)
                A dictionary of fluxes in each filter in filters.
        """
        # Loop over spectra in the component
        for spectra in self.particle_spectra:
            # Create the photometry collection and store it in the object
            self.particle_photo_fluxes[spectra] = self.particle_spectra[
                spectra
            ].get_photo_fluxes(filters, verbose)

        return self.particle_photo_fluxes

    def integrate_particle_spectra(self):
        """
        Integrate any particle spectra to get integrated spectra.

        This will take all spectra in self.particle_spectra and call the sum
        method on them, populating self.spectra with the results.
        """
        # Loop over the particle spectra
        for key, sed in self.particle_spectra.items():
            # Sum the spectra
            self.spectra[key] = sed.sum()


class CoordinateGenerator:
    """
    A collection of helper methods for generating random coordinate arrays from
    various distribution functions.
    """

    def generate_3D_gaussian(n, mean=np.zeros(3), cov=None):
        """
        A generator for coordinates from a 3D gaussian distribution.

        Args:
            n (int)
                The number of coordinates to sample.
            mean (array-like, float)
                The centre of the gaussian distribution. Must be a 3D array
                containing the centre along each axis.
            cov (array-like, float)
                The covariance of the gaussian distribution.

        Returns:
            coords (array-like, float)
                The sampled coordinates in an (n, 3) array.
        """

        # If we haven't been passed a covariance make one
        if not cov:
            cov = np.zeros((3, 3))
            np.fill_diagonal(cov, 1.0)

        # Get the coordinates
        coords = multivariate_normal(mean, cov, n)

        return coords

    def generate_2D_Sersic(N):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def generate_3D_spline(N, kernel_func):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def calculate_centre_of_mass(self):
        """Calculate the centre of mass of the collection
        of particles.

        Uses the `masses` and `coordinates` attributes,
        and assigns the centre of mass to the `centre` attribute
        """
        total_mass = np.sum(self.masses)
        com = np.array([0.0, 0.0, 0.0])

        for i, coods in enumerate(self.coordinates):
            com += coods * self.masses[i]

        com /= total_mass

        self.center = com
