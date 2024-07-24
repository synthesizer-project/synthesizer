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
from synthesizer.utils import TableFormatter


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
        centre (array, float)
            Centre of the particle distribution.
        metallicity_floor (float)
            The metallicity floor when using log properties (only matters for
            baryons). This is used to avoid log(0) errors
        radii (array-like, float)
            The radii of the particles.
    """

    # Define class level Quantity attributes
    coordinates = Quantity()
    velocities = Quantity()
    masses = Quantity()
    softening_lengths = Quantity()
    centre = Quantity()
    radii = Quantity()

    def __init__(
        self,
        coordinates,
        velocities,
        masses,
        redshift,
        softening_length,
        nparticles,
        centre,
        metallicity_floor=1e-5,
        name="Particles",
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
            metallicity_floor (float)
                The metallicity floor when using log properties (only matters
                for baryons). This is used to avoid log(0) errors.
            name (str)
                The name of the particle type.
        """
        # Set phase space coordinates
        self.coordinates = coordinates
        self.velocities = velocities

        # Define the dictionary to hold particle spectra
        self.particle_spectra = {}

        # Define the dictionary to hold particle lines
        self.particle_lines = {}

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

        # Set the radius to None, this will be populated when needed and
        # can then be subsequently accessed
        self.radii = None

        # Set the metallicity floor when using log properties (only matters for
        # baryons)
        self.metallicity_floor = metallicity_floor

        # Attach the name of the particle type
        self.name = name

    def _check_part_args(
        self, coordinates, velocities, masses, softening_length
    ):
        """
        Sanitize the inputs ensuring all arguments agree and are compatible.

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
        """
        Rotate the particle distribution.

        Not yet implemented.
        """
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
        Convert comoving coordinates and velocities to physical.

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
        Convert physical coordinates and velocities to comoving.

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
        if self.centre is None:
            raise exceptions.InconsistentArguments(
                "Can't centre coordinates without a centre."
            )
        return self.coordinates - self.centre

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

    def get_mask(self, attr, thresh, op, mask=None):
        """
        Create a mask using a threshold and attribute on which to mask.

        Args:
            attr (str)
                The attribute to derive the mask from.
            thresh (float)
                The threshold value.
            op (str)
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (array)
                Optionally, a mask to combine with the new mask.

        Returns:
            mask (array)
                The mask array.
        """
        # Get the attribute
        attr = getattr(self, attr)

        # Apply the operator
        if op == ">":
            new_mask = attr > thresh
        elif op == "<":
            new_mask = attr < thresh
        elif op == ">=":
            new_mask = attr >= thresh
        elif op == "<=":
            new_mask = attr <= thresh
        elif op == "==":
            new_mask = attr == thresh
        elif op == "!=":
            new_mask = attr != thresh
        else:
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )

        # Combine with the existing mask
        if mask is not None:
            new_mask = np.logical_and(new_mask, mask)

        return new_mask

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

    def __str__(self):
        """
        Return a string representation of the particle object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Particles")

    def calculate_centre_of_mass(self):
        """
        Calculate the centre of mass of the collection of particles.

        Uses the `masses` and `coordinates` attributes,
        and assigns the centre of mass to the `centre` attribute
        """
        total_mass = np.sum(self.masses)
        com = np.array([0.0, 0.0, 0.0])

        for i, coods in enumerate(self.coordinates):
            com += coods * self.masses[i]

        com /= total_mass

        self.center = com

    def get_radii(self):
        """
        Calculate the radii of the particles.

        Returns:
            radii (array-like, float)
                The radii of the particles.
        """
        # Raise an error if the centre is not set
        if self.centre is None:
            raise exceptions.InconsistentArguments(
                "Can't calculate radii without a centre."
            )

        # Calculate the radii
        self.radii = np.linalg.norm(self.centered_coordinates, axis=1)

    def _get_radius(self, weights, frac):
        """
        Calculate the radius of a particle distribution.

        Args:
            weights (array-like, float)
                The weights to use to weight the particles.
            frac (float)
                The fraction of the total weight for the radius.

        Returns:
            radius (float)
                The radius of the particle distribution.
        """
        # Get the radii if not already set
        if self.radii is None:
            self.get_radii()

        # Strip units off the weights if they have them
        if hasattr(weights, "units"):
            weights = weights.value

        # Sort the weights and radii by radius
        sinds = np.argsort(self.radii)
        weights = weights[sinds]
        radii = self._radii[sinds]

        # Get the total of the weights
        total = np.sum(weights)

        # Get the cumulative array for the weights
        cum_weight = np.cumsum(weights)

        # Interpolate to get an accurate radius
        radius = np.interp(frac * total, cum_weight, radii)

        return radius * self.radii.units

    def get_attr_radius(self, weight_attr, frac=0.5):
        """
        Calculate the radius of a particle distribution.

        By default this will return the "half attr radius."

        Args:
            weight_attr (str)
                The attribute to use to weight the particles.
            frac (float)
                The fraction of the total attribute for the radius.

        Returns:
            radius (float)
                The radius of the particle distribution.
        """
        # Get the weight attribute
        weights = getattr(self, weight_attr, None)

        # Ensure we found the attribute
        if weights is None:
            raise exceptions.InconsistentArguments(
                f"{weight_attr} not found in particle object."
            )

        return self._get_radius(weights, frac)

    def get_luminosity_radius(self, spectra_type, filter_code, frac=0.5):
        """
        Calculate the radius of a particle distribution based on luminoisty.

        Args:
            spectra_type (str)
                The type of spectra to use to compute radius.
            filter_code (str)
                The filter code to compute the radius for.
            frac (float)
                The fraction of the total light for the radius.

        Returns:
            radius (float)
                The radius of the particle distribution.
        """
        # Check we have that spectra type, if so unpack it
        if spectra_type not in self.particle_photo_luminosities:
            raise exceptions.InconsistentArguments(
                f"{spectra_type} not found in particle photometry. "
                "Call get_particle_photo_luminosities first."
            )
        else:
            phot_collection = self.particle_photo_luminosities[spectra_type]

        # Check we have the filter code
        if filter_code not in phot_collection.filter_codes:
            raise exceptions.InconsistentArguments(
                f"{filter_code} not found in particle photometry. "
                "Call get_particle_photo_luminosities first."
            )
        else:
            light = phot_collection[filter_code]

        return self._get_radius(light, frac)

    def get_flux_radius(self, spectra_type, filter_code, frac=0.5):
        """
        Calculate the radius of a particle distribution based on flux.

        Args:
            spectra_type (str)
                The type of spectra to use to compute radius.
            filter_code (str)
                The filter code to compute the radius for.
            frac (float)
                The fraction of the total light for the radius.

        Returns:
            radius (float)
                The radius of the particle distribution.
        """
        # Check we have that spectra type, if so unpack it
        if spectra_type not in self.particle_photo_fluxes:
            raise exceptions.InconsistentArguments(
                f"{spectra_type} not found in particle photometry. "
                "Call get_particle_photo_fluxes first."
            )
        else:
            phot_collection = self.particle_photo_fluxes[spectra_type]

        # Check we have the filter code
        if filter_code not in phot_collection.filter_codes:
            raise exceptions.InconsistentArguments(
                f"{filter_code} not found in particle photometry. "
                "Call get_particle_photo_fluxes first."
            )
        else:
            light = phot_collection[filter_code]

        return self._get_radius(light, frac)

    def get_half_mass_radius(self):
        """
        Calculate the half mass radius of the particle distribution.

        Returns:
            radius (float)
                The half mass radius of the particle distribution.
        """
        # Hanlde
        return self.get_attr_radius("masses", 0.5)

    def get_half_luminosity_radius(self, spectra_type, filter_code):
        """
        Calculate the half luminosity radius of the particle distribution.

        Args:
            spectra_type (str)
                The type of spectra to use to compute radius.
            filter_code (str)
                The filter code to compute the radius for.

        Returns:
            radius (float)
                The half luminosity radius of the particle distribution.
        """
        return self.get_luminosity_radius(spectra_type, filter_code, 0.5)

    def get_half_flux_radius(self, spectra_type, filter_code):
        """
        Calculate the half flux radius of the particle distribution.

        Args:
            spectra_type (str)
                The type of spectra to use to compute radius.
            filter_code (str)
                The filter code to compute the radius for.

        Returns:
            radius (float)
                The half flux radius of the particle distribution.
        """
        return self.get_flux_radius(spectra_type, filter_code, 0.5)

    def _prepare_los_args(
        self,
        other_parts,
        attr,
        kernel,
        mask,
        threshold,
        force_loop,
        min_count,
        nthreads,
    ):
        """
        Prepare the arguments for line of sight surface density computation.

        Args:
            other_parts (Particles)
                The other particles to compute the surface density with.
            attr (str)
                The attribute to compute the surface density of.
            kernel (array_like, float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool)
                Whether to force the use of a simple loop rather than the tree.
            min_count (int)
                The minimum number of particles allowed in a leaf cell when
                using the tree. This can be used for tuning the tree
                performance.
            nthreads (int)
                The number of threads to use for the calculation.
        """
        # Ensure we actually have the properties needed
        if self.coordinates is None:
            raise exceptions.InconsistentArguments(
                f"{self.name} object is missing coordinates!"
            )
        if other_parts.coordinates is None:
            raise exceptions.InconsistentArguments(
                f"{other_parts.name} object is missing coordinates!"
            )
        if other_parts.smoothing_lengths is None:
            raise exceptions.InconsistentArguments(
                f"{other_parts.name} object is missing smoothing lengths!"
            )
        if getattr(other_parts, attr, None) is None:
            raise exceptions.InconsistentArguments(
                f"{other_parts.name} object is missing {attr}!"
            )

        # Set up the kernel inputs to the C function.
        kernel = np.ascontiguousarray(kernel, dtype=np.float64)
        kdim = kernel.size

        # Get particle counts
        npart_i = self.nparticles
        npart_j = other_parts.nparticles

        # Set up the inputs from this particle instance.
        pos_i = np.ascontiguousarray(
            self._coordinates[mask, :], dtype=np.float64
        )

        # Set up the inputs from the other particle instance.
        pos_j = np.ascontiguousarray(
            other_parts._coordinates, dtype=np.float64
        )
        smls = np.ascontiguousarray(
            other_parts._smoothing_lengths, dtype=np.float64
        )
        surf_den_vals = np.ascontiguousarray(
            getattr(other_parts, attr), dtype=np.float64
        )

        return (
            kernel,
            pos_i,
            pos_j,
            smls,
            surf_den_vals,
            npart_i,
            npart_j,
            kdim,
            threshold,
            force_loop,
            min_count,
            nthreads,
        )

    def get_los_column_density(
        self,
        other_parts,
        density_attr,
        kernel,
        mask=None,
        threshold=1,
        force_loop=0,
        min_count=100,
        nthreads=1,
    ):
        """
        Calculate the surface density of an attribute.

        This will calculate the surafce density of an attribute on another
        Particles child instance at the positions of the particles in this
        Particles instance.

        Args:
            other_parts (Particles)
                The other particles to calculate the surface density with.
            density_attr (str)
                The attribute to use to calculate the surface density.
            kernel (array-like, float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            mask (array-like, bool)
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool)
                Whether to force the use of a simple loop rather than the tree.
            min_count (int)
                The minimum number of particles allowed in a leaf cell when
                using the tree. This can be used for tuning the tree
                performance.
            nthreads (int)
                The number of threads to use for the calculation.

        Returns:
            column_density (float)
                The surface density of the particles.
        """
        from synthesizer.extensions.column_density import (
            compute_column_density,
        )

        # If we don't have a mask make a fake one for consistency
        if mask is None:
            mask = np.ones(self.nparticles, dtype=bool)

        # Compute the surface density
        return compute_column_density(
            *self._prepare_los_args(
                other_parts,
                density_attr,
                kernel,
                mask,
                threshold,
                force_loop,
                min_count,
                nthreads,
            )
        )


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
