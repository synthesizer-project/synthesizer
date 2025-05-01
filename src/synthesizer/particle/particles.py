"""The base module from which all other particle types inherit.

This generic particle class forms the base class containing all attributes and
methods common to all child particle types. It should rarely if ever be
directly instantiated.
"""

import copy

import numpy as np
from numpy.random import multivariate_normal
from unyt import Mpc, Msun, km, pc, rad, s

from synthesizer import exceptions
from synthesizer.particle.utils import rotate
from synthesizer.synth_warnings import deprecation
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter, ensure_array_c_compatible_double
from synthesizer.utils.geometry import get_rotation_matrix


class Particles:
    """The base particle class.

    All attributes of this class are optional keyword arguments in the child
    classes. Here we demand they are passed as positional arguments to protect
    against future changes.

    Attributes:
        coordinates (np.ndarray of float):
            The 3D coordinates of each particle.
        velocities (np.ndarray of float):
            The 3D velocity of each stellar particle.
        masses (np.ndarray of float):
            The mass of each particle in Msun.
        redshift (np.ndarray of float):
            The redshift/s of the stellar particles.
        softening_length (float):
            The physical gravitational softening length.
        nparticle : int
            How many particles are there?
        centre (array, float):
            Centre of the particle distribution.
        metallicity_floor (float):
            The metallicity floor when using log properties (only matters for
            baryons). This is used to avoid log(0) errors.
        tau_v (float):
            The V band optical depth.
        radii (np.ndarray of float):
            The radii of the particles.
        _grid_weights (dict, np.ndarray of float):
            Weights for each particle sorted onto a grid. This dictionary takes
            the form of {<method>: {grid_name: weights}} where weights is an
            array of the same shape as the grid containg the particles sorted
            onto the grid.
    """

    # Define class level Quantity attributes
    coordinates = Quantity("spatial")
    velocities = Quantity("velocity")
    masses = Quantity("mass")
    softening_lengths = Quantity("spatial")
    centre = Quantity("spatial")
    radii = Quantity("spatial")

    @accepts(
        coordinates=Mpc,
        velocities=km / s,
        masses=Msun.in_base("galactic"),
        softening_length=Mpc,
        centre=Mpc,
    )
    def __init__(
        self,
        coordinates,
        velocities,
        masses,
        redshift,
        softening_lengths,
        nparticles,
        centre,
        metallicity_floor=1e-5,
        tau_v=None,
        name="Particles",
    ):
        """Initialise the Particles.

        Args:
            coordinates (np.ndarray of float):
                The 3D positions of the particles.
            velocities (np.ndarray of float):
                The 3D velocities of the particles.
            masses (np.ndarray of float):
                The mass of each particle.
            redshift (float):
                The redshift/s of the particles.
            softening_lengths (float):
                The physical gravitational softening length.
            nparticles (int):
                How many particles are there?
            centre (array, float):
                Centre of the particle distribution.
            metallicity_floor (float):
                The metallicity floor when using log properties (only matters
                for baryons). This is used to avoid log(0) errors.
            tau_v (float):
                The V band optical depth.
            name (str):
                The name of the particle type.
        """
        # Set phase space coordinates
        self.coordinates = ensure_array_c_compatible_double(coordinates)
        self.velocities = ensure_array_c_compatible_double(velocities)

        # Define the dictionary to hold particle spectra
        self.particle_spectra = {}

        # Define the dictionary to hold particle lines
        self.particle_lines = {}

        # Initialise the particle photometry dictionaries
        self.particle_photo_lnu = {}
        self.particle_photo_fnu = {}

        # Set unit information

        # Set the softening length
        self.softening_lengths = ensure_array_c_compatible_double(
            softening_lengths
        )

        # Set the particle masses
        self.masses = ensure_array_c_compatible_double(masses)

        # Set the redshift of the particles
        self.redshift = redshift

        # How many particles are there?
        self.nparticles = nparticles

        # Set the centre of the particle distribution
        self.centre = ensure_array_c_compatible_double(centre)

        # Set the radius to None, this will be populated when needed and
        # can then be subsequently accessed
        self.radii = None

        # Set the metallicity floor when using log properties (only matters for
        # baryons)
        self.metallicity_floor = metallicity_floor

        # Set the V band optical depths
        self.tau_v = tau_v

        # Attach the name of the particle type
        self.name = name

    @property
    def particle_photo_fluxes(self):
        """Get the particle photometry fluxes.

        Returns:
            dict
                The photometry fluxes.
        """
        deprecation(
            "The `particle_photo_fluxes` attribute is deprecated. Use "
            "`particle_photo_fnu` instead. Will be removed in v1.0.0"
        )
        return self.particle_photo_fnu

    @property
    def particle_photo_luminosities(self):
        """Get the photometry luminosities.

        Returns:
            dict
                The photometry luminosities.
        """
        deprecation(
            "The `particle_photo_luminosities` attribute is deprecated. Use "
            "`particle_photo_lnu` instead. Will be removed in v1.0.0"
        )
        return self.particle_photo_lnu

    @property
    def centered_coordinates(self):
        """Returns the coordinates centred on the geometric mean."""
        if self.centre is None:
            raise exceptions.InconsistentArguments(
                "Can't centre coordinates without a centre."
            )
        return self.coordinates - self.centre

    @property
    def log10metallicities(self):
        """Return particle metallicities in log (base 10).

        Zero valued metallicities are set to `metallicity_floor`,
        which is set on initialisation of this particle object.

        Returns:
            log10metallicities (np.ndarray):
                log10 particle metallicities.
        """
        mets = self.metallicities
        mets[mets == 0.0] = self.metallicity_floor

        return np.log10(mets, dtype=np.float64)

    def get_projected_angular_coordinates(
        self,
        cosmo=None,
        los_dists=None,
    ):
        """Get the projected angular coordinates in radians.

        This will return the angular coordinates of the particles in radians
        projected along the line of sight axis (always the z-axis). The
        coordinates along the line of sight axis will be set to 0.0, to
        maintain the shape of coordinates array.

        The coordinates will be centred on the centre of the particle
        distribution before calculating the angular coordinates. If the centre
        is not set then an error will be raised.

        Note that a redshift is required if the los_dists aren't given to
        convert the coordinates to angular coordinates. If this redshift
        is 0.0 then the particles will be treated as if they are at 10 pc
        (minimum distance) from the observer.

        Args:
            cosmo (astropy.cosmology):
                The cosmology object from which to derive the luminosity
                distance.
            los_dists (unyt_quantity):
                The line of sight distances to the particles. If None, this
                will be calculated using the redshift and cosmology object.

        Returns:
            unyt_array: The projected angular coordinates of the particles
                in radians.
        """
        # Either cosmo or los_dists must be provided
        if cosmo is None and los_dists is None:
            raise exceptions.InconsistentArguments(
                "Either cosmo or los_dists must be provided to get "
                "projected angular coordinates."
            )

        # Get the centered coordinates
        cent_coords = self.centered_coordinates

        # If we don't have the LOS distances then we need to calculate them
        if los_dists is None:
            # Get the luminosity distance
            lum_dist = self.get_luminosity_distance(cosmo)

            # Combine the luminosity distance with the line of sight distance
            # (along the z-axis)
            los_dists = lum_dist + cent_coords[:, 2]

            # If we are at redshift 0.0 then we need to shift things to
            # put the closest particle at 10 pc
            if self.redshift == 0.0:
                z_min = np.min(cent_coords[:, 2])
                los_dists += np.abs(z_min) + 10 * pc
        else:
            # Ok, we have been handed LOS distances, make sure they are the
            # right shape
            if los_dists.size != self.nparticles:
                raise exceptions.InconsistentArguments(
                    "The LOS distances must be the same shape as the "
                    f"coordinates. Got {los_dists.size} but expected "
                    f"{self.nparticles}."
                )

        # Ensure the distances are in the right units
        x = cent_coords[:, 0].value
        y = cent_coords[:, 1].value
        d = los_dists.to_value(cent_coords.units)

        # Get the angular coordinates and store them in a (N, 3) array
        coords = np.zeros((self.nparticles, 3), dtype=np.float64)
        coords[:, 0] = np.arctan2(x, d)
        coords[:, 1] = np.arctan2(y, d)

        # Ensure the array is C-contiguous
        coords = ensure_array_c_compatible_double(coords)

        return coords * rad

    def get_projected_angular_smoothing_lengths(
        self,
        cosmo=None,
        los_dists=None,
    ):
        """Get the projected angular smoothing lengths in radians.

        This will return the angular smoothing lengths of the particles in
        radians projected along the line of sight axis (always the z-axis). The
        coordinates along the line of sight axis will be set to 0.0, to
        maintain the shape of coordinates array.

        Note that a redshift is required if the los_dists aren't given to
        convert the coordinates to angular coordinates. If this redshift
        is 0.0 then the particles will be treated as if they are at 10 pc
        (minimum distance) from the observer.

        Args:
            cosmo (astropy.cosmology):
                The cosmology object from which to derive the luminosity
                distance.
            los_dists (unyt_quantity):
                The line of sight distances to the particles. If None, this
                will be calculated using the redshift and cosmology object.

        Returns:
            unyt_array: The projected angular smoothing lengths of the
                particles in radians.
        """
        # Either cosmo or los_dists must be provided
        if cosmo is None and los_dists is None:
            raise exceptions.InconsistentArguments(
                "Either cosmo or los_dists must be provided to get "
                "projected angular smoothing lengths."
            )

        # Get the centered coordinates
        cent_coords = self.centered_coordinates

        # If we don't have the LOS distances then we need to calculate them
        if los_dists is None:
            # Get the luminosity distance
            lum_dist = self.get_luminosity_distance(cosmo)

            # Combine the luminosity distance with the line of sight distance
            # (along the z-axis)
            los_dists = lum_dist + cent_coords[:, 2]

            # If we are at redshift 0.0 then we need to shift things to
            # put the closest particle at 10 pc
            if self.redshift == 0.0:
                z_min = np.min(cent_coords[:, 2])
                los_dists += np.abs(z_min) + 10 * pc
        else:
            # Ok, we have been handed LOS distances, make sure they are the
            # right shape
            if los_dists.size != self.nparticles:
                raise exceptions.InconsistentArguments(
                    "The LOS distances must be the same shape as the "
                    f"coordinates. Got {los_dists.size} but expected "
                    f"{self.nparticles}."
                )

        # Ensure the distances are in the right units
        d = los_dists.to_value(self.smoothing_lengths.units)

        # Calculate and return the projected angular smoothing lengths
        projected_smoothing_lengths = np.arctan2(self._smoothing_lengths, d)

        # Ensure the array is C-contiguous
        projected_smoothing_lengths = ensure_array_c_compatible_double(
            projected_smoothing_lengths
        )

        return projected_smoothing_lengths * rad

    def get_projected_angular_imaging_props(self, cosmo):
        """Get the projected angular imaging properties.

        This is a convenience method to reduce repeated calculations when
        getting angular coordinates and smoothing lengths since they both
        require similar calculations. This method will return the
        projected angular coordinates and projected angular smoothing
        lengths of the particles in radians projected along the line of
        sight axis (always the z-axis). The coordinates along the line of
        sight axis will be set to 0.0, to maintain the shape of coordinates
        array.

        Arguments:
            cosmo (astropy.cosmology):
                The cosmology object from which to derive the luminosity
                distance.

        Returns:
            tuple: A tuple containing the projected angular coordinates and
                projected angular smoothing lengths of the particles in
                radians.
        """
        # Get the centered coordinates
        cent_coords = self.centered_coordinates

        # Get the luminosity distance
        lum_dist = self.get_luminosity_distance(cosmo)

        # Combine the luminosity distance with the line of sight distance
        # (along the z-axis)
        los_dists = lum_dist + cent_coords[:, 2]

        # If we are at redshift 0.0 then we need to shift things to
        # put the closest particle at 10 pc
        if self.redshift == 0.0:
            z_min = np.min(cent_coords[:, 2])
            los_dists += np.abs(z_min) + 10 * pc

        # Compute the projected angular properties
        projected_angular_coords = self.get_projected_angular_coordinates(
            los_dists=los_dists,
        )
        projected_angular_smls = self.get_projected_angular_smoothing_lengths(
            los_dists=los_dists,
        )

        return (
            projected_angular_coords,
            projected_angular_smls,
        )

    def get_particle_photo_lnu(self, filters, verbose=True, nthreads=1):
        """Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            photo_lnu (dict): A dictionary of rest frame broadband
                luminosities.
        """
        # Loop over spectra in the component
        for spectra in self.particle_spectra:
            # Create the photometry collection and store it in the object
            self.particle_photo_lnu[spectra] = self.particle_spectra[
                spectra
            ].get_photo_lnu(filters, verbose, nthreads=nthreads)

        return self.particle_photo_lnu

    def get_particle_photo_fnu(self, filters, verbose=True, nthreads=1):
        """Calculate flux photometry using a FilterCollection object.

        Args:
            filters (object):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            dict: A dictionary of fluxes in each filter in filters.
        """
        # Loop over spectra in the component
        for spectra in self.particle_spectra:
            # Create the photometry collection and store it in the object
            self.particle_photo_fnu[spectra] = self.particle_spectra[
                spectra
            ].get_photo_fnu(filters, verbose, nthreads=nthreads)

        return self.particle_photo_fnu

    def get_mask(self, attr, thresh, op, mask=None):
        """Create a mask using a threshold and attribute on which to mask.

        Args:
            attr (str):
                The attribute to derive the mask from.
            thresh (float):
                The threshold value.
            op (str):
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (array):
                Optionally, a mask to combine with the new mask.

        Returns:
            mask (np.ndarray):
                The mask array.
        """
        # Get the attribute
        attr_str = attr
        attr = getattr(self, attr_str)

        # Ensure the attribute is not None
        if attr is None:
            raise exceptions.MissingMaskAttribute(
                f"Masking attribute ({attr_str}) not found on particle object."
            )

        # If only one value has units throw an error
        if hasattr(attr, "units") and not hasattr(thresh, "units"):
            raise exceptions.InconsistentArguments(
                f"Masking attribute ({attr_str}) has units "
                f"but threshold does not ({thresh})."
            )
        elif not hasattr(attr, "units") and hasattr(thresh, "units"):
            raise exceptions.InconsistentArguments(
                f"Masking attribute ({attr_str}) does not have units "
                f"but threshold has ({thresh})."
            )

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
        """Integrate any particle spectra to get integrated spectra.

        This will take all spectra in self.particle_spectra and call the sum
        method on them, populating self.spectra with the results.
        """
        # Loop over the particle spectra
        for key, sed in self.particle_spectra.items():
            # Sum the spectra
            self.spectra[key] = sed.sum()

    def __str__(self):
        """Return a string representation of the particle object.

        Returns:
            table (str):
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Particles")

    def calculate_centre_of_mass(self):
        """Calculate the centre of mass of the collection of particles.

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
        """Calculate the radii of the particles.

        Returns:
            radii (np.ndarray of float):
                The radii of the particles.
        """
        # Raise an error if the centre is not set
        if self.centre is None:
            raise exceptions.InconsistentArguments(
                "Can't calculate radii without a centre."
            )

        # Calculate the radii
        self.radii = np.linalg.norm(self.centered_coordinates, axis=1)

        return self.radii

    @accepts(aperture_radius=Mpc)
    def _aperture_mask(self, aperture_radius):
        """Mask for particles within spherical aperture.

        Args:
            aperture_radius (float):
                Radius of spherical aperture in kpc
        """
        if self.centre is None:
            raise ValueError(
                "Centre of particles must be set to use aperture mask."
            )

        # Get the radii if not already set
        if self.radii is None:
            self.get_radii()

        return self.radii < aperture_radius

    def _get_radius(self, weights, frac):
        """Calculate the radius of a particle distribution.

        Args:
            weights (np.ndarray of float):
                The weights to use to weight the particles.
            frac (float):
                The fraction of the total weight for the radius.

        Returns:
            radius (float):
                The radius of the particle distribution.
        """
        # Handle special cases
        if frac == 0:
            return 0 * self.radii.units
        elif frac == 1:
            return np.max(self.radii.value) * self.radii.units
        elif self.nparticles == 0:
            return 0 * self.radii.units
        elif self.nparticles == 1:
            return self.radii[0].value * frac * self.radii.units
        elif np.sum(weights) == 0:
            return 0 * self.radii.units

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
        """Calculate the radius of a particle distribution.

        By default this will return the "half attr radius."

        Args:
            weight_attr (str):
                The attribute to use to weight the particles.
            frac (float):
                The fraction of the total attribute for the radius.

        Returns:
            radius (float):
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
        """Calculate the radius of a particle distribution based on luminoisty.

        Args:
            spectra_type (str):
                The type of spectra to use to compute radius.
            filter_code (str):
                The filter code to compute the radius for.
            frac (float):
                The fraction of the total light for the radius.

        Returns:
            radius (float):
                The radius of the particle distribution.
        """
        # Check we have that spectra type, if so unpack it
        if spectra_type not in self.particle_photo_lnu:
            raise exceptions.InconsistentArguments(
                f"{spectra_type} not found in particle photometry. "
                "Call get_particle_photo_lnu first."
            )
        else:
            phot_collection = self.particle_photo_lnu[spectra_type]

        # Check we have the filter code
        if filter_code not in phot_collection.filter_codes:
            raise exceptions.InconsistentArguments(
                f"{filter_code} not found in particle photometry. "
                "Call get_particle_photo_lnu first."
            )
        else:
            light = phot_collection[filter_code]

        return self._get_radius(light, frac)

    def get_flux_radius(self, spectra_type, filter_code, frac=0.5):
        """Calculate the radius of a particle distribution based on flux.

        Args:
            spectra_type (str):
                The type of spectra to use to compute radius.
            filter_code (str):
                The filter code to compute the radius for.
            frac (float):
                The fraction of the total light for the radius.

        Returns:
            radius (float):
                The radius of the particle distribution.
        """
        # Check we have that spectra type, if so unpack it
        if spectra_type not in self.particle_photo_fnu:
            raise exceptions.InconsistentArguments(
                f"{spectra_type} not found in particle photometry. "
                "Call get_particle_photo_fnu first."
            )
        else:
            phot_collection = self.particle_photo_fnu[spectra_type]

        # Check we have the filter code
        if filter_code not in phot_collection.filter_codes:
            raise exceptions.InconsistentArguments(
                f"{filter_code} not found in particle photometry. "
                "Call get_particle_photo_fnu first."
            )
        else:
            light = phot_collection[filter_code]

        return self._get_radius(light, frac)

    def get_half_mass_radius(self):
        """Calculate the half mass radius of the particle distribution.

        Returns:
            radius (float):
                The half mass radius of the particle distribution.
        """
        # Hanlde
        return self.get_attr_radius("masses", 0.5)

    def get_half_luminosity_radius(self, spectra_type, filter_code):
        """Calculate the half luminosity radius of the particle distribution.

        Args:
            spectra_type (str):
                The type of spectra to use to compute radius.
            filter_code (str):
                The filter code to compute the radius for.

        Returns:
            radius (float):
                The half luminosity radius of the particle distribution.
        """
        return self.get_luminosity_radius(spectra_type, filter_code, 0.5)

    def get_half_flux_radius(self, spectra_type, filter_code):
        """Calculate the half flux radius of the particle distribution.

        Args:
            spectra_type (str):
                The type of spectra to use to compute radius.
            filter_code (str):
                The filter code to compute the radius for.

        Returns:
            radius (float):
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
        """Prepare the arguments for line of sight column density computation.

        Args:
            other_parts (Particles):
                The other particles to compute the column density with.
            attr (str):
                The attribute to compute the column density of.
            kernel (array_like, float):
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            mask (bool):
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float):
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool):
                Whether to force the use of a simple loop rather than the tree.
            min_count (int):
                The minimum number of particles allowed in a leaf cell when
                using the tree. This can be used for tuning the tree
                performance.
            nthreads (int):
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
        column_density_attr=None,
        mask=None,
        threshold=1,
        force_loop=0,
        min_count=100,
        nthreads=1,
    ):
        """Calculate the column density of an attribute.

        This will calculate the column density of an attribute on another
        Particles child instance at the positions of the particles in this
        Particles instance.

        Args:
            other_parts (Particles):
                The other particles to calculate the column density with.
            density_attr (str):
                The attribute to use to calculate the column density.
            kernel (np.ndarray of float):
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            column_density_attr (str):
                The attribute to store the column density in on the Particles
                instance. If None, the column density will not be stored. By
                default this is None.
            mask (array-like, bool):
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float):
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool):
                Whether to force the use of a simple loop rather than the tree.
            min_count (int):
                The minimum number of particles allowed in a leaf cell when
                using the tree. This can be used for tuning the tree
                performance.
            nthreads (int):
                The number of threads to use for the calculation.

        Returns:
            column_density (float):
                The column density of the particles.
        """
        from synthesizer.extensions.column_density import (
            compute_column_density,
        )

        # If have no particles return 0
        if self.nparticles == 0:
            return np.zeros(self.nparticles)

        # If the other particles have no particles return 0
        if other_parts.nparticles == 0:
            return np.zeros(self.nparticles)

        # If we don't have a mask make a fake one for consistency
        if mask is None:
            mask = np.ones(self.nparticles, dtype=bool)

        # Compute the column density
        col_den = compute_column_density(
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

        # Set the column density attribute (if requested)
        if column_density_attr is not None:
            setattr(self, column_density_attr, col_den)

        return col_den

    @accepts(phi=rad, theta=rad)
    def rotate_particles(
        self,
        phi=0 * rad,
        theta=0 * rad,
        rot_matrix=None,
        inplace=True,
    ):
        """Rotate coordinates.

        This method can either use angles or a provided rotation matrix.

        When using angles phi is the rotation around the z-axis while theta
        is the rotation around the y-axis.

        This can both be done in place or return a new instance, by default
        this will be done in place.

        Args:
            phi (unyt_quantity):
                The angle in radians to rotate around the z-axis. If rot_matrix
                is defined this will be ignored.
            theta (unyt_quantity):
                The angle in radians to rotate around the y-axis. If rot_matrix
                is defined this will be ignored.
            rot_matrix (np.ndarray of float):
                A 3x3 rotation matrix to apply to the coordinates
                instead of phi and theta.
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles: A new instance of the particles with the rotated
                coordinates, if inplace is False.
        """
        # Are we rotating in place or returning a new instance?
        if inplace:
            # Rotate the coordinates
            self.coordinates = rotate(self.coordinates, phi, theta, rot_matrix)
            self.velocities = rotate(self.velocities, phi, theta, rot_matrix)
            if self.centre is not None:
                self.centre = rotate(self.centre, phi, theta, rot_matrix)

            return

        # Ok, we're returning a new one, make a copy
        new_parts = copy.deepcopy(self)

        # Rotate the coordinates
        new_parts.coordinates = rotate(
            new_parts.coordinates, phi, theta, rot_matrix
        )
        new_parts.velocities = rotate(
            new_parts.velocities, phi, theta, rot_matrix
        )
        if self.centre is not None:
            new_parts.centre = rotate(new_parts.centre, phi, theta, rot_matrix)

        # Return the new one
        return new_parts

    @property
    def angular_momentum(self):
        """Calculate the total angular momentum vector of the particles.

        Returns:
            unyt_array: The angular momentum vector of the particle system.
        """
        # Ensure we have all the attributes we need
        if self.coordinates is None:
            raise exceptions.InconsistentArguments(
                "Can't calculate angular momentum without coordinates."
            )
        if self.velocities is None:
            raise exceptions.InconsistentArguments(
                "Can't calculate angular momentum without velocities."
            )
        if self.masses is None:
            raise exceptions.InconsistentArguments(
                "Can't calculate angular momentum without masses."
            )

        # We have to do some unit gymnastics to make sure we return sensible
        # units. Since coordinates and velocities won't necessarily agree
        # on the length unit we adopt the velocity length unit which we can
        # extract with some simple string manipulation.
        distance_unit = str(self.velocities.units).split("/")[0]
        ang_mom_unit = (
            f"{distance_unit} * {self.masses.units} * {self.velocities.units}"
        )

        # Cross product of position and velocity, weighted by mass
        return np.sum(
            np.cross(self.coordinates, self.velocities) * self.masses[:, None],
            axis=0,
        ).to(ang_mom_unit)

    def rotate_edge_on(self, inplace=True):
        """Rotate the particle distribution to edge-on.

        This will rotate the particle distribution such that the angular
        momentum vector is aligned with the y-axis in an image.

        Args:
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles
                A new instance of the particles with rotated coordinates,
                if inplace is False.
        """
        # Get the rotation matrix to rotate ang_mom_hat to the y-axis
        rot_matrix = get_rotation_matrix(
            self.angular_momentum, np.array([1, 0, 0])
        )

        # Call the rotate_particles method with the computed angles
        return self.rotate_particles(rot_matrix=rot_matrix, inplace=inplace)

    def rotate_face_on(self, inplace=True):
        """Rotate the particle distribution to face-on.

        This will rotate the particle distribution such that the angular
        momentum vector is aligned with the z-axis in an image.

        Args:
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles
                A new instance of the particles with rotated coordinates,
                if inplace is False.
        """
        # Get the rotation matrix to rotate ang_mom_hat to the z-axis
        rot_matrix = get_rotation_matrix(
            self.angular_momentum, np.array([0, 0, -1])
        )

        # Call the rotate_particles method with the computed angles
        return self.rotate_particles(rot_matrix=rot_matrix, inplace=inplace)

    def get_weighted_attr(self, attr, weights, axis=None):
        """Get a weighted attribute.

        This will compute the weighted average of an attribute using the
        provided weights.

        Args:
            attr (str):
                The attribute to weight.
            weights (str/np.ndarray of float):
                The weights to apply to the attribute. This can either be a
                string to get the attribute from the particle object or an
                array-like of the weights.
            axis (int):
                The axis to compute the weighted attribute along.

        Returns:
            weighted_attr (float): The weighted attribute.
        """
        # Get the attribute
        attr_vals = getattr(self, attr)

        # If the weights are a string get the attribute
        if isinstance(weights, str):
            weights_vals = getattr(self, weights)
        else:
            weights_vals = weights

        # Strip units off the weights if they have them, this can confuse
        # things
        if hasattr(weights_vals, "units"):
            weights_vals = weights_vals.value

        return np.average(attr_vals, weights=weights_vals, axis=axis)

    def get_lum_weighted_attr(
        self, attr, spectra_type, filter_code, axis=None
    ):
        """Get a luminosity weighted attribute.

        This will compute the luminosity weighted average of an attribute
        using the provided weights.

        Args:
            attr (str):
                The attribute to weight.
            spectra_type (str):
                The type of spectra to use to compute the luminosity.
            filter_code (str):
                The filter code to compute the luminosity for.
            axis (int):
                The axis to compute the weighted attribute along.

        Returns:
            weighted_attr (float):
                The luminosity weighted attribute.
        """
        # Get the luminosity
        lum = self.particle_photo_lnu[spectra_type][filter_code]

        return self.get_weighted_attr(attr, lum, axis)

    def get_flux_weighted_attr(
        self, attr, spectra_type, filter_code, axis=None
    ):
        """Get a flux weighted attribute.

        This will compute the flux weighted average of an attribute using the
        provided weights.

        Args:
            attr (str):
                The attribute to weight.
            spectra_type (str):
                The type of spectra to use to compute the flux.
            filter_code (str):
                The filter code to compute the flux for.
            axis (int):
                The axis to compute the weighted attribute along.

        Returns:
            weighted_attr (float): The flux weighted attribute.
        """
        # Get the flux
        flux = self.particle_photo_fnu[spectra_type][filter_code]

        return self.get_weighted_attr(attr, flux, axis)


class CoordinateGenerator:
    """A collection of functions for generating random coordinate arrays."""

    def generate_3D_gaussian(n, mean=np.zeros(3), cov=None):
        """Generate for coordinates from a 3D gaussian distribution.

        Args:
            n (int):
                The number of coordinates to sample.
            mean (np.ndarray of float):
                The centre of the gaussian distribution. Must be a 3D array
                containing the centre along each axis.
            cov (np.ndarray of float):
                The covariance of the gaussian distribution.

        Returns:
            coords (np.ndarray of float):
                The sampled coordinates in an (n, 3) array.
        """
        # If we haven't been passed a covariance make one
        if cov is None:
            cov = np.zeros((3, 3))
            np.fill_diagonal(cov, 1.0)

        # Get the coordinates
        coords = multivariate_normal(mean, cov, n)

        return coords

    def generate_2D_Sersic(N):
        """Generate a 2D Sersic profile."""
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def generate_3D_spline(N, kernel_func):
        """Generate a 3D spline profile."""
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )
