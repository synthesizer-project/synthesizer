"""A module for working with arrays of black holes.

Contains the BlackHoles class for use with particle based systems. This houses
all the data detailing collections of black hole particles. Each property is
stored in (N_bh, ) shaped arrays for efficiency.

When instantiate a BlackHoles object a myriad of extra optional properties can
be set by providing them as keyword arguments.

Example usages:

    bhs = BlackHoles(masses, metallicities,
                     redshift=redshift, accretion_rate=accretion_rate, ...)
"""

import os

import numpy as np
from unyt import rad, unyt_array

from synthesizer import exceptions
from synthesizer.components import BlackholesComponent
from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer.utils import value_to_array


class BlackHoles(Particles, BlackholesComponent):
    """
    The base BlackHoles class. This contains all data a collection of black
    holes could contain. It inherits from the base Particles class holding
    attributes and methods common to all particle types.

    The BlackHoles class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be initialised with a BlackHoles object for use with any of the Galaxy
    helper methods.

    Note that due to the many possible operations, this class has a large
    number of optional attributes which are set to None if not provided.

    Attributes:
        nbh (int)
            The number of black hole particles in the object.
        smoothing_lengths (array-like, float)
            The smoothing length describing the black holes neighbour kernel.
        particle_spectra (dict)
            A dictionary of Sed objects containing any of the generated
            particle spectra.
    """

    # Define the allowed attributes
    attrs = [
        "_masses",
        "_coordinates",
        "_velocities",
        "metallicities",
        "nparticles",
        "redshift",
        "_accretion_rate",
        "_bb_temperature",
        "_bolometric_luminosity",
        "_softening_lengths",
        "_smoothing_lengths",
        "nbh",
    ]

    # Define quantities
    smoothing_lengths = Quantity()

    def __init__(
        self,
        masses,
        accretion_rates,
        epsilons=0.1,
        inclinations=None,
        spins=None,
        metallicities=None,
        redshift=None,
        coordinates=None,
        velocities=None,
        softening_length=None,
        smoothing_lengths=None,
        centre=None,
        **kwargs,
    ):
        """
        Intialise the Stars instance. The first two arguments are always
        required. All other arguments are optional attributes applicable
        in different situations.

        Args:
            masses (array-like, float)
                The mass of each particle in Msun.
            metallicities (array-like, float)
                The metallicity of the region surrounding the/each black hole.
            epsilons (array-like, float)
                The radiative efficiency. By default set to 0.1.
            inclination (array-like, float)
                The inclination of the blackhole. Necessary for many emission
                models.
            redshift (float)
                The redshift/s of the black hole particles.
            accretion_rate (array-like, float)
                The accretion rate of the/each black hole in Msun/yr.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            softening_length (float)
                The physical gravitational softening length.
            smoothing_lengths (array-like, float)
                The smoothing length describing the black holes neighbour
                kernel.
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """

        # Handle singular values being passed (arrays are just returned)
        masses = value_to_array(masses)
        accretion_rates = value_to_array(accretion_rates)
        epsilons = value_to_array(epsilons)
        inclinations = value_to_array(inclinations)
        spins = value_to_array(spins)
        metallicities = value_to_array(metallicities)
        smoothing_lengths = value_to_array(smoothing_lengths)

        # Instantiate parents
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(masses),
            centre=centre,
        )
        BlackholesComponent.__init__(
            self,
            mass=masses,
            accretion_rate=accretion_rates,
            epsilon=epsilons,
            inclination=inclinations,
            spin=spins,
            metallicity=metallicities,
            **kwargs,
        )

        # Set a frontfacing clone of the number of particles with clearer
        # naming
        self.nbh = self.nparticles

        # Make pointers to the singular black hole attributes for consistency
        # in the backend
        for singular, plural in [
            ("mass", "masses"),
            ("accretion_rate", "accretion_rates"),
            ("metallicity", "metallicities"),
            ("spin", "spins"),
            ("inclination", "inclinations"),
            ("epsilon", "epsilons"),
            ("bb_temperature", "bb_temperatures"),
            ("bolometric_luminosity", "bolometric_luminosities"),
            ("accretion_rate_eddington", "accretion_rates_eddington"),
            ("epsilon", "epsilons"),
            ("eddington_ratio", "eddington_ratios"),
        ]:
            setattr(self, plural, getattr(self, singular))

        # Set the smoothing lengths
        self.smoothing_lengths = smoothing_lengths

        # Check the arguments we've been given
        self._check_bh_args()

        # Define the particle spectra dictionary
        self.particle_spectra = {}

    def _check_bh_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.attrs:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent black hole array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def calculate_random_inclination(self):
        """
        Calculate random inclinations to blackholes.
        """

        self.inclination = (
            np.random.uniform(low=0.0, high=np.pi / 2.0, size=self.nbh) * rad
        )

        self.cosine_inclination = np.cos(self.inclination.to("rad").value)

    def _prepare_sed_args(
        self,
        emission_model,
        grid,
        fesc,
        spectra_type,
        line_region,
        mask,
        grid_assignment_method,
        nthreads,
    ):
        """
        A method to prepare the arguments for SED computation with the C
        functions.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).
            grid (Grid)
                The SPS grid object to extract spectra from.
            fesc (float)
                The escape fraction.
            spectra_type (str)
                The type of spectra to extract from the Grid. This must match a
                type of spectra stored in the Grid.
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use for the computation. If -1 then
                all available threads are used.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Handle the case where mask is None
        if mask is None:
            mask = np.ones(self.nbh, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = []
        for axis in grid.axes:
            # Source parameters from the appropriate place
            if axis in emission_model.required_parameters:
                # Parameters that need to be provided from the black hole
                # (these already exclude any parameters fixed on the
                # emission model)
                props.append(getattr(self, axis))

            elif (
                axis in emission_model.variable_parameters
                and getattr(self, axis, None) is not None
            ):
                # Variable parameters defined on the black hole (not including
                # line region parameters)
                props.append(getattr(self, axis))

            elif (
                f"{axis} {line_region}" in emission_model.variable_parameters
                and getattr(self, axis + "_" + line_region, None) is not None
            ):
                # Variable line region parameters defined on the black hole
                props.append(getattr(self, axis + "_" + line_region))

            elif getattr(emission_model, axis, None) is not None:
                # Parameters required from the emission_model (not including
                # line region parameters)
                props.append(getattr(emission_model, axis))

            elif (
                getattr(emission_model, axis + "_" + line_region, None)
                is not None
            ):
                # Line region parameters required from the emission_model
                props.append(getattr(emission_model, axis + "_" + line_region))

            else:
                raise exceptions.MissingArgument(
                    f"{axis} can't be sourced from emission_model or self."
                )

        # Calculate npart from the mask
        npart = np.sum(mask)

        # Remove units from any unyt_arrays
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]

        # Ensure any parameters inherited from the emission model have
        # as many values as particles
        for ind, prop in enumerate(props):
            if isinstance(prop, float):
                props[ind] = np.full(self.nbh, prop)
            elif prop.size == 1:
                props[ind] = np.full(self.nbh, prop)

        # Apply the mask to each property and make contiguous
        props = [
            np.ascontiguousarray(prop[mask], dtype=np.float64)
            for prop in props
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(npart, dtype=np.float64)

        # Make sure we get the wavelength index of the grid array
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Get the grid spctra
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type],
            dtype=np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        props = tuple(props)

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_spectra,
            grid_props,
            props,
            mass,
            fesc,
            grid_dims,
            len(grid_props),
            np.int32(npart),
            nlam,
            grid_assignment_method,
            nthreads,
        )

    def generate_particle_lnu(
        self,
        emission_model,
        grid,
        spectra_name,
        line_region,
        fesc=0.0,
        mask=None,
        verbose=False,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra.

        Args:
            emission_model (synthesizer.blackhole_emission_models.*)
                An instance of a blackhole emission model.
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            spectra_name (string)
                The name of the target spectra inside the grid file
                (e.g. "incident", "transmitted", "nebular").
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use for the computation. If -1 then
                all available threads are used.
        """
        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        from ..extensions.particle_spectra import compute_particle_seds

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            emission_model,
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            line_region=line_region,
            mask=mask,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        masked_spec = compute_particle_seds(*args)

        # If there's no mask we're done
        if mask is None:
            return masked_spec

        # If we have a mask we need to account for the zeroed spectra
        spec = np.zeros((self.nbh, masked_spec.shape[-1]))
        spec[mask] = masked_spec
        return spec

    def get_particle_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate blackhole spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            covering_fraction (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the covering_fraction defined on the
                      emission model should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<covering_fraction>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool)
                Are we talking?
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        # Get the spectra
        spectra = emission_model._get_spectra(
            components={"blackhole": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the spectra dictionary
        self.particle_spectra.update(spectra)

        return self.particle_spectra[emission_model.label]

    def get_particle_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate stellar lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested list
                or using a comma (e.g. 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            covering_fraction (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the covering_fraction defined on the
                      emission model should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<covering_fraction>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool)
                Are we talking?
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection
                A LineCollection object containing the lines defined by the
                root model.
        """
        # Get the lines
        lines = emission_model._get_lines(
            line_ids=line_ids,
            components={"blackhole": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the lines dictionary
        self.particle_lines.update(lines)

        return self.particle_lines[emission_model.label]
