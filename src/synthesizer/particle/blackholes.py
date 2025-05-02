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

import numpy as np
from unyt import (
    Mpc,
    Msun,
    cm,
    deg,
    km,
    rad,
    s,
    yr,
)

from synthesizer import exceptions
from synthesizer.components.blackhole import BlackholesComponent
from synthesizer.particle.particles import Particles
from synthesizer.synth_warnings import deprecated
from synthesizer.units import Quantity, accepts
from synthesizer.utils import scalar_to_array


class BlackHoles(Particles, BlackholesComponent):
    """The particle BlackHoles class.

    This contains all data a collection of black
    holes could contain. It inherits from the base Particles class holding
    attributes and methods common to all particle types.

    The BlackHoles class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be initialised with a BlackHoles object for use with any of the Galaxy
    helper methods.
    Note that due to the many possible operations, this class has a large
    number ofoptional attributes which are set to None if not provided.

    Attributes:
        nbh (int):
            The number of black hole particles in the object.
        smoothing_lengths (np.ndarray of float):
            The smoothing length describing the black holes neighbour kernel.
        particle_spectra (dict):
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
    smoothing_lengths = Quantity("spatial")

    @accepts(
        masses=Msun.in_base("galactic"),
        accretion_rates=Msun.in_base("galactic") / yr,
        inclinations=deg,
        coordinates=Mpc,
        velocities=km / s,
        softening_length=Mpc,
        smoothing_lengths=Mpc,
        centre=Mpc,
        hydrogen_density_blr=1 / cm**3,
        hydrogen_density_nlr=1 / cm**3,
        velocity_dispersion_blr=km / s,
        velocity_dispersion_nlr=km / s,
        theta_torus=deg,
    )
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
        softening_lengths=None,
        smoothing_lengths=None,
        centre=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        tau_v=None,
        fesc=None,
        **kwargs,
    ):
        """Intialise the BlackHoles instance.

        Args:
            masses (np.ndarray of float):
                The mass of each particle in Msun.
            accretion_rates (np.ndarray of float):
                The accretion rate of the/each black hole in Msun/yr.
            epsilons (np.ndarray of float):
                The radiative efficiency. By default set to 0.1.
            inclinations (np.ndarray of float):
                The inclination of the blackhole. Necessary for many emission
                models.
            spins (np.ndarray of float):
                The spin of the black hole. Necessary for many emission
                models.
            metallicities (np.ndarray of float):
                The metallicity of the region surrounding the/each black hole.
            redshift (float):
                The redshift/s of the black hole particles.
            coordinates (np.ndarray of float):
                The 3D positions of the particles.
            velocities (np.ndarray of float):
                The 3D velocities of the particles.
            softening_lengths (float):
                The physical gravitational softening length.
            smoothing_lengths (np.ndarray of float):
                The smoothing length describing the black holes neighbour
                kernel.
            centre (np.ndarray of float):
                The centre of the black hole particles. This will be used for
                centered calculations (e.g. imaging or angular momentum).
            ionisation_parameter_blr (np.ndarray of float):
                The ionisation parameter of the broad line region.
            hydrogen_density_blr (np.ndarray of float):
                The hydrogen density of the broad line region.
            covering_fraction_blr (np.ndarray of float):
                The covering fraction of the broad line region (effectively
                the escape fraction).
            velocity_dispersion_blr (np.ndarray of float):
                The velocity dispersion of the broad line region.
            ionisation_parameter_nlr (np.ndarray of float):
                The ionisation parameter of the narrow line region.
            hydrogen_density_nlr (np.ndarray of float):
                The hydrogen density of the narrow line region.
            covering_fraction_nlr (np.ndarray of float):
                The covering fraction of the narrow line region (effectively
                the escape fraction).
            velocity_dispersion_nlr (np.ndarray of float):
                The velocity dispersion of the narrow line region.
            theta_torus (np.ndarray of float):
                The angle of the torus.
            tau_v (np.ndarray of float):
                The optical depth of the dust model.
            fesc (np.ndarray of float):
                The escape fraction of the black hole emission.
            **kwargs (dict):
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """
        # Handle singular values being passed (arrays are just returned)
        masses = scalar_to_array(masses)
        accretion_rates = scalar_to_array(accretion_rates)
        epsilons = scalar_to_array(epsilons)
        inclinations = scalar_to_array(inclinations)
        spins = scalar_to_array(spins)
        metallicities = scalar_to_array(metallicities)
        smoothing_lengths = scalar_to_array(smoothing_lengths)

        # Instantiate parents
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_lengths=softening_lengths,
            nparticles=masses.size,
            centre=centre,
            tau_v=tau_v,
            name="Black Holes",
        )
        BlackholesComponent.__init__(
            self,
            fesc=fesc,
            mass=masses,
            accretion_rate=accretion_rates,
            epsilon=epsilons,
            inclination=inclinations,
            spin=spins,
            metallicity=metallicities,
            ionisation_parameter_blr=ionisation_parameter_blr,
            hydrogen_density_blr=hydrogen_density_blr,
            covering_fraction_blr=covering_fraction_blr,
            velocity_dispersion_blr=velocity_dispersion_blr,
            ionisation_parameter_nlr=ionisation_parameter_nlr,
            hydrogen_density_nlr=hydrogen_density_nlr,
            covering_fraction_nlr=covering_fraction_nlr,
            velocity_dispersion_nlr=velocity_dispersion_nlr,
            theta_torus=theta_torus,
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

    def _check_bh_args(self):
        """Sanitize the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """
        # Need an early exit if we have no black holes since any
        # multidimensional  attributes will trigger the error below erroneously
        if self.nbh == 0:
            return

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
        """Calculate random inclinations to blackholes."""
        self.inclination = (
            np.random.uniform(low=0.0, high=np.pi / 2.0, size=self.nbh) * rad
        )

        self.cosine_inclination = np.cos(self.inclination.to("rad").value)

    @deprecated(
        message="is now just a wrapper "
        "around get_spectra. It will be removed by v1.0.0."
    )
    def get_particle_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        covering_fraction=None,
        mask=None,
        vel_shift=None,
        verbose=True,
        **kwargs,
    ):
        """Generate blackhole spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            covering_fraction (dict):
                An override to the emission model covering fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific covering fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            vel_shift (bool):
                Whether to apply a velocity shift to the spectra.
            verbose (bool):
                Are we talking?
            **kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        previous_per_part = emission_model.per_particle
        emission_model.set_per_particle(True)
        spectra = self.get_spectra(
            emission_model=emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            vel_shift=vel_shift,
            verbose=verbose,
            **kwargs,
        )
        emission_model.set_per_particle(previous_per_part)

        return spectra

    @deprecated(
        message="is now just a wrapper "
        "around get_lines. It will be removed by v1.0.0."
    )
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
        """Generate stellar lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested list
                or using a comma (e.g. 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            covering_fraction (dict):
                An override to the emission model covering fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific covering fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            **kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection
                A LineCollection object containing the lines defined by the
                root model.
        """
        previous_per_part = emission_model.per_particle
        emission_model.set_per_particle(True)
        lines = self.get_lines(
            line_ids=line_ids,
            emission_model=emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )
        emission_model.set_per_particle(previous_per_part)
        return lines
