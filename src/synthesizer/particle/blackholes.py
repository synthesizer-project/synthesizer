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
        accretion_rates=None,
        accretion_rates_eddington=None,
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
                The accretion rate of the/each black hole in Msun/yr. No need
                to provide both this and accretion_rates_eddington.
            accretion_rates_eddington (np.ndarray of float):
                The accretion rate in terms of the Eddington accretion rate.
                No need to provide both this and accretion_rates.
            epsilons (np.ndarray of float):
                The radiative efficiency. By default set to 0.1.
            inclinations (np.ndarray of float):
                The inclination of the black hole. Necessary for many emission
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
            accretion_rate_eddington=accretion_rates_eddington,
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

        # Set a front facing clone of the number of particles with clearer
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
            ("bolometric_luminosity", "bolometric_luminosities"),
            ("accretion_rate_eddington", "accretion_rates_eddington"),
            ("epsilon", "epsilons"),
        ]:
            setattr(self, plural, getattr(self, singular))

        # Set the smoothing lengths
        self.smoothing_lengths = smoothing_lengths

    def calculate_random_inclination(self):
        """Calculate random inclinations to blackholes."""
        self.inclination = (
            np.random.uniform(low=0.0, high=np.pi / 2.0, size=self.nbh) * rad
        )

        self.cosine_inclination = np.cos(self.inclination.to("rad").value)

    def calculate_ionising_luminosity(self):
        """Calculates the ionising luminosity of the blackhole(s).

        This requires that the disc_incident spectra be available.

        Returns:
             unyt_array:
                The ionising photon production rate (s^-1).
        """
        if "disc_incident" in self.particle_spectra.keys():
            return self.particle_spectra[
                "disc_incident"
            ].calculate_ionising_photon_production_rate()
        else:
            raise exceptions.MissingSpectraType(
                "It is necessary to first calculate the disc_incident "
                "particle_spectra before calculating the ionising luminosity"
            )
