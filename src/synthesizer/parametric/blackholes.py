"""A module for working with a parametric black holes.

Contains the BlackHole class for use with parametric systems. This houses
all the attributes and functionality related to parametric black holes.

Example usages::

    bhs = BlackHole(
        bolometric_luminosity,
        mass,
        accretion_rate,
        epsilon,
        inclination,
        spin,
        metallicity,
        offset,
    )
"""

import numpy as np
from unyt import Msun, cm, deg, erg, km, kpc, s, yr

from synthesizer import exceptions
from synthesizer.components.blackhole import BlackholesComponent
from synthesizer.emission_models.utils import get_param
from synthesizer.extensions.timers import tic, toc
from synthesizer.parametric.morphology import PointSource
from synthesizer.units import accepts


class BlackHole(BlackholesComponent):
    """The base parametric BlackHole class.

    Attributes:
        morphology (PointSource)
            An instance of the PointSource morphology that describes the
            location of this blackhole
    """

    @accepts(
        mass=Msun.in_base("galactic"),
        accretion_rate=Msun.in_base("galactic") / yr,
        inclination=deg,
        offset=kpc,
        bolometric_luminosity=erg / s,
        hydrogen_density_blr=1 / cm**3,
        hydrogen_density_nlr=1 / cm**3,
        velocity_dispersion_blr=km / s,
        velocity_dispersion_nlr=km / s,
        theta_torus=deg,
    )
    def __init__(
        self,
        mass=None,
        accretion_rate=None,
        accretion_rate_eddington=None,
        epsilon=0.1,
        inclination=None,
        spin=None,
        offset=np.array([0.0, 0.0]) * kpc,
        bolometric_luminosity=None,
        metallicity=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        fesc=None,
        **kwargs,
    ):
        """Intialise the BlackHole instance.

        Args:
            mass (float):
                The mass of each particle in Msun.
            accretion_rate (float):
                The accretion rate of the black hole in Msun/yr. No need to
                provide if accretion_rate_eddington is given.
            accretion_rate_eddington (float):
                The accretion rate of the black hole in terms of the Eddington
                accretion rate. No need to provide if accretion_rate is given.
            epsilon (float):
                The radiative efficiency. By default set to 0.1.
            inclination (float):
                The inclination of the black hole. Necessary for some disc
                models.
            spin (float):
                The spin of the black hole. Necessary for some disc models.
            offset (unyt_array):
                The (x,y) offsets of the black hole relative to the centre of
                the image. Units can be length or angle but should be
                consistent with the scene.
            bolometric_luminosity (float):
                The bolometric luminosity
            metallicity (float):
                The metallicity of the region surrounding the/each black hole.
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
            fesc (np.ndarray of float):
                The escape fraction of the black hole. If None then the
                escape fraction is set to 0.0.
            kwargs (dict):
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """
        # Initialise base class
        BlackholesComponent.__init__(
            self,
            fesc=fesc,
            bolometric_luminosity=bolometric_luminosity,
            mass=mass,
            accretion_rate=accretion_rate,
            accretion_rate_eddington=accretion_rate_eddington,
            epsilon=epsilon,
            inclination=inclination,
            spin=spin,
            metallicity=metallicity,
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

        # By default a parametric black hole will explicitly have 1 "particle",
        # set this here so that the downstream extraction can access the
        # attribute.
        self.nparticles = 1
        self.nbh = 1

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

    def get_mask(
        self,
        attr_str,
        thresh,
        op,
        mask=None,
        attr_override_obj=None,
    ):
        """Return a mask based on the attribute and threshold.

        Will derive a mask of the form attr op thresh, e.g. age > 10 Myr.

        Args:
            attr_str (str):
                The attribute to derive the mask from.
            thresh (float):
                The threshold value.
            op (str):
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (np.ndarray):
                Optionally, a mask to combine with the new mask.
            attr_override_obj (object):
                An alternative object to check from the attribute. This
                is specifically used when an EmissionModel may have a
                fixed parameter override, but can be used more generally.

        Returns:
            mask (np.ndarray):
                The mask array.
        """
        start = tic()

        # Get the attribute
        attr = get_param(attr_str, attr_override_obj, None, self)

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

        toc("Generating parametric mask", start)

        return new_mask

    def get_weighted_attr(self, *args, **kwargs):
        """Raise an error, weighted attributes are meaningless for a BlackHole.

        Raises:
            NotImplementedError
                Parametric black holes are singular and so weighted attributes
                make no sense.
        """
        raise NotImplementedError(
            "Parametric black holes are by definition singular "
            "making weighted attributes non-sensical."
        )

    def calculate_ionising_luminosity(self):
        """Calculates the ionising luminosity of the blackhole(s).

        This requires that the disc_incident spectra be available.

        Returns:
             unyt_array:
                The ionising photon production rate (s^-1).
        """
        if "disc_incident" in self.spectra.keys():
            return self.spectra[
                "disc_incident"
            ].calculate_ionising_photon_production_rate()
        else:
            raise exceptions.MissingSpectraType(
                "It is necessary to first calculate the disc_incident "
                "spectra before calculating the ionising luminosity"
            )
