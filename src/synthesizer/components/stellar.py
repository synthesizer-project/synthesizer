"""A module defining general functionality for particle and parametric stars.

This should never be directly instantiated. Instead it is the parent class for
particle.Stars and parametric.Stars and contains attributes and methods
common between them. StarsComponent is a child class of Component.
"""

import numpy as np
from unyt import Myr, unyt_quantity

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter


class StarsComponent(Component):
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly.

    Attributes:
        ages (Quantity)
            The age of each stellar particle/bin (particle/parametric).
        metallicities (Quantity)
            The metallicity of each stellar particle/bin
            (particle/parametric).
    """

    # Define quantities
    ages = Quantity("time")

    @accepts(ages=Myr)
    def __init__(
        self,
        ages,
        metallicities,
        _star_type,
        **kwargs,
    ):
        """
        Initialise the StellarComponent.

        Args:
            ages (array-like, float)
                The age of each stellar particle/bin (particle/parametric).
            metallicities (array-like, float)
                The metallicity of each stellar particle/bin
                (particle/parametric).
        """
        # Initialise the parent class
        Component.__init__(self, "Stars", **kwargs)

        # The common stellar attributes between particle and parametric stars
        self.ages = ages
        self.metallicities = metallicities

        # The type of stars object (parametric or particle). This is useful for
        # determining the type of stars object without relying on isinstance
        # and possible circular imports.
        self._star_type = _star_type

    @property
    def log10ages(self):
        """
        Return stellar particle ages in log (base 10).

        Returns:
            log10ages (array)
                log10 stellar ages
        """
        return np.log10(self._ages)

    @property
    def log10metallicities(self):
        """
        Return stellar particle metallicities in log (base 10).

        Returns:
            log10metallicities (array)
                log10 stellar metallicities
        """
        return np.log10(self.metallicities)

    def __str__(self):
        """
        Return a string representation of the stars object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Stars")

    def _check_young_old_units(self, young, old):
        """
        Checks whether the `young` and `old` arguments to many
        spectra generation methods are in the right units (Myr)

        Args:
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
        """

        if young is not None:
            if isinstance(young, (unyt_quantity)):
                young = young.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )
        if old is not None:
            if isinstance(old, (unyt_quantity)):
                old = old.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )

        return young, old
