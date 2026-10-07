"""A submodule for creating and manipulating cosmicstar formation histories.

These differ from other parametric SFHs in that the relative star
formation rate is tied to the age of the universe.

NOTE: This module is imported as CSFH in parametric.__init__ enabling the
      syntax shown below.

Example usage:

    from synthesizer.parametric import CSFH

    print(CSFH.parametrisations)

    csfh = CSFH.MadauDickinson(...)

    csfh.calculate_sfh()

"""

import numpy as np
from unyt import yr

from synthesizer.parametric.sf_hist import Common

# Define a list of the available parametrisations
parametrisations = [
    "MadauDickinson",
]


class MadauDickinson(Common):
    """The Madau & Dickinson (2014) cosmic star formation history.

    The shape of the cosmic star formation history is parametrised as

        psi(z) = (1 + z)^b / (1 + ((1 + z) / c)^d),

    and converted to a function of stellar age using a redshift grid.

    Attributes:
        redshift (float):
            The redshift at which the SFH is observed.
        b (float):
            The low redshift power law slope.
        c (float):
            The turnover in (1 + z).
        d (float):
            The high redshift power law slope.
        cosmo (astropy.cosmology):
            The cosmology used to convert between age and redshift.
        min_age (float):
            The age below which the star formation history is truncated.
        max_age (float):
            The age above which the star formation history is truncated.
        t_univ (float):
            The age of the universe at the observation redshift (in years).
        finegrid (np.ndarray of float):
            The age grid (in years) on which the SFH is stored.
        intsfh (np.ndarray of float):
            The SFH evaluated on finegrid.
    """

    def __init__(
        self,
        redshift,
        b,
        c,
        d,
        cosmo,
        min_age=0 * yr,
        max_age=1e11 * yr,
        n_grid=1000,
    ):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            redshift (float):
                The redshift at which the SFH is observed.
            b (float):
                The low redshift power law slope.
            c (float):
                The turnover in (1 + z).
            d (float):
                The high redshift power law slope.
            cosmo (astropy.cosmology):
                The cosmology used to relate stellar age and redshift.
                MD+14 assume a flat h = 0.7 and Omega_m = 0.3 cosmology.
            min_age (unyt_quantity):
                The age below which the SFH is truncated.
            max_age (unyt_quantity):
                The age above which the SFH is truncated.
            n_grid (int):
                The number of points in the redshift grid used to build the
                age-space SFH.
        """
        # Initialise the parent.
        Common.__init__(
            self,
            name="MadauDickinsonCSFH",
            redshift=redshift,
            b=b,
            c=c,
            d=d,
            cosmo=cosmo,
            min_age=min_age,
            max_age=max_age,
            n_grid=n_grid,
        )

        # Set the model parameters.
        self.redshift = redshift
        self.b = b
        self.c = c
        self.d = d
        self.cosmo = cosmo
        self.min_age = min_age.to("yr").value
        self.max_age = max_age.to("yr").value

        # The age of the universe at the observation redshift.
        self.t_univ = cosmo.age(redshift).to("yr").value

        # Evaluate the SFRD on a grid log spaced in 1 + z.
        z = (
            np.logspace(np.log10(1.0 + redshift), np.log10(101.0), n_grid)
            - 1.0
        )
        sfrd = (1.0 + z) ** b / (1.0 + ((1.0 + z) / c) ** d)

        # Convert redshift to age.
        self.finegrid = (self.t_univ - cosmo.age(z).to("yr").value).astype(
            np.float64
        )
        self.intsfh = sfrd.astype(np.float64)

    def _sfr(self, age):
        """Get the SFR at a given stellar age.

        Args:
            age (float):
                The stellar age (in years) at which to evaluate the SFR.

        Returns:
            float:
                The SFR at the passed age (zero outside the grid and outside
                the min_age to max_age range).
        """
        if (age >= self.min_age) & (age < self.max_age):
            return np.interp(
                age, self.finegrid, self.intsfh, left=0.0, right=0.0
            )

        return 0.0

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The stellar ages (in years) at which to evaluate the SFR.

        Returns:
            np.ndarray of float:
                The SFR at each age.
        """
        ages = np.asarray(ages, dtype=np.float64)
        sfrs = np.interp(ages, self.finegrid, self.intsfh, left=0.0, right=0.0)
        mask = (ages >= self.min_age) & (ages < self.max_age)

        return np.where(mask, sfrs, 0.0)
