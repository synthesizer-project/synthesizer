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
from unyt import c, Angstrom


class Feltre16:

    """
    A class to hold routines for employing the Feltre16 AGN model.
    """

    def __init__(self):
        return None

    def incident(self, lam, alpha, luminosity=1):
        """
        Create intrinsic narrow-line AGN spectra as utilised by
        Feltre et al. (2016). This is utilised to build the cloudy grid.

        Parameters
        ----------
        lam : array
            Wavelength grid (array) in angstrom or unyt

        alpha: float
            UV/optical power-law index. Expected to be -2.0<alpha<-1.2

        luminosity: float
            Bolometric luminosity. Set to unity.


        Returns
        -------
        lnu
            Spectral luminosity density.
        """

        # create empty luminosity array
        lnu = np.zeros(lam.shape)

        # calculate frequency
        nu = c / lam

        # define edges
        edges = [10.0, 2500.0, 100000.0, 1000000.0] * Angstrom  # Angstrom

        # define indices
        indices = [alpha, -0.5, 2.0]

        # define normalisations
        norms = [
            1.0,
        ]

        # calcualte remaining normalisations
        for i, (edge_lam, ind1, ind2) in enumerate(
            zip(edges[1:], indices, indices[1:])
        ):
            edge_nu = c / edge_lam

            norm = norms[i] * (edge_nu**ind1) / (edge_nu**ind2)
            norms.append(norm)

        # now construct spectra
        for e1, e2, ind, norm in zip(edges[0:], edges[1:], indices, norms):
            # identify indices within the wavelength range
            s = (lam >= e1) & (lam < e2)

            lnu[s] = norm * nu[s] ** ind

        # normalise -- not yet implemented

        return lnu
